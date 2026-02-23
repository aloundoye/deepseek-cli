package com.deepseek.ide

import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.project.Project
import com.intellij.openapi.wm.ToolWindow
import com.intellij.openapi.wm.ToolWindowFactory
import com.intellij.ui.jcef.JBCefBrowser
import com.intellij.ui.jcef.JBCefJSQuery
import org.cef.browser.CefBrowser
import org.cef.browser.CefFrame
import org.cef.handler.CefLoadHandlerAdapter
import java.util.concurrent.atomic.AtomicBoolean
import javax.swing.JPanel
import java.awt.BorderLayout

class DeepSeekToolWindowFactory : ToolWindowFactory {

    override fun createToolWindowContent(project: Project, toolWindow: ToolWindow) {
        val panel = JPanel(BorderLayout())
        val browser = JBCefBrowser()
        val workspaceRoot = project.basePath ?: "."

        val sendQuery = JBCefJSQuery.create(browser)
        val sessionHolder = arrayOfNulls<String>(1) // [sessionId]
        val streaming = AtomicBoolean(false)

        sendQuery.addHandler { prompt ->
            ApplicationManager.getApplication().executeOnPooledThread {
                try {
                    if (sessionHolder[0] == null) {
                        val resp = DeepSeekRpcClient.shared.sessionOpen(workspaceRoot)
                        val match = """"session_id"\s*:\s*"([^"]+)"""".toRegex().find(resp)
                        sessionHolder[0] = match?.groupValues?.get(1)
                    }
                    val sid = sessionHolder[0] ?: return@executeOnPooledThread

                    // Execute prompt with partial messages enabled
                    val resp = DeepSeekRpcClient.shared.request(
                        "prompt/execute",
                        """{"session_id":"$sid","prompt":${escapeJson(prompt)},"include_partial_messages":true}"""
                    )

                    // Extract prompt_id for streaming
                    val promptIdMatch = """"prompt_id"\s*:\s*"([^"]+)"""".toRegex().find(resp)
                    val promptId = promptIdMatch?.groupValues?.get(1)

                    if (promptId != null) {
                        streaming.set(true)
                        streamChunks(browser, promptId, sid)
                    } else {
                        // Fallback: extract output directly
                        val outputMatch = """"output"\s*:\s*"([^"]*?)"""".toRegex().find(resp)
                        val output = outputMatch?.groupValues?.get(1) ?: resp
                        callJs(browser, "window.onChunk(${escapeJson(output)})")
                        callJs(browser, "window.onDone()")
                    }
                } catch (e: Throwable) {
                    callJs(browser, "window.onError(${escapeJson(e.message ?: "unknown error")})")
                }
            }
            null
        }

        browser.jbCefClient.addLoadHandler(object : CefLoadHandlerAdapter() {
            override fun onLoadEnd(cefBrowser: CefBrowser?, frame: CefFrame?, httpStatusCode: Int) {
                if (frame?.isMain == true) {
                    val injected = sendQuery.inject("prompt")
                    callJs(browser, """
                        window.sendToPlugin = function(prompt) {
                            $injected
                        };
                    """.trimIndent())
                }
            }
        }, browser.cefBrowser)

        val htmlUrl = javaClass.getResource("/webview/chat.html")
        if (htmlUrl != null) {
            browser.loadURL(htmlUrl.toExternalForm())
        } else {
            browser.loadHTML(FALLBACK_HTML)
        }

        panel.add(browser.component, BorderLayout.CENTER)
        val content = toolWindow.contentManager.factory.createContent(panel, "Chat", false)
        toolWindow.contentManager.addContent(content)
    }

    private fun streamChunks(browser: JBCefBrowser, promptId: String, sessionId: String) {
        var cursor = 0
        var done = false
        while (!done) {
            val resp = DeepSeekRpcClient.shared.request(
                "prompt/stream_next",
                """{"prompt_id":"$promptId","cursor":$cursor,"max_chunks":16}"""
            )

            // Parse chunks array from response
            val chunksMatch = """"chunks"\s*:\s*\[([^\]]*)]""".toRegex().find(resp)
            val doneMatch = """"done"\s*:\s*(true|false)""".toRegex().find(resp)
            val cursorMatch = """"next_cursor"\s*:\s*(\d+)""".toRegex().find(resp)

            if (chunksMatch != null) {
                val chunksJson = chunksMatch.groupValues[1]
                // Extract content_delta text from each chunk
                val deltaPattern = """"content_delta"\s*:\s*"((?:[^"\\]|\\.)*)"""".toRegex()
                for (delta in deltaPattern.findAll(chunksJson)) {
                    val text = delta.groupValues[1]
                        .replace("\\n", "\n")
                        .replace("\\\"", "\"")
                        .replace("\\\\", "\\")
                    callJs(browser, "window.onChunk(${escapeJson(text)})")
                }
            }

            done = doneMatch?.groupValues?.get(1) == "true"
            cursor = cursorMatch?.groupValues?.get(1)?.toIntOrNull() ?: cursor

            if (!done) {
                Thread.sleep(50)
            }
        }
        callJs(browser, "window.onDone()")
    }

    private fun callJs(browser: JBCefBrowser, js: String) {
        ApplicationManager.getApplication().invokeLater {
            browser.cefBrowser.executeJavaScript(js, browser.cefBrowser.url, 0)
        }
    }

    companion object {
        fun escapeJson(s: String): String {
            val escaped = s
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            return "\"$escaped\""
        }

        private const val FALLBACK_HTML = """
            <html><body style="font-family:sans-serif;padding:16px;">
            <p>Could not load chat UI. Ensure <code>/webview/chat.html</code> is bundled.</p>
            </body></html>
        """
    }
}
