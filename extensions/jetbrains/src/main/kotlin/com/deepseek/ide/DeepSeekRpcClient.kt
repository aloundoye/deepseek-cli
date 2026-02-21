package com.deepseek.ide

import java.io.BufferedReader
import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.util.concurrent.atomic.AtomicLong

class DeepSeekRpcClient private constructor() {
    private var process: Process? = null
    private var reader: BufferedReader? = null
    private var writer: BufferedWriter? = null
    private val requestId = AtomicLong(1)

    @Synchronized
    fun request(method: String, params: String = "{}"): String {
        ensureStarted()
        val id = requestId.getAndIncrement()
        val payload = """{"jsonrpc":"2.0","id":$id,"method":"$method","params":$params}"""
        writer?.write(payload)
        writer?.newLine()
        writer?.flush()
        return reader?.readLine() ?: """{"error":"no_response"}"""
    }

    @Synchronized
    fun status(): String = request("status")

    @Synchronized
    fun sessionOpen(workspaceRoot: String): String =
        request("session/open", """{"workspace_root":"$workspaceRoot"}""")

    @Synchronized
    fun sessionResume(sessionId: String): String =
        request("session/resume", """{"session_id":"$sessionId"}""")

    @Synchronized
    fun sessionFork(sessionId: String): String =
        request("session/fork", """{"session_id":"$sessionId"}""")

    @Synchronized
    fun sessionList(): String = request("session/list")

    @Synchronized
    fun promptExecute(sessionId: String, prompt: String): String {
        val escapedPrompt = prompt.replace("\\", "\\\\").replace("\"", "\\\"")
        return request("prompt/execute", """{"session_id":"$sessionId","prompt":"$escapedPrompt"}""")
    }

    @Synchronized
    fun toolApprove(sessionId: String, invocationId: String): String =
        request("tool/approve", """{"session_id":"$sessionId","invocation_id":"$invocationId"}""")

    @Synchronized
    fun toolDeny(sessionId: String, invocationId: String): String =
        request("tool/deny", """{"session_id":"$sessionId","invocation_id":"$invocationId"}""")

    @Synchronized
    fun patchPreview(sessionId: String, patchId: String): String =
        request("patch/preview", """{"session_id":"$sessionId","patch_id":"$patchId"}""")

    @Synchronized
    fun patchApply(sessionId: String, patchId: String): String =
        request("patch/apply", """{"session_id":"$sessionId","patch_id":"$patchId"}""")

    @Synchronized
    fun shutdown() {
        try {
            request("shutdown")
        } catch (_: Exception) { }
        process?.destroy()
        process = null
        reader = null
        writer = null
    }

    @Synchronized
    private fun ensureStarted() {
        if (process != null) {
            return
        }
        val pb = ProcessBuilder("deepseek", "serve", "--transport", "stdio")
        pb.redirectErrorStream(true)
        process = pb.start()
        reader = BufferedReader(InputStreamReader(process!!.inputStream))
        writer = BufferedWriter(OutputStreamWriter(process!!.outputStream))

        val initId = requestId.getAndIncrement()
        val initialize = """{"jsonrpc":"2.0","id":$initId,"method":"initialize","params":{"client":"jetbrains"}}"""
        writer?.write(initialize)
        writer?.newLine()
        writer?.flush()
        reader?.readLine()
    }

    companion object {
        val shared: DeepSeekRpcClient by lazy { DeepSeekRpcClient() }
    }
}
