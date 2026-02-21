package com.deepseek.ide

import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.ui.Messages

class DeepSeekChatAction : AnAction("DeepSeek Chat") {
    private var sessionId: String? = null

    override fun actionPerformed(event: AnActionEvent) {
        val project = event.project
        val prompt = Messages.showInputDialog(
            project,
            "Enter your prompt:",
            "DeepSeek Chat",
            null
        ) ?: return

        ApplicationManager.getApplication().executeOnPooledThread {
            val response = try {
                ensureSession(project?.basePath ?: ".")
                DeepSeekRpcClient.shared.promptExecute(sessionId!!, prompt)
            } catch (error: Throwable) {
                "Error: ${error.message}"
            }

            ApplicationManager.getApplication().invokeLater {
                NotificationGroupManager.getInstance()
                    .getNotificationGroup("DeepSeek")
                    .createNotification(response, NotificationType.INFORMATION)
                    .notify(project)
            }
        }
    }

    private fun ensureSession(workspaceRoot: String) {
        if (sessionId != null) return
        val response = DeepSeekRpcClient.shared.sessionOpen(workspaceRoot)
        // Extract session_id from JSON response — simple string parsing.
        val match = """"session_id"\s*:\s*"([^"]+)"""".toRegex().find(response)
        sessionId = match?.groupValues?.get(1)
    }

    override fun update(event: AnActionEvent) {
        event.presentation.isEnabledAndVisible = true
    }
}
