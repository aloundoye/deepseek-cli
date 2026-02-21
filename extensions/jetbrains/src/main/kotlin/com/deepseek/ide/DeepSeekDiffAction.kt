package com.deepseek.ide

import com.intellij.diff.DiffContentFactory
import com.intellij.diff.DiffManager
import com.intellij.diff.requests.SimpleDiffRequest
import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.ui.Messages

class DeepSeekDiffAction : AnAction("DeepSeek: View Patch Diff") {
    override fun actionPerformed(event: AnActionEvent) {
        val project = event.project ?: return

        val sessionId = Messages.showInputDialog(
            project,
            "Session ID:",
            "DeepSeek Diff",
            null
        ) ?: return

        val patchId = Messages.showInputDialog(
            project,
            "Patch ID:",
            "DeepSeek Diff",
            null
        ) ?: return

        ApplicationManager.getApplication().executeOnPooledThread {
            val response = try {
                DeepSeekRpcClient.shared.patchPreview(sessionId, patchId)
            } catch (error: Throwable) {
                ApplicationManager.getApplication().invokeLater {
                    NotificationGroupManager.getInstance()
                        .getNotificationGroup("DeepSeek")
                        .createNotification("Diff error: ${error.message}", NotificationType.ERROR)
                        .notify(project)
                }
                return@executeOnPooledThread
            }

            ApplicationManager.getApplication().invokeLater {
                val factory = DiffContentFactory.getInstance()
                val left = factory.create(project, "Original content (patch: $patchId)")
                val right = factory.create(project, "Preview: $response")
                val request = SimpleDiffRequest("DeepSeek Patch Preview", left, right, "Before", "After")
                DiffManager.getInstance().showDiff(project, request)
            }
        }
    }

    override fun update(event: AnActionEvent) {
        event.presentation.isEnabledAndVisible = event.project != null
    }
}
