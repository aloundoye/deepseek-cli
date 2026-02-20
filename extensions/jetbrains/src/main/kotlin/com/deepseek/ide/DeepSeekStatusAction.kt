package com.deepseek.ide

import com.intellij.notification.NotificationGroupManager
import com.intellij.notification.NotificationType
import com.intellij.openapi.actionSystem.AnAction
import com.intellij.openapi.actionSystem.AnActionEvent
import com.intellij.openapi.application.ApplicationManager

class DeepSeekStatusAction : AnAction("DeepSeek Status") {
    override fun actionPerformed(event: AnActionEvent) {
        val project = event.project
        ApplicationManager.getApplication().executeOnPooledThread {
            val response = try {
                DeepSeekRpcClient.shared.status()
            } catch (error: Throwable) {
                "DeepSeek status failed: ${error.message}"
            }

            ApplicationManager.getApplication().invokeLater {
                NotificationGroupManager.getInstance()
                    .getNotificationGroup("DeepSeek")
                    .createNotification(response, NotificationType.INFORMATION)
                    .notify(project)
            }
        }
    }

    override fun update(event: AnActionEvent) {
        event.presentation.isEnabledAndVisible = true
    }
}
