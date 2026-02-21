import * as vscode from "vscode";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import * as path from "path";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params: unknown;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: {
    code: number;
    message: string;
    data?: unknown;
  };
}

class DeepSeekRpcClient {
  private process: ChildProcessWithoutNullStreams | null = null;
  private nextId = 1;
  private readonly pending = new Map<number, { resolve: (v: JsonRpcResponse) => void; reject: (e: Error) => void }>();
  private readonly output = vscode.window.createOutputChannel("DeepSeek RPC");
  private readBuffer = "";

  async start(): Promise<void> {
    if (this.process) {
      return;
    }
    const cfg = vscode.workspace.getConfiguration("deepseek.ide");
    const binaryPath = cfg.get<string>("binaryPath", "deepseek");
    this.output.appendLine(`Starting ${binaryPath} serve --transport stdio`);

    this.process = spawn(binaryPath, ["serve", "--transport", "stdio"], {
      stdio: "pipe"
    });

    this.process.stdout.on("data", (chunk: Buffer) => this.onStdout(chunk.toString("utf8")));
    this.process.stderr.on("data", (chunk: Buffer) => this.output.appendLine(`[stderr] ${chunk.toString("utf8").trim()}`));
    this.process.on("exit", (code, signal) => {
      this.output.appendLine(`RPC process exited code=${code} signal=${signal}`);
      this.process = null;
      for (const [id, waiter] of this.pending.entries()) {
        waiter.reject(new Error(`RPC process exited before response for id=${id}`));
      }
      this.pending.clear();
    });

    await this.request("initialize", {
      client: "vscode",
      version: vscode.version
    });
  }

  async status(): Promise<JsonRpcResponse> {
    return this.request("status", {});
  }

  async sessionOpen(workspaceRoot: string): Promise<JsonRpcResponse> {
    return this.request("session/open", { workspace_root: workspaceRoot });
  }

  async getFileSuggestions(query: string, limit: number = 10): Promise<JsonRpcResponse> {
    return this.request("context/suggest", { query, limit });
  }

  async analyzeImports(filePath: string): Promise<JsonRpcResponse> {
    return this.request("context/analyze", { file_path: filePath });
  }

  async sessionResume(sessionId: string): Promise<JsonRpcResponse> {
    return this.request("session/resume", { session_id: sessionId });
  }

  async sessionFork(sessionId: string): Promise<JsonRpcResponse> {
    return this.request("session/fork", { session_id: sessionId });
  }

  async sessionList(): Promise<JsonRpcResponse> {
    return this.request("session/list", {});
  }

  async promptExecute(sessionId: string, prompt: string): Promise<JsonRpcResponse> {
    return this.request("prompt/execute", { session_id: sessionId, prompt });
  }

  async toolApprove(sessionId: string, invocationId: string): Promise<JsonRpcResponse> {
    return this.request("tool/approve", { session_id: sessionId, invocation_id: invocationId });
  }

  async toolDeny(sessionId: string, invocationId: string): Promise<JsonRpcResponse> {
    return this.request("tool/deny", { session_id: sessionId, invocation_id: invocationId });
  }

  async patchPreview(sessionId: string, patchId: string): Promise<JsonRpcResponse> {
    return this.request("patch/preview", { session_id: sessionId, patch_id: patchId });
  }

  async patchApply(sessionId: string, patchId: string): Promise<JsonRpcResponse> {
    return this.request("patch/apply", { session_id: sessionId, patch_id: patchId });
  }

  async request(method: string, params: unknown): Promise<JsonRpcResponse> {
    await this.start();
    if (!this.process) {
      throw new Error("DeepSeek RPC process is not running");
    }
    const id = this.nextId++;
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      id,
      method,
      params
    };
    const line = `${JSON.stringify(request)}\n`;

    return new Promise<JsonRpcResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.process?.stdin.write(line, (error) => {
        if (error) {
          this.pending.delete(id);
          reject(error);
        }
      });
    });
  }

  stop(): void {
    if (!this.process) {
      return;
    }
    this.process.kill();
    this.process = null;
  }

  private onStdout(data: string): void {
    this.readBuffer += data;
    while (true) {
      const newline = this.readBuffer.indexOf("\n");
      if (newline < 0) {
        break;
      }
      const line = this.readBuffer.slice(0, newline).trim();
      this.readBuffer = this.readBuffer.slice(newline + 1);
      if (!line) {
        continue;
      }

      let response: JsonRpcResponse;
      try {
        response = JSON.parse(line) as JsonRpcResponse;
      } catch (error) {
        this.output.appendLine(`Invalid JSON-RPC line: ${line}`);
        continue;
      }

      const waiter = this.pending.get(response.id);
      if (!waiter) {
        this.output.appendLine(`Unmatched response id=${response.id}: ${line}`);
        continue;
      }
      this.pending.delete(response.id);
      waiter.resolve(response);
    }
  }
}

// ---------------------------------------------------------------------------
// Chat Panel (webview-based)
// ---------------------------------------------------------------------------

interface ChatEntry {
  role: "user" | "assistant" | "tool";
  content: string;
}

class DeepSeekChatPanel {
  private static panels = new Map<string, DeepSeekChatPanel>();
  private readonly panel: vscode.WebviewPanel;
  private readonly client: DeepSeekRpcClient;
  private sessionId: string | null = null;
  private history: ChatEntry[] = [];
  private autoAccept: boolean;

  static create(
    extensionUri: vscode.Uri,
    client: DeepSeekRpcClient,
    title: string,
    autoAccept: boolean,
  ): DeepSeekChatPanel {
    const panel = vscode.window.createWebviewPanel(
      "deepseekChat",
      title,
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    const chatPanel = new DeepSeekChatPanel(panel, client, autoAccept);
    DeepSeekChatPanel.panels.set(title, chatPanel);
    panel.onDidDispose(() => DeepSeekChatPanel.panels.delete(title));
    return chatPanel;
  }

  private constructor(
    panel: vscode.WebviewPanel,
    client: DeepSeekRpcClient,
    autoAccept: boolean,
  ) {
    this.panel = panel;
    this.client = client;
    this.autoAccept = autoAccept;
    this.panel.webview.html = this.getHtml();
    this.panel.webview.onDidReceiveMessage((msg) => this.onMessage(msg));
  }

  private async onMessage(msg: { type: string; text?: string; invocationId?: string; patchId?: string }): Promise<void> {
    switch (msg.type) {
      case "send": {
        if (!msg.text) return;
        await this.ensureSession();
        this.addEntry({ role: "user", content: msg.text });
        try {
          const resp = await this.client.promptExecute(this.sessionId!, msg.text);
          if (resp.error) {
            this.addEntry({ role: "assistant", content: `Error: ${resp.error.message}` });
          } else {
            const result = resp.result as { prompt_id: string; status: string };
            this.addEntry({ role: "assistant", content: `[Prompt queued: ${result.prompt_id}]` });
          }
        } catch (e) {
          this.addEntry({ role: "assistant", content: `Error: ${e instanceof Error ? e.message : String(e)}` });
        }
        break;
      }
      case "approve":
        if (msg.invocationId) {
          await this.client.toolApprove(this.sessionId!, msg.invocationId);
          this.addEntry({ role: "tool", content: `Approved: ${msg.invocationId}` });
        }
        break;
      case "deny":
        if (msg.invocationId) {
          await this.client.toolDeny(this.sessionId!, msg.invocationId);
          this.addEntry({ role: "tool", content: `Denied: ${msg.invocationId}` });
        }
        break;
      case "applyPatch":
        if (msg.patchId) {
          const resp = await this.client.patchApply(this.sessionId!, msg.patchId);
          if (resp.error) {
            this.addEntry({ role: "tool", content: `Patch error: ${resp.error.message}` });
          } else {
            this.addEntry({ role: "tool", content: `Patch applied: ${msg.patchId}` });
          }
        }
        break;
      case "previewPatch":
        if (msg.patchId) {
          const resp = await this.client.patchPreview(this.sessionId!, msg.patchId);
          if (resp.error) {
            this.addEntry({ role: "tool", content: `Preview error: ${resp.error.message}` });
          } else {
            this.addEntry({ role: "tool", content: `Patch preview: ${JSON.stringify(resp.result)}` });
          }
        }
        break;
      case "toggleAutoAccept":
        this.autoAccept = !this.autoAccept;
        break;
      case "atMention": {
        // Get smart file suggestions based on current context
        const editor = vscode.window.activeTextEditor;
        let query = "";
        
        if (editor) {
          // Use current word or selection as query
          const selection = editor.document.getText(editor.selection);
          if (selection.trim().length > 0) {
            query = selection.trim();
          } else {
            // Get word at cursor
            const wordRange = editor.document.getWordRangeAtPosition(editor.selection.active);
            if (wordRange) {
              query = editor.document.getText(wordRange);
            }
          }
        }
        
        // Get file suggestions from DeepSeek
        try {
          const resp = await this.client.getFileSuggestions(query, 10);
          if (!resp.error && resp.result) {
            const suggestions = (resp.result as { suggestions: Array<{ path: string, score: number, reasons: string[] }> }).suggestions;
            
            if (suggestions.length > 0) {
              // Show quick pick with suggestions
              const items = suggestions.map(s => ({
                label: path.basename(s.path),
                description: s.path,
                detail: `Score: ${s.score.toFixed(2)} - ${s.reasons.join(", ")}`,
                path: s.path,
              }));
              
              const selected = await vscode.window.showQuickPick(items, {
                placeHolder: "Select a file to reference",
                matchOnDescription: true,
                matchOnDetail: true,
              });
              
              if (selected) {
                const relativePath = vscode.workspace.asRelativePath(selected.path);
                let mention = `@${relativePath}`;
                
                // Add line range if editor is open on this file
                const editor = vscode.window.activeTextEditor;
                if (editor && editor.document.uri.fsPath === selected.path && !editor.selection.isEmpty) {
                  const startLine = editor.selection.start.line + 1;
                  const endLine = editor.selection.end.line + 1;
                  mention = `@${relativePath}:${startLine}-${endLine}`;
                }
                
                this.panel.webview.postMessage({ type: "insertMention", text: mention });
              }
              break;
            }
          }
        } catch (error) {
          // Fall back to file picker if suggestions fail
          console.warn("Failed to get file suggestions:", error);
        }
        
        // Fallback: Open file picker
        const uris = await vscode.window.showOpenDialog({ canSelectMany: false });
        if (uris && uris.length > 0) {
          const relativePath = vscode.workspace.asRelativePath(uris[0]);
          // Get active editor selection for line range.
          const editor = vscode.window.activeTextEditor;
          let mention = `@${relativePath}`;
          if (editor && editor.document.uri.fsPath === uris[0].fsPath && !editor.selection.isEmpty) {
            const startLine = editor.selection.start.line + 1;
            const endLine = editor.selection.end.line + 1;
            mention = `@${relativePath}:${startLine}-${endLine}`;
          }
          this.panel.webview.postMessage({ type: "insertMention", text: mention });
        }
        break;
      }
    }
  }

  private async ensureSession(): Promise<void> {
    if (this.sessionId) return;
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? ".";
    const resp = await this.client.sessionOpen(workspaceRoot);
    if (resp.result) {
      this.sessionId = (resp.result as { session_id: string }).session_id;
    }
  }

  private addEntry(entry: ChatEntry): void {
    this.history.push(entry);
    this.panel.webview.postMessage({ type: "addEntry", entry });
  }

  private getHtml(): string {
    return `<!DOCTYPE html>
<html>
<head>
  <style>
    :root {
      --ds-radius: 8px;
      --ds-border: color-mix(in srgb, var(--vscode-editor-foreground) 14%, transparent);
      --ds-soft-border: color-mix(in srgb, var(--vscode-editor-foreground) 10%, transparent);
      --ds-muted: color-mix(in srgb, var(--vscode-editor-foreground) 62%, transparent);
      --ds-surface: color-mix(in srgb, var(--vscode-editor-background) 86%, var(--vscode-sideBar-background) 14%);
      --ds-shadow: 0 1px 0 color-mix(in srgb, var(--vscode-editor-foreground) 8%, transparent);
    }
    body {
      font-family: var(--vscode-font-family);
      margin: 0;
      padding: 8px;
      display: flex;
      flex-direction: column;
      height: 100vh;
      gap: 6px;
      background: color-mix(in srgb, var(--vscode-editor-background) 94%, var(--vscode-sideBar-background) 6%);
    }
    #chat {
      flex: 1;
      overflow-y: auto;
      padding: 2px 2px 8px 2px;
      scroll-behavior: smooth;
    }
    .entry {
      margin: 6px 0;
      padding: 8px 10px;
      border-radius: var(--ds-radius);
      border: 1px solid var(--ds-soft-border);
      box-shadow: var(--ds-shadow);
      animation: fadeIn 120ms ease-out;
    }
    .entry-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }
    .role-label {
      font-size: 0.73rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--ds-muted);
      border: 1px solid var(--ds-soft-border);
      border-radius: 999px;
      padding: 1px 8px;
      line-height: 1.5;
    }
    .entry-time {
      font-size: 0.72rem;
      color: var(--ds-muted);
      opacity: 0.9;
    }
    .entry-content {
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.46;
      font-size: 0.85rem;
      color: var(--vscode-editor-foreground);
    }
    .user {
      background: color-mix(in srgb, var(--vscode-input-background) 82%, var(--ds-surface) 18%);
      border-left: 3px solid color-mix(in srgb, var(--vscode-inputOption-activeBorder) 75%, transparent);
    }
    .assistant {
      background: var(--ds-surface);
      border-left: 3px solid color-mix(in srgb, var(--vscode-editorInfo-foreground) 70%, transparent);
    }
    .assistant .entry-content {
      font-size: 0.89rem;
      line-height: 1.52;
    }
    .tool {
      background: color-mix(in srgb, var(--vscode-editor-background) 86%, var(--vscode-editorWidget-background) 14%);
      border-left: 3px solid color-mix(in srgb, var(--vscode-editorWarning-foreground) 75%, transparent);
    }
    .tool .entry-content {
      font-family: var(--vscode-editor-font-family, var(--vscode-font-family));
      font-size: 0.8rem;
      line-height: 1.4;
      opacity: 0.96;
    }
    .inline-tool {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 6px;
      background: color-mix(in srgb, var(--vscode-editorInfo-background) 48%, transparent);
      border: 1px solid color-mix(in srgb, var(--vscode-editorInfo-foreground) 22%, transparent);
      color: color-mix(in srgb, var(--vscode-editorInfo-foreground) 88%, var(--vscode-editor-foreground) 12%);
    }
    .inline-error {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 6px;
      background: color-mix(in srgb, var(--vscode-inputValidation-errorBackground) 42%, transparent);
      border: 1px solid color-mix(in srgb, var(--vscode-inputValidation-errorBorder) 58%, transparent);
      color: color-mix(in srgb, var(--vscode-inputValidation-errorForeground) 88%, var(--vscode-editor-foreground) 12%);
    }
    #input-area {
      display: flex;
      gap: 6px;
    }
    #prompt {
      flex: 1;
      padding: 7px 9px;
      font-size: 14px;
      border: 1px solid var(--vscode-input-border);
      border-radius: 6px;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
    }
    #prompt:focus {
      outline: 1px solid var(--vscode-focusBorder);
      outline-offset: 0;
    }
    button {
      padding: 6px 12px;
      cursor: pointer;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none;
      border-radius: 6px;
      transition: filter 90ms ease-out;
    }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button:active { filter: brightness(0.96); }
    .toolbar {
      display: flex;
      gap: 6px;
      margin-bottom: 2px;
    }
    .toolbar button {
      font-size: 0.82rem;
      padding: 4px 9px;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(2px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="toolbar">
    <button onclick="atMention()">@ File</button>
    <button onclick="toggleAutoAccept()">Auto-Accept: <span id="autoAcceptState">${this.autoAccept ? "ON" : "OFF"}</span></button>
  </div>
  <div id="chat"></div>
  <div id="input-area">
    <input id="prompt" type="text" placeholder="Ask DeepSeek..." />
    <button onclick="send()">Send</button>
  </div>
  <script>
    const vscode = acquireVsCodeApi();
    const chat = document.getElementById("chat");
    const prompt = document.getElementById("prompt");
    const autoAcceptState = document.getElementById("autoAcceptState");

    function send() {
      const text = prompt.value.trim();
      if (!text) return;
      vscode.postMessage({ type: "send", text });
      prompt.value = "";
    }
    prompt.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

    function atMention() { vscode.postMessage({ type: "atMention" }); }
    function toggleAutoAccept() { vscode.postMessage({ type: "toggleAutoAccept" }); }

    function roleName(role) {
      if (role === "assistant") return "Generated";
      if (role === "user") return "You";
      if (role === "tool") return "Tool";
      return role;
    }

    function renderInlineHighlights(role, text) {
      const escaped = escapeHtml(text);
      return escaped
        .split(/\\r?\\n/)
        .map((line) => {
          const trimmed = line.trimStart();
          if (trimmed.startsWith("[tool:")) {
            return '<span class="inline-tool">' + line + '</span>';
          }
          if (/^(error|failed|exception|patch error):/i.test(trimmed)) {
            return '<span class="inline-error">' + line + '</span>';
          }
          return line;
        })
        .join("\\n");
    }

    function addEntry(entry) {
      const div = document.createElement("div");
      div.className = "entry " + entry.role;
      const stamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      div.innerHTML =
        '<div class="entry-header">' +
          '<div class="role-label">' + roleName(entry.role) + '</div>' +
          '<div class="entry-time">' + stamp + '</div>' +
        '</div>' +
        '<div class="entry-content">' + renderInlineHighlights(entry.role, entry.content) + '</div>';
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    window.addEventListener("message", (event) => {
      const msg = event.data;
      if (msg.type === "addEntry") {
        addEntry(msg.entry);
      }
      if (msg.type === "insertMention") {
        prompt.value += msg.text + " ";
        prompt.focus();
      }
      if (msg.type === "autoAcceptState") {
        autoAcceptState.textContent = msg.enabled ? "ON" : "OFF";
      }
    });

    function escapeHtml(s) {
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
  </script>
</body>
</html>`;
  }
}

// ---------------------------------------------------------------------------
// Extension activation
// ---------------------------------------------------------------------------

const rpcClient = new DeepSeekRpcClient();

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  // Start server command.
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.startServer", async () => {
      await rpcClient.start();
      vscode.window.showInformationMessage("DeepSeek RPC server started.");
    })
  );

  // Status command.
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.status", async () => {
      try {
        const response = await rpcClient.status();
        if (response.error) {
          vscode.window.showErrorMessage(`DeepSeek status error: ${response.error.message}`);
          return;
        }
        vscode.window.showInformationMessage(`DeepSeek status: ${JSON.stringify(response.result)}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`DeepSeek status failed: ${message}`);
      }
    })
  );

  // Open chat panel command.
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.openChat", () => {
      const cfg = vscode.workspace.getConfiguration("deepseek.ide");
      const autoAccept = cfg.get<boolean>("autoAcceptEdits", false);
      DeepSeekChatPanel.create(context.extensionUri, rpcClient, "DeepSeek Chat", autoAccept);
    })
  );

  // Session list command.
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.sessionList", async () => {
      try {
        const resp = await rpcClient.sessionList();
        if (resp.error) {
          vscode.window.showErrorMessage(`DeepSeek error: ${resp.error.message}`);
          return;
        }
        const sessions = (resp.result as { sessions: Array<{ session_id: string; workspace_root: string; status: string }> }).sessions;
        if (sessions.length === 0) {
          vscode.window.showInformationMessage("No sessions found.");
          return;
        }
        const items = sessions.map(s => ({
          label: s.session_id,
          description: `${s.workspace_root} (${s.status})`
        }));
        const selected = await vscode.window.showQuickPick(items, { placeHolder: "Select a session to resume" });
        if (selected) {
          await rpcClient.sessionResume(selected.label);
          vscode.window.showInformationMessage(`Resumed session ${selected.label}`);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`DeepSeek session list failed: ${message}`);
      }
    })
  );

  // Fork session command.
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.forkSession", async () => {
      const sessionId = await vscode.window.showInputBox({ prompt: "Session ID to fork" });
      if (!sessionId) return;
      try {
        const resp = await rpcClient.sessionFork(sessionId);
        if (resp.error) {
          vscode.window.showErrorMessage(`Fork error: ${resp.error.message}`);
          return;
        }
        const result = resp.result as { session_id: string };
        vscode.window.showInformationMessage(`Forked session: ${result.session_id}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`Fork failed: ${message}`);
      }
    })
  );

  // Diff viewer command — opens a diff for a file before/after a patch.
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.showDiff", async (leftUri: vscode.Uri, rightUri: vscode.Uri, title: string) => {
      if (leftUri && rightUri) {
        await vscode.commands.executeCommand("vscode.diff", leftUri, rightUri, title ?? "DeepSeek Diff");
      }
    })
  );

  context.subscriptions.push({
    dispose: () => rpcClient.stop()
  });
}

export function deactivate(): void {
  rpcClient.stop();
}
