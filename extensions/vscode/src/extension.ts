import * as vscode from "vscode";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";

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

const rpcClient = new DeepSeekRpcClient();

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  context.subscriptions.push(
    vscode.commands.registerCommand("deepseek.startServer", async () => {
      await rpcClient.start();
      vscode.window.showInformationMessage("DeepSeek RPC server started.");
    })
  );

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

  context.subscriptions.push({
    dispose: () => rpcClient.stop()
  });
}

export function deactivate(): void {
  rpcClient.stop();
}
