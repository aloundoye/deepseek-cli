# IDE Extensions

This directory contains starter IDE integrations that communicate with DeepSeek CLI via JSON-RPC over stdio.

## VS Code

Path: `extensions/vscode`

1. Install dependencies:

   ```bash
   cd extensions/vscode
   npm install
   npm run compile
   ```

2. Run the extension in a VS Code Extension Development Host.
3. Use commands:
   - `DeepSeek: Start RPC Server`
   - `DeepSeek: Status`

The extension starts `deepseek serve --transport stdio` and sends JSON-RPC requests.

## JetBrains

Path: `extensions/jetbrains`

1. Open `extensions/jetbrains` in IntelliJ IDEA.
2. Run Gradle task `runIde`.
3. In the spawned IDE, use `Tools -> DeepSeek Status`.

The plugin starts `deepseek serve --transport stdio` and sends JSON-RPC `initialize` and `status` requests.

## Chrome

Path: `extensions/chrome`

This package provides a native messaging bridge architecture:

1. Load `extensions/chrome` as an unpacked extension in Chrome.
2. Install native host manifest:

   ```bash
   cd extensions/chrome/native-messaging
   ./install_native_host.sh <extension-id> /absolute/path/to/deepseek
   ```

3. The host launches:

   ```bash
   deepseek native-host
   ```

See `extensions/chrome/README.md` for details.
