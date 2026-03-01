# Chrome Bridge (Native Messaging)

This extension is a starter architecture for Chrome parity with a native messaging host.

## What It Adds

- MV3 extension shell (`manifest.json`, `background.js`, `popup.html`, `popup.js`)
- Native messaging host template (`native-messaging/com.codingbuddy.cli.template.json`)
- Installer script (`native-messaging/install_native_host.sh`)
- Host wrapper script (`native-messaging/codingbuddy_native_host.sh`) that runs:

```bash
deepseek native-host
```

## Install (Developer)

1. Load `extensions/chrome` as an unpacked extension in Chrome.
2. Copy the extension ID from `chrome://extensions`.
3. Install host manifest:

```bash
cd extensions/chrome/native-messaging
./install_native_host.sh <extension-id> /absolute/path/to/deepseek
```

Optional browser selection:

```bash
BROWSER=chromium ./install_native_host.sh <extension-id> /absolute/path/to/deepseek
```

## Runtime Flow

1. Extension service worker uses `chrome.runtime.connectNative("com.codingbuddy.cli")`.
2. Native host launches `deepseek native-host`.
3. Host bridges Chrome native messaging frames to CodingBuddy JSON-RPC handler.
4. Extension sends JSON-RPC requests (`status`, `session/open`, `session/list`, `prompt/execute`, etc.).

## Notes

- This is a parity-focused foundation, not a published store extension.
- Keep extension and host IDs aligned (`com.codingbuddy.cli`).
