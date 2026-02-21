#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="$SCRIPT_DIR/com.deepseek.cli.template.json"
LOCAL_HOST_DIR="${HOME}/.deepseek/native-messaging"
LOCAL_HOST_PATH="$LOCAL_HOST_DIR/deepseek_native_host.sh"

EXTENSION_ID="${1:-}"
DEEPSEEK_BIN_INPUT="${2:-deepseek}"
BROWSER="${BROWSER:-chrome}"

if [[ -z "$EXTENSION_ID" ]]; then
  echo "usage: $0 <extension-id> [deepseek-binary]" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin)
    if [[ "$BROWSER" == "chromium" ]]; then
      MANIFEST_DIR="$HOME/Library/Application Support/Chromium/NativeMessagingHosts"
    else
      MANIFEST_DIR="$HOME/Library/Application Support/Google/Chrome/NativeMessagingHosts"
    fi
    ;;
  Linux)
    if [[ "$BROWSER" == "chromium" ]]; then
      MANIFEST_DIR="$HOME/.config/chromium/NativeMessagingHosts"
    else
      MANIFEST_DIR="$HOME/.config/google-chrome/NativeMessagingHosts"
    fi
    ;;
  *)
    echo "unsupported platform for installer: $(uname -s)" >&2
    exit 1
    ;;
esac

mkdir -p "$LOCAL_HOST_DIR"
cat > "$LOCAL_HOST_PATH" <<HOST
#!/usr/bin/env bash
set -euo pipefail
exec "$DEEPSEEK_BIN_INPUT" native-host "\$@"
HOST
chmod +x "$LOCAL_HOST_PATH"

mkdir -p "$MANIFEST_DIR"
MANIFEST_PATH="$MANIFEST_DIR/com.deepseek.cli.json"
sed \
  -e "s|__HOST_PATH__|$LOCAL_HOST_PATH|g" \
  -e "s|__EXTENSION_ID__|$EXTENSION_ID|g" \
  "$TEMPLATE_PATH" > "$MANIFEST_PATH"

echo "Installed native host manifest: $MANIFEST_PATH"
echo "Host executable wrapper: $LOCAL_HOST_PATH"
