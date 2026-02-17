#!/usr/bin/env bash
set -euo pipefail

VERSION="latest"
REPO="${DEEPSEEK_REPO:-aloutndoye/deepseek-cli}"
INSTALL_DIR="${DEEPSEEK_INSTALL_DIR:-}"
TARGET="${DEEPSEEK_TARGET:-}"
DRY_RUN=0

usage() {
  cat <<USAGE
Install DeepSeek CLI from GitHub releases.

Usage: scripts/install.sh [--version <tag|latest>] [--repo <owner/repo>] [--install-dir <dir>] [--target <triple>] [--dry-run]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --repo)
      REPO="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$INSTALL_DIR" ]]; then
  if [[ -w "/usr/local/bin" ]]; then
    INSTALL_DIR="/usr/local/bin"
  else
    INSTALL_DIR="$HOME/.local/bin"
  fi
fi

os="$(uname -s)"
arch="$(uname -m)"

if [[ -z "$TARGET" ]]; then
  case "$os" in
    Linux)
      case "$arch" in
        x86_64|amd64)
          TARGET="x86_64-unknown-linux-gnu"
          ;;
        arm64|aarch64)
          TARGET="aarch64-unknown-linux-gnu"
          ;;
        *)
          echo "unsupported architecture for prebuilt binaries: $arch (linux). build from source or pass --target." >&2
          exit 1
          ;;
      esac
      ;;
    Darwin)
      case "$arch" in
        x86_64|amd64)
          TARGET="x86_64-apple-darwin"
          ;;
        arm64|aarch64)
          TARGET="aarch64-apple-darwin"
          ;;
        *)
          echo "unsupported architecture for prebuilt binaries: $arch (darwin). build from source or pass --target." >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "unsupported OS: $os" >&2
      exit 1
      ;;
  esac
fi

asset="deepseek-${TARGET}.tar.gz"
if [[ "$VERSION" == "latest" ]]; then
  base_url="https://github.com/${REPO}/releases/latest/download"
else
  base_url="https://github.com/${REPO}/releases/download/${VERSION}"
fi
asset_url="${base_url}/${asset}"
checksum_url="${base_url}/checksums.txt"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry-run"
  echo "repo: ${REPO}"
  echo "version: ${VERSION}"
  echo "target: ${TARGET}"
  echo "asset: ${asset_url}"
  echo "install_dir: ${INSTALL_DIR}"
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

curl -fsSL "$asset_url" -o "$tmp_dir/$asset"
if curl -fsSL "$checksum_url" -o "$tmp_dir/checksums.txt"; then
  expected="$(awk -v a="$asset" '$2==a {print $1}' "$tmp_dir/checksums.txt")"
  if [[ -n "$expected" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      actual="$(sha256sum "$tmp_dir/$asset" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      actual="$(shasum -a 256 "$tmp_dir/$asset" | awk '{print $1}')"
    else
      actual=""
      echo "warning: no sha256 tool found; skipping checksum verification" >&2
    fi
    if [[ -n "$actual" && "$actual" != "$expected" ]]; then
      echo "checksum verification failed" >&2
      echo "expected: $expected" >&2
      echo "actual:   $actual" >&2
      exit 1
    fi
  fi
fi

tar -xzf "$tmp_dir/$asset" -C "$tmp_dir"

bin_path="$tmp_dir/deepseek"
if [[ ! -f "$bin_path" ]]; then
  bin_path="$(find "$tmp_dir" -type f -name deepseek | head -n 1)"
fi
if [[ -z "$bin_path" || ! -f "$bin_path" ]]; then
  echo "unable to locate extracted deepseek binary" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
install -m 0755 "$bin_path" "$INSTALL_DIR/deepseek"

echo "Installed deepseek to $INSTALL_DIR/deepseek"
echo "If needed, add to PATH: export PATH=\"$INSTALL_DIR:\$PATH\""
echo "Uninstall: rm \"$INSTALL_DIR/deepseek\""
