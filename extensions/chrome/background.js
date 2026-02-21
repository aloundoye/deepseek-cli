const DEFAULT_HOST = "com.deepseek.cli";

let port = null;
let nextId = 1;
const pending = new Map();

async function hostName() {
  const stored = await chrome.storage.local.get(["deepseekNativeHost"]);
  const value = stored.deepseekNativeHost;
  if (typeof value === "string" && value.trim().length > 0) {
    return value.trim();
  }
  return DEFAULT_HOST;
}

async function connectPort() {
  if (port) {
    return port;
  }

  const host = await hostName();
  port = chrome.runtime.connectNative(host);

  port.onMessage.addListener((message) => {
    const id = message && message.id;
    if (id === undefined || id === null || !pending.has(id)) {
      return;
    }
    const waiter = pending.get(id);
    pending.delete(id);
    waiter.resolve(message);
  });

  port.onDisconnect.addListener(() => {
    const reason = chrome.runtime.lastError
      ? chrome.runtime.lastError.message
      : "native host disconnected";
    for (const [, waiter] of pending) {
      waiter.reject(new Error(reason));
    }
    pending.clear();
    port = null;
  });

  return port;
}

async function rpc(method, params = {}) {
  const activePort = await connectPort();
  const id = nextId++;
  const payload = {
    jsonrpc: "2.0",
    id,
    method,
    params
  };

  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    try {
      activePort.postMessage(payload);
    } catch (error) {
      pending.delete(id);
      reject(error);
    }
  });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "deepseek.rpc") {
    return false;
  }

  rpc(message.method, message.params || {})
    .then((response) => sendResponse({ ok: true, response }))
    .catch((error) => {
      sendResponse({
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      });
    });

  return true;
});

chrome.runtime.onInstalled.addListener(() => {
  console.log("DeepSeek bridge installed");
});
