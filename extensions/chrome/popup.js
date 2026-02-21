const output = document.getElementById("output");
const workspace = document.getElementById("workspace");

function render(value) {
  output.textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function rpc(method, params = {}) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      {
        type: "deepseek.rpc",
        method,
        params
      },
      (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!response || !response.ok) {
          reject(new Error(response?.error || "native request failed"));
          return;
        }
        resolve(response.response);
      }
    );
  });
}

document.getElementById("status").addEventListener("click", async () => {
  try {
    const response = await rpc("status", {});
    render(response);
  } catch (error) {
    render(error instanceof Error ? error.message : String(error));
  }
});

document.getElementById("list").addEventListener("click", async () => {
  try {
    const response = await rpc("session/list", {});
    render(response);
  } catch (error) {
    render(error instanceof Error ? error.message : String(error));
  }
});

document.getElementById("openSession").addEventListener("click", async () => {
  try {
    const response = await rpc("session/open", {
      workspace_root: workspace.value.trim() || "."
    });
    render(response);
  } catch (error) {
    render(error instanceof Error ? error.message : String(error));
  }
});
