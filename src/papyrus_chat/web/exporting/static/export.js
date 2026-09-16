import { readSnapshot } from "./export-storage.js";

// A shadow root keeps the add-on independent of the stock React tree and styles.
const host = document.createElement("papyrus-export");
const root = host.attachShadow({ mode: "open" });
root.innerHTML = `
  <link rel="stylesheet" href="/papyrus-assets/export.css">
  <div class="toolbar"><span class="brand">Papyrus Chat</span>
    <button class="launcher" type="button" aria-haspopup="dialog" aria-label="Export conversation">Export</button></div>
  <dialog aria-labelledby="export-heading" aria-describedby="export-description">
    <header><h2 id="export-heading">Export conversation</h2>
      <button class="close" type="button" aria-label="Close export panel">×</button></header>
    <p id="export-description">Download the latest browser-saved snapshot, including reasoning and tool activity.
      If the agent is still responding, the snapshot may be incomplete.</p>
    <p class="conversation"></p>
    <div class="downloads"><button type="button" data-format="json">Download JSON</button></div>
    <p class="status" role="status" aria-live="polite"></p>
  </dialog>`;
document.body.append(host);
const dialog = root.querySelector("dialog");
const launcher = root.querySelector(".launcher");
const status = root.querySelector(".status");
const title = root.querySelector(".conversation");
const buttons = [...root.querySelectorAll("[data-format]")];
let conversationId = null;
let busy = false;
let generation = 0;

function setBusy(value) {
  busy = value;
  buttons.forEach(button => { button.disabled = value || !conversationId; });
  dialog.setAttribute("aria-busy", String(value));
}

async function openPanel() {
  const currentGeneration = ++generation;
  conversationId = null;
  title.textContent = "";
  status.textContent = "Loading saved conversation…";
  dialog.showModal();
  setBusy(true);
  try {
    const snapshot = await readSnapshot(window.location.pathname);
    if (generation !== currentGeneration) return;
    conversationId = snapshot.id;
    title.textContent = snapshot.title;
    status.textContent = "Ready to download.";
  } catch (error) {
    if (generation === currentGeneration) status.textContent = error.message;
  } finally {
    if (generation === currentGeneration) setBusy(false);
  }
}

async function download(format) {
  if (busy || !conversationId) return;
  const currentGeneration = generation;
  const selectedId = conversationId;
  setBusy(true);
  status.textContent = "Preparing download…";
  try {
    // Read again at click time: the agent may have saved more content since opening.
    const conversation = await readSnapshot(selectedId);
    const response = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format, conversation }),
    });
    if (!response.ok) {
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.error || "Export failed. Please retry.");
    }
    const blob = await response.blob();
    if (generation !== currentGeneration) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = response.headers.get("Content-Disposition")?.match(/filename="([^"]+)"/)?.[1]
      || `papyrus-conversation.${format}`;
    root.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Download ready.";
  } catch (error) {
    if (generation === currentGeneration) status.textContent = error.message || "Export failed. Please retry.";
  } finally {
    if (generation === currentGeneration) setBusy(false);
  }
}

launcher.addEventListener("click", openPanel);
root.querySelector(".close").addEventListener("click", () => dialog.close());
dialog.addEventListener("close", () => {
  generation++;
  conversationId = null;
  setBusy(false);
  launcher.focus();
});
buttons.forEach(button => button.addEventListener("click", () => download(button.dataset.format)));
// The pinned UI emits this event for same-tab navigation; back/forward uses popstate.
for (const event of ["popstate", "history-state-changed"]) {
  window.addEventListener(event, () => { if (dialog.open) dialog.close(); });
}
