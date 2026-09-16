// Read-only adapter for @pydantic/ai-chat-ui 2.1.0 (chat-storage, schema 1).
// Do not create, upgrade, migrate, or write the stock UI's database here.
function openSavedDatabase() {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open("chat-storage");
    let settled = false;
    const timer = setTimeout(() => fail("Browser storage is busy. Close other chat tabs and retry."), 5000);
    function fail(message) {
      settled = true;
      clearTimeout(timer);
      reject(new Error(message));
    }
    request.onupgradeneeded = () => {
      request.transaction.abort();
      fail("No saved conversation found. Open a conversation before exporting.");
    };
    request.onerror = () => fail("Saved conversations are unavailable. Check browser storage permissions and retry.");
    request.onsuccess = () => {
      const db = request.result;
      clearTimeout(timer);
      if (settled) {
        db.close();
        return;
      }
      if (db.version !== 1 || !["conversations", "messages"].every(name => db.objectStoreNames.contains(name))) {
        db.close();
        fail("This browser storage version is not supported by the exporter.");
        return;
      }
      db.onversionchange = () => db.close();
      resolve(db);
    };
  });
}

export async function readSnapshot(id) {
  if (!id || id === "/") {
    throw new Error("Open a saved conversation before exporting.");
  }
  let db;
  try {
    db = await openSavedDatabase();
  } catch (error) {
    throw new Error(error.message || "Browser storage is unavailable.");
  }
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(["conversations", "messages"], "readonly");
      const conversation = tx.objectStore("conversations").get(id);
      const history = tx.objectStore("messages").get(id);
      tx.onabort = tx.onerror = () => reject(new Error("Could not read the saved conversation. Please retry."));
      tx.oncomplete = () => {
        if (!conversation.result || !Array.isArray(history.result?.messages) || !history.result.messages.length) {
          reject(new Error("No saved messages found for this conversation. Wait for the chat to save and retry."));
          return;
        }
        const messages = history.result.messages;
        const firstQuestion = messages.find(message => message.role === "user")?.parts
          ?.find(part => part.type === "text")?.text;
        resolve({
          id,
          title: conversation.result.firstMessage || firstQuestion || "Papyrus conversation",
          messages,
        });
      };
    });
  } finally {
    db.close();
  }
}
