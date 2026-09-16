"""Real IndexedDB and download tests; opt in with pytest -m browser."""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import expect, sync_playwright
from pydantic_ai.models.test import TestModel

from papyrus_chat.web.application import load_app

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=os.environ.get("PAPYRUS_BROWSER_CHANNEL"))
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def live_url(corpus_artifact, tmp_path_factory, request):
    html = tmp_path_factory.mktemp("export-ui") / "index.html"
    html.write_text(
        '<!doctype html><html><head><meta name="viewport" content="width=device-width">'
        '<link rel="icon" href="data:,"></head><body><main id="root">Chat</main></body></html>'
    )
    app = load_app(
        corpus_artifact,
        env={"LLM_BASE_URL": "https://provider.example/v1", "LLM_MODEL": "research-model"},
        model=TestModel(call_tools=[], custom_output_text="Unused"),
        html_source=None if getattr(request, "param", False) else html,
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)


@pytest.fixture
def page(browser, live_url):
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{live_url}/first")
    yield page
    context.close()


def seed(page, *, version=1, stores=("conversations", "messages")):
    page.evaluate(
        """async ({version, stores}) => {
          await new Promise((resolve, reject) => {
            const request = indexedDB.open('chat-storage', version);
            request.onupgradeneeded = () => {
              for (const name of stores) request.result.createObjectStore(name, {keyPath: 'id'});
            };
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
              const db = request.result;
              const tx = db.transaction(stores, 'readwrite');
              for (const id of ['/first', '/second']) {
                if (stores.includes('conversations')) tx.objectStore('conversations').put({
                  id, firstMessage: `Question ${id}`, timestamp: 1
                });
                if (stores.includes('messages')) tx.objectStore('messages').put({id, messages: [
                  {id: 'q', role: 'user', parts: [{type: 'text', text: `Question ${id}`}]},
                  {id: 'a', role: 'assistant', parts: [
                    {type: 'reasoning', text: 'Inspect πάπυρος'},
                    {type: 'tool-inspect_documents', toolCallId: 'inspect',
                     state: 'output-available',
                     input: {id}, output: {text: 'πάπυρος '.repeat(10000)}},
                    {type: 'text', text: `Answer ${id}`}
                  ]}
                ]});
              }
              tx.oncomplete = () => { db.close(); resolve(); };
              tx.onerror = () => reject(tx.error);
            };
          });
        }""",
        {"version": version, "stores": list(stores)},
    )


def download_json(page):
    with page.expect_download() as saved:
        page.get_by_role("button", name="Download JSON").click()
    download = saved.value
    assert download.suggested_filename.endswith(".json")
    return json.loads(Path(download.path()).read_text())


def test_existing_threads_reload_navigation_and_complete_outputs(page, live_url):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    seed(page)
    page.reload()
    page.get_by_role("button", name="Export conversation").click()
    expect(page.get_by_role("dialog")).to_be_visible()
    first = download_json(page)["conversation"]
    assert first["id"] == "/first"
    assert first["messages"][1]["parts"][1]["output"]["text"] == "πάπυρος " * 10_000
    page.keyboard.press("Escape")
    expect(page.get_by_role("button", name="Export conversation")).to_be_focused()
    page.evaluate(
        "history.pushState({}, '', '/second'); dispatchEvent(new Event('history-state-changed'))"
    )
    page.get_by_role("button", name="Export conversation").click()
    second = download_json(page)["conversation"]
    assert second["id"] == "/second"
    assert second["messages"][0]["parts"][0]["text"] == "Question /second"
    page.goto(f"{live_url}/first")
    page.get_by_role("button", name="Export conversation").click()
    assert download_json(page)["conversation"] == first
    assert not errors


def test_download_rereads_snapshot_and_does_not_write_storage(page):
    seed(page)
    page.get_by_role("button", name="Export conversation").click()
    expect(page.get_by_role("button", name="Download JSON")).to_be_enabled()
    page.evaluate("""async () => {
      const db = await new Promise(resolve => {
        const request = indexedDB.open('chat-storage');
        request.onsuccess = () => resolve(request.result);
      });
      await new Promise(resolve => {
        const tx = db.transaction('messages', 'readwrite');
        const store = tx.objectStore('messages');
        const request = store.get('/first'); request.onsuccess = () => {
          const record = request.result;
          record.messages.push({id: 'new', role: 'assistant', parts: [
            {type: 'tool-read', toolCallId: 'partial',
             state: 'input-streaming', input: {id: 'partial'}}
          ]}); store.put(record);
        }; tx.oncomplete = resolve;
      }); db.close();
      const original = IDBDatabase.prototype.transaction;
      IDBDatabase.prototype.transaction = function(stores, mode, ...rest) {
        if (mode === 'readwrite') throw new Error('Exporter must not write');
        return original.call(this, stores, mode, ...rest);
      };
    }""")
    data = download_json(page)
    assert data["conversation"]["messages"][-1]["parts"][0]["state"] == "input-streaming"


def test_empty_storage_is_not_created(page):
    page.get_by_role("button", name="Export conversation").click()
    expect(page.get_by_role("status")).to_contain_text("No saved conversation")
    expect(page.get_by_role("button", name="Download JSON")).to_be_disabled()
    assert page.evaluate("indexedDB.databases()") == []


@pytest.mark.parametrize("version,stores", [(2, ("conversations", "messages")), (1, ("messages",))])
def test_incompatible_storage_is_reported(page, version, stores):
    seed(page, version=version, stores=stores)
    page.get_by_role("button", name="Export conversation").click()
    expect(page.get_by_role("status")).to_contain_text("not supported")
    expect(page.get_by_role("button", name="Download JSON")).to_be_disabled()


def test_storage_denied_and_new_conversation(page, live_url):
    page.evaluate(
        "Object.defineProperty(window, 'indexedDB', {get() {throw new Error('Storage denied')}})"
    )
    page.get_by_role("button", name="Export conversation").click()
    expect(page.get_by_role("status")).to_contain_text("Storage denied")
    page.goto(live_url)
    page.get_by_role("button", name="Export conversation").click()
    expect(page.get_by_role("status")).to_contain_text("Open a saved conversation")


def test_mobile_panel_fits_viewport_and_download_failure_is_retryable(page):
    page.set_viewport_size({"width": 320, "height": 640})
    seed(page)
    page.route("**/api/export", lambda route: route.fulfill(status=500, body="failure"))
    page.get_by_role("button", name="Export conversation").click()
    panel = page.get_by_role("dialog").bounding_box()
    assert panel and panel["x"] >= 0 and panel["x"] + panel["width"] <= 320
    page.get_by_role("button", name="Download JSON").click()
    expect(page.get_by_role("status")).to_contain_text("Export failed")
    expect(page.get_by_role("button", name="Download JSON")).to_be_enabled()
    page.unroute("**/api/export")
    assert download_json(page)["conversation"]["id"] == "/first"


@pytest.mark.network
@pytest.mark.parametrize("live_url", [True], indirect=True)
def test_pinned_stock_ui_export_smoke(page):
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    seed(page)
    page.reload()
    expect(page.get_by_text("Answer /first", exact=True)).to_be_visible(timeout=30_000)
    question = page.locator("#root").get_by_text("Question /first", exact=True).last
    launcher = page.get_by_role("button", name="Export conversation")
    assert (
        question.bounding_box()["y"]
        >= launcher.bounding_box()["y"] + launcher.bounding_box()["height"]
    )
    page.get_by_role("button", name="Export conversation").click()
    assert download_json(page)["conversation"]["id"] == "/first"
    screenshot_dir = os.environ.get("PAPYRUS_SCREENSHOT_DIR")
    if screenshot_dir:
        page.screenshot(path=str(Path(screenshot_dir) / "export-desktop.png"))
        page.set_viewport_size({"width": 320, "height": 640})
        page.screenshot(path=str(Path(screenshot_dir) / "export-mobile.png"))
        page.keyboard.press("Escape")
        assert question.bounding_box()["y"] >= 60
        page.screenshot(path=str(Path(screenshot_dir) / "chat-mobile.png"))
    assert not errors
