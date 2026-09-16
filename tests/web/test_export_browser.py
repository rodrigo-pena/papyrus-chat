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
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
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
def research_counts():
    return {"summaries": 0}


@pytest.fixture(scope="module")
def live_url(corpus_artifact, tmp_path_factory, request, research_counts):
    html = tmp_path_factory.mktemp("export-ui") / "index.html"
    html.write_text(
        '<!doctype html><html><head><meta name="viewport" content="width=device-width">'
        '<link rel="icon" href="data:,"></head><body><main id="root">Chat</main></body></html>'
    )
    env = {"LLM_BASE_URL": "https://provider.example/v1", "LLM_MODEL": "research-model"}
    model = TestModel(call_tools=[], custom_output_text="Unused")
    if getattr(request, "param", False) == "compaction":
        env["LLM_CONTEXT_WINDOW"] = "32768"
        research = 0

        def summarize(messages, info):
            research_counts["summaries"] += 1
            return ModelResponse([TextPart("PRIVATE_CHECKPOINT inventory inspected")])

        async def stream(messages, info):
            nonlocal research
            research += 1
            if research < 4:
                yield "PRIVATE_UNVALIDATED πάπυρος " * 1000
                yield {
                    0: DeltaToolCall(
                        name="describe_corpus", json_args="{}", tool_call_id=f"inventory-{research}"
                    )
                }
            elif research == 4:
                yield "Corpus evidence: https://papyri.info/ddbdp/PRIVATE_REJECTED."
            else:
                yield "Inventory examined. No corpus evidence was inspected."

        model = FunctionModel(function=summarize, stream_function=stream)
    app = load_app(
        corpus_artifact,
        env=env,
        model=model,
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


def download_html(page):
    with page.expect_download() as saved:
        page.get_by_role("button", name="Download HTML").click()
    assert saved.value.suggested_filename.endswith(".html")
    return Path(saved.value.path())


def test_html_download_opens_offline_without_requests_and_expands_tools(page, tmp_path):
    seed(page)
    page.get_by_role("button", name="Export conversation").click()
    path = tmp_path / "conversation.html"
    path.write_bytes(download_html(page).read_bytes())
    requests = []
    page.on("request", lambda request: requests.append(request.url))
    page.context.set_offline(True)
    page.goto(path.as_uri())
    expect(page.get_by_role("heading", name="Question /first")).to_be_visible()
    expect(page.get_by_text("Answer /first", exact=True)).to_be_visible()
    page.locator("summary").filter(has_text="Reasoning").click()
    expect(page.get_by_text("Inspect πάπυρος", exact=True)).to_be_visible()
    page.locator("summary").filter(has_text="inspect_documents").click()
    output = page.locator("pre").filter(has_text="πάπυρος")
    expect(output).to_be_visible()
    assert json.loads(output.inner_text())["text"] == "πάπυρος " * 10_000
    assert requests == [path.as_uri()]
    for width in (320, 768, 1440):
        page.set_viewport_size({"width": width, "height": 800})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    screenshot_dir = os.environ.get("PAPYRUS_SCREENSHOT_DIR")
    if screenshot_dir:
        page.set_viewport_size({"width": 1024, "height": 900})
        page.screenshot(path=str(Path(screenshot_dir) / "conversation-html.png"))


@pytest.mark.network
@pytest.mark.parametrize("live_url", [True], indirect=True)
def test_pinned_stock_ui_export_smoke(page):
    errors = []
    requests = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
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
    assert "Answer /first" in download_html(page).read_text()
    screenshot_dir = os.environ.get("PAPYRUS_SCREENSHOT_DIR")
    if screenshot_dir:
        page.screenshot(path=str(Path(screenshot_dir) / "export-desktop.png"))
        page.set_viewport_size({"width": 320, "height": 640})
        page.screenshot(path=str(Path(screenshot_dir) / "export-mobile.png"))
        page.keyboard.press("Escape")
        assert question.bounding_box()["y"] >= 60
        page.screenshot(path=str(Path(screenshot_dir) / "chat-mobile.png"))
    assert not errors
    assert all(url.startswith(("http://127.0.0.1:", "data:")) for url in requests)


@pytest.mark.network
@pytest.mark.parametrize("live_url", ["compaction"], indirect=True)
def test_compacted_and_retried_run_exports_earlier_visible_tools(page, live_url, research_counts):
    page.goto(live_url)
    composer = page.get_by_placeholder("What would you like to know?")
    composer.fill("Investigate inventory.")
    composer.press("Enter")
    expect(
        page.get_by_text("Inventory examined. No corpus evidence was inspected.", exact=True)
    ).to_be_visible(timeout=30_000)
    page.evaluate("""async () => {
      const {readSnapshot} = await import('/papyrus-assets/export-storage.js');
      const deadline = Date.now() + 5000;
      while (Date.now() < deadline) {
        try {
          const snapshot = await readSnapshot(location.pathname);
          if (JSON.stringify(snapshot).includes('Inventory examined.')) return;
        } catch { /* The stock UI throttles its saves. */ }
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      throw new Error('Stock UI did not save the completed answer');
    }""")
    assert research_counts["summaries"] >= 1
    page.get_by_role("button", name="Export conversation").click()
    expect(page.locator("papyrus-export").get_by_role("status")).to_have_text("Ready to download.")
    document = download_json(page)
    parts = [part for message in document["conversation"]["messages"] for part in message["parts"]]
    calls = [part["toolCallId"] for part in parts if part["type"] == "tool-describe_corpus"]
    assert calls == ["inventory-1", "inventory-2", "inventory-3"]
    rendered = download_html(page).read_text()
    for private_text in ("PRIVATE_CHECKPOINT", "PRIVATE_UNVALIDATED", "PRIVATE_REJECTED"):
        assert private_text not in json.dumps(document)
        assert private_text not in rendered
    for call in calls:
        assert call in rendered
