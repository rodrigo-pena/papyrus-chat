# Conversation export integration

The export panel reads a saved conversation from the current browser and submits
it to the local renderer. Neither download format runs the agent, reads the model's
compacted history, changes browser storage, or creates a server-side conversation
archive. Downloads contain the recorded research transcript, not a verification
that the model's conclusions are correct.

## Browser and upstream contract

The application uses `@pydantic/ai-chat-ui` **2.1.0**, Vercel AI protocol version 7,
and the upstream `offline/index.html` build. Pydantic AI fetches and caches this
versioned single-file build. The ordinary upstream `dist/index.html` points to
unversioned assets, so pinning that HTML URL alone would not pin the running UI.
The single-file build avoids those moving asset references and needs no local
frontend build. The `html_source` injection remains available for tests and custom
HTML; the exporter is attached to that HTML too.

The small export launcher lives outside the stock React root in a shadow DOM.
`shell.css` reserves a toolbar above the pinned UI's viewport-height containers;
its selectors are part of this compatibility contract. The normal CLI serves the
application at the origin root.

`export-storage.js` is the only browser storage adapter. Its contract is:

| Item | Expected value |
| --- | --- |
| Database | IndexedDB `chat-storage`, schema version 1 |
| Conversation store | `conversations`, keyed by `id`; title from `firstMessage` |
| Message store | `messages`, keyed by `id`, containing `{id, messages}` |
| Active conversation ID | `window.location.pathname`, including its leading `/` |
| New conversation | `/`; no exportable saved thread yet |
| Navigation | `popstate` and upstream's `history-state-changed` event |

Conversation metadata and messages are read in one read-only transaction. If the
database does not exist, the attempted creation is aborted. Unsupported versions,
missing stores, and unavailable storage produce an explicit error. The adapter
does not migrate legacy localStorage; the stock UI performs its own migration.

The panel rereads the selected conversation when a download is requested, so it
includes saves made since opening the panel. Upstream throttles saves by 500 ms;
completion on screen does not guarantee that the final save has happened yet.
Changing the browser, hostname, or port changes the storage origin. An empty
history at a different origin cannot be recovered by this endpoint.

Before upgrading the UI, verify the storage schema, navigation events, viewport
layout selectors, and protocol parts against the new upstream source. Then update
the pinned URL and this contract and run the stock UI browser smoke tests. Do not
silently drop unknown content or change stored conversations to make exports work.

## Download endpoint and format

`POST /api/export` accepts `Content-Type: application/json`:

```json
{
  "format": "html",
  "conversation": {
    "id": "/thread-id",
    "title": "Research question",
    "messages": [
      {
        "id": "message-id",
        "role": "user",
        "parts": [{"type": "text", "text": "Research question"}]
      }
    ]
  }
}
```

`format` is `html` or `json`. Both return an attachment with a safe filename and
`Cache-Control: no-store`. Invalid snapshots return 422 with a readable error;
non-JSON requests return 415. The endpoint and static assets retain the stock
application's host validation.

JSON exports have this envelope:

```json
{
  "kind": "papyrus-chat-browser-conversation",
  "schema_version": 1,
  "exported_at": "2026-09-16T12:00:00+00:00",
  "conversation": {"id": "/thread-id", "title": "Research question", "messages": []}
}
```

The example omits messages for brevity; a download requires a nonempty message
list. Message IDs, roles, ordered parts, tool call IDs/states, inputs, outputs,
errors, source links, exposed reasoning text, and unfamiliar JSON part data are
preserved. Message-level metadata and part-level `providerMetadata`,
`callProviderMetadata`, `providerDetails`, and `signature` fields are omitted.
Identically named fields inside tool inputs or results remain intact as evidence.
No current server model configuration or corpus provenance is inferred for an old
conversation.

HTML is generated from the same normalized document. Markdown and tables are
rendered with raw HTML disabled. Tool data is escaped, links allow only HTTP(S),
images are not loaded, and the file includes an asset-blocking content security
policy. Reasoning and tool sections use native `<details>` controls; no JavaScript
or remote resources are required. Expand sections to read their complete contents.

The formats describe the browser transcript. They are not lossless SDK histories,
do not enable import/resume, and cannot recover unexposed reasoning, discarded
drafts, deleted threads, unsaved content, or internal checkpoints. PDF export,
hosted sharing links, and internal diagnostic recording are outside this feature.

## Verification

The default suite skips browser and network tests:

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

Playwright is a development dependency. Install its browser once, then run the
offline IndexedDB and download tests:

```bash
uv run playwright install chromium
uv run pytest -m 'browser and not network' tests/web/test_export_browser.py
```

To also check the pinned stock UI and a deterministic research run that compacts
history and rejects an invalid answer before succeeding:

```bash
uv run pytest -m browser tests/web/test_export_browser.py
```

That command needs CDN access on the first run; no LLM credentials are required.
Alternatively, use an installed Chrome in an isolated temporary profile:

```bash
PAPYRUS_BROWSER_CHANNEL=chrome uv run pytest -m browser tests/web/test_export_browser.py
```

Set `PAPYRUS_SCREENSHOT_DIR` to an existing directory to save desktop/mobile panel
screenshots and a screenshot of an exported HTML document. The tests cover old
threads, reloads, navigation, storage errors, snapshot refreshes, partial tool
states, failed downloads, offline reading, responsive layout, compaction, and
exclusion of rejected drafts. A release smoke test should also install the built
wheel in a separate environment and run both downloads to verify asset packaging.
