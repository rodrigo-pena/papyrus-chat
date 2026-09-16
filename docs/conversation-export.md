# Exporting a conversation

Papyrus Chat can save a conversation you had in the browser as a file you can
keep, share, or read later. This document explains what gets exported, how the
export works, and how to test it.

Two things are worth knowing up front:

- An export is a copy of the conversation as it appeared on screen. It records
  what the user and the assistant said.
- Exporting is read-only and offline.

## How the export panel gets your conversation

The chat interface comes from an upstream project, `@pydantic/ai-chat-ui`
(version 2.1.0, speaking the Vercel AI protocol, version 7). The app downloads
a specific, versioned copy of that interface as a single HTML file and caches
it locally. We pin that exact file and, using the single-file build, we don't
need to build the chat interface ourselves.

The export button itself is a small script that lives just outside the chat
interface, in an isolated part of the page (a "shadow DOM") so the two can't
interfere with each other. A stylesheet in the app (`shell.css`) reserves space
for the button's toolbar above the chat area. Those style rules are part of the
agreement between our app and the pinned upstream version, so they need to be
rechecked whenever the pinned version changes.

## Where conversations are stored

`export-storage.js` is the only code that talks to the browser's storage. It
reads from a database that the chat interface maintains, with these specifics:

| Item                      | Value                                                                         |
| ------------------------- | ----------------------------------------------------------------------------- |
| Database                  | IndexedDB `chat-storage`, schema version 1                                    |
| Conversation list         | `conversations` store, keyed by `id`; the title comes from the first message  |
| Messages                  | `messages` store, keyed by `id`, holding `{id, messages}`                     |
| Current conversation      | The URL path, including its leading `/` (e.g. `/thread-id`)                   |
| Brand-new conversation    | `/` — nothing saved yet, so nothing to export                                 |
| Detecting page navigation | The browser's `popstate` event and the upstream `history-state-changed` event |

When you open a conversation, its details and messages are read in a single
read-only pass. If no database exists yet, the export aborts cleanly instead of
creating an empty one. If the database has an unexpected version, is missing
the expected stores, or storage is unavailable, you get a clear error.

A few practical consequences:

- The conversation is re-read from storage at the moment you click download, so
  anything saved since you opened the panel is included.
- The chat interface saves your conversation at most every half second while
  you're using it. If you click download immediately after the assistant
  finishes answering, the very last part of the answer might not be saved yet.
- Storage is tied to the exact origin — the same browser, hostname, and port.
  A conversation saved at one address won't appear if you open the app at
  another.

## Upgrading the chat interface

Before moving to a newer version of `@pydantic/ai-chat-ui`, compare it against
this document: check that the storage layout, navigation events, and layout
style rules still match, and see which protocol parts changed. Then update the
pinned URL, update this document, and run the interface browser tests
(described at the end). If the new version stores content we don't recognize,
the right response is to keep it in the export unchanged.

## The download endpoint

Exports go through `POST /api/export` on the local app, with JSON like:

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
                "parts": [{ "type": "text", "text": "Research question" }]
            }
        ]
    }
}
```

Choose `html` or `json` as the format. Either way the response is a file
download with a safe filename and headers that prevent caching. If the
conversation snapshot is malformed you get a 422 response with a plain-language
error; if the request isn't JSON you get a 415. The endpoint follows the same
host-checking rules as the rest of the app.

## What the exported files contain

### JSON format

```json
{
    "kind": "papyrus-chat-browser-conversation",
    "schema_version": 1,
    "exported_at": "2026-09-16T12:00:00+00:00",
    "conversation": { "id": "/thread-id", "title": "Research question", "messages": [] }
}
```

(The example leaves out messages for brevity — a real export always contains at
least one.)

The JSON keeps everything you'd want as a record of the conversation: message
IDs, who said what, the ordered pieces of each message, tool calls (including
their IDs, states, inputs, outputs, and errors), source links, any reasoning
the interface chose to show, and any other content in a format we don't
specifically know about. A few bookkeeping fields are deliberately left out:
message-level metadata and the part-level `providerMetadata`,
`callProviderMetadata`, `providerDetails`, and `signature` fields.

### HTML format

The HTML export renders the same content as a standalone web page. Markdown
and tables are rendered with raw HTML turned off, tool data is escaped, links
are restricted to `http` and `https`, images are not loaded, and the file
carries a content security policy that blocks external assets. Sections with
reasoning or tool activity are collapsed behind native `<details>` controls —
no JavaScript and no internet connection are needed to read the file. Click a
collapsed section to see its full contents.

### What exports don't include

The exports capture the conversation as the browser saw it. They are not the
model's complete internal history, and they cannot bring back reasoning the
interface never displayed, discarded drafts, deleted conversations, content
that was never saved, or internal checkpoints. Conversations cannot be imported
back into the app, and PDF export or shareable links are not part of this
feature.

## Testing

The browser tests use Playwright (already a development dependency). Install
its browser once, then run the offline storage and download tests:

```bash
uv run playwright install chromium
uv run pytest -m 'browser and not network' tests/web/test_export_browser.py
```

To also test against the real pinned chat interface, plus a scripted research
run that exercises history compaction and a rejected-then-accepted answer:

```bash
uv run pytest -m browser tests/web/test_export_browser.py
```

The first run needs internet access to download the pinned interface; no model
API keys are required. If you'd rather use an installed Chrome in a throwaway
profile instead of Playwright's browser:

```bash
PAPYRUS_BROWSER_CHANNEL=chrome uv run pytest -m browser tests/web/test_export_browser.py
```

Set `PAPYRUS_SCREENSHOT_DIR` to an existing directory and the tests will save
desktop and mobile screenshots of the panel, plus a screenshot of an exported
HTML file. Together, the tests cover: old conversations, page reloads and
navigation, storage failures, refreshing the snapshot before download, tool
calls still in progress, failed downloads, reading an export offline, small
screens, history compaction, and the exclusion of drafts that were rejected.
Before a release, also install the built package in a clean environment and
try both download formats to confirm everything ships with the package.
