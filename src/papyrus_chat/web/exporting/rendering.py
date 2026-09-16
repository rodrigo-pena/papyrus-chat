"""Script-free, self-contained HTML; all conversation content is untrusted."""

import json
from datetime import datetime
from html import escape
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
STYLE = """
* { box-sizing: border-box; }
body { margin: 0; background: #f7f7f5; color: #252525; font: 16px/1.65 system-ui, sans-serif; }
main { max-width: 960px; margin: auto; padding: 40px 24px; }
h1 { font-size: 28px; line-height: 1.3; overflow-wrap: anywhere; }
h2 { font-size: 18px; margin-top: 0; }
h3 { font-size: 16px; }
.meta { color: #595959; font-size: 14px; overflow-wrap: anywhere; }
article { background: #fff; padding: 24px; border: 1px solid #d8d8d3; margin: 24px 0; }
article[data-role="user"] { border-left: 4px solid #607768; }
article p, article li { overflow-wrap: anywhere; }
a { color: #175b9b; overflow-wrap: anywhere; }
details { border: 1px solid #d8d8d3; padding: 12px 16px; margin: 16px 0; }
summary { cursor: pointer; font-weight: 600; overflow-wrap: anywhere; }
summary:focus-visible, a:focus-visible { outline: 3px solid #3980cc; outline-offset: 3px; }
pre { font: 13px/1.6 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere;
      background: #f4f4f1; padding: 16px; max-height: 32rem; overflow: auto; }
code { font-family: ui-monospace, monospace; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
td, th { border: 1px solid #d8d8d3; padding: 8px; text-align: left; overflow-wrap: anywhere; }
hr { border: 0; border-top: 1px solid #ddd; margin: 24px 0; }
@media (max-width: 600px) { main { padding: 20px 12px; } article { padding: 16px; } }
@media print { body { background: #fff; } main { max-width: none; padding: 0; }
              pre { max-height: none; overflow: visible; } }
"""
TOOL_STATES = {
    "input-streaming": "Partial input",
    "input-available": "Awaiting result",
    "output-available": "Completed",
    "output-error": "Error",
    "approval-requested": "Awaiting approval",
    "approval-responded": "Approval recorded",
    "output-denied": "Denied",
}


def safe_link(value: str) -> bool:
    try:
        url = urlsplit(value)
    except ValueError:
        return False
    return url.scheme.lower() in {"http", "https"} and bool(url.netloc)


class TranscriptMarkdown(MarkdownIt):
    def validateLink(self, url: str) -> bool:
        return safe_link(url)


def json_block(value: object) -> str:
    return "<pre data-json>" + escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"


def render_part(part: dict, markdown: MarkdownIt) -> str:
    kind = part["type"]
    if kind == "text":
        partial = '<p class="meta">Partial text</p>' if part.get("state") == "streaming" else ""
        return partial + markdown.render(part["text"])
    if kind == "reasoning":
        state = f" · {escape(str(part['state']))}" if part.get("state") else ""
        return (
            f"<details><summary>Reasoning{state}</summary>"
            + markdown.render(part["text"])
            + "</details>"
        )
    if kind == "dynamic-tool" or kind.startswith("tool-"):
        name = part.get("toolName", kind.removeprefix("tool-"))
        state = escape(TOOL_STATES.get(part["state"], part["state"]))
        content = (
            f"<details><summary>{escape(str(name))} · {state}</summary>"
            f'<p class="meta">Call ID: {escape(str(part["toolCallId"]))}</p>'
        )
        for key, label in (("input", "Input"), ("output", "Output"), ("errorText", "Error")):
            if key in part:
                content += f"<h3>{label}</h3>" + json_block(part[key])
        remaining = {
            key: value
            for key, value in part.items()
            if key
            not in {"type", "toolName", "toolCallId", "state", "input", "output", "errorText"}
        }
        if remaining:
            content += "<h3>Details</h3>" + json_block(remaining)
        return content + "</details>"
    if kind == "source-url":
        url = part.get("url")
        if isinstance(url, str) and safe_link(url):
            label = escape(str(part.get("title") or url))
            return (
                f'<p>Source: <a href="{escape(url, quote=True)}" rel="noreferrer">{label}</a></p>'
            )
    if kind == "step-start":
        return '<hr aria-label="Research step">'
    return f"<details><summary>{escape(kind)}</summary>{json_block(part)}</details>"


def render_html(document: dict) -> str:
    markdown = TranscriptMarkdown("commonmark", {"html": False}).enable("table").disable("image")
    conversation = document["conversation"]
    title = escape(conversation["title"])
    exported_at = datetime.fromisoformat(document["exported_at"]).strftime("%Y-%m-%d %H:%M UTC")
    articles = []
    for index, message in enumerate(conversation["messages"], start=1):
        role = escape(message["role"])
        parts = "".join(render_part(part, markdown) for part in message["parts"])
        articles.append(
            f'<article data-role="{role}"><h2>{index}. {role.capitalize()}</h2>{parts}</article>'
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{escape(CSP, quote=True)}">'
        f"<title>{title}</title><style>{STYLE}</style></head><body><main>"
        f'<header><p class="meta">Papyrus Chat · Conversation export</p><h1>{title}</h1>'
        f'<p class="meta">Exported {exported_at}<br>'
        f"Conversation: {escape(conversation['id'])}</p>"
        '<p class="meta">Browser-saved snapshot. A response still in progress may be incomplete. '
        "Reasoning is included only where exposed and saved. Expand sections to read tool activity."
        "</p></header>" + "".join(articles) + "</main></body></html>"
    )
