# Local MCP integration

`papyrus-mcp` exposes one validated Papyrus Chat artifact as a deterministic,
read-only MCP server over STDIO. The MCP process does local retrieval only: it
makes no LLM, web-search, provider, credential, or other network calls. The
connected MCP host, such as Codex, ChatGPT desktop, OpenCode, or another
compatible client, interprets natural language, calls any model it uses, and
writes the final answer.

The artifact must already exist. Its path is fixed when the server starts, so a
tool caller cannot switch artifacts during a session.

## Build an artifact

The semantic index is optional, but this example builds it so
`suggest_subjects` is available. Download the pinned model snapshot once:

```console
uvx --from huggingface-hub hf download intfloat/multilingual-e5-small \
  --revision 4a4cddf9cf6d77a61cc1c73f824ec2127773db85 \
  --local-dir ./models/multilingual-e5-small
```

From any directory, build through the GitHub package:

```console
uvx --from 'papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-corpus-build dclp ddbdp translations \
  --semantic-model-dir ./models/multilingual-e5-small \
  --output ./data/papyrus-corpus
```

From a local checkout, install the extras and run the same builder:

```console
uv sync --extra mcp --extra semantic
uv run papyrus-corpus-build dclp ddbdp translations \
  --semantic-model-dir ./models/multilingual-e5-small \
  --output ./data/papyrus-corpus
```

An existing compatible artifact can be used without rebuilding. It must retain
`manifest.json`, `corpus.sqlite`, and `ATTRIBUTION.md` at the artifact root.
The server performs full validation before it starts.

## Start the server

Recommended, with the GitHub package through `uvx`:

```console
uvx --from 'papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-mcp --artifact ./data/papyrus-corpus
```

With a local checkout:

```console
uv run --extra mcp --extra semantic papyrus-mcp \
  --artifact ./data/papyrus-corpus
```

`--artifact` is required. `--verbose` is the only optional flag and sends
diagnostics to stderr. STDOUT is reserved for MCP frames. There are no model,
provider, API-key, web-search, host, or port options.

The `[mcp]` extra installs only the protocol SDK. Add `[semantic]` as well when
the artifact contains a semantic index and you want subject suggestions. A
server can still serve the other five tools without the semantic runtime. If
the artifact has no semantic index, `suggest_subjects` returns
`available: false`; if the index exists but the runtime is missing, its reason
names the `[mcp,semantic]` extras. An available index with no matching labels
returns `available: true` and an empty list.

## Register the MCP server

An MCP host normally starts a local STDIO server itself. Register the command
below in the host rather than leaving a separate `papyrus-mcp` process running.
Every client needs the same process definition, although configuration names and
file formats differ:

| Setting           | Value                                                            |
| ----------------- | ---------------------------------------------------------------- |
| Server name       | `papyrus-corpus` (or another local name)                         |
| Transport         | STDIO, sometimes labeled Local or Command                        |
| Executable        | `uvx`                                                            |
| Arguments         | Use the ordered argument list in the JSON example below          |
| Environment       | None required                                                    |
| Working directory | None required                                                    |
| Timeouts          | Allow about 60 seconds for startup and 120 seconds per tool call |

In this setup, `papyrus-mcp` is not a separately installed executable. The MCP
client starts `uvx`, which creates or reuses a managed environment for the Git
package and runs its `papyrus-mcp` console entry point. Test and pre-warm that
environment once before registering it:

```console
uvx --from 'papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-mcp --help
```

The first uncached run requires network access to resolve and install the
package and dependencies. `uvx` stores the disposable environment in its cache
to reduce later startup overhead. This launcher activity happens before the MCP
server starts; the running `papyrus-mcp` server itself still performs only local
retrieval.

If `uvx` is not on the host application's `PATH`, use its absolute path. As
alternatives, a persistent package installation can register `papyrus-mcp`
directly, while a local checkout can register `uv` with
`run --extra mcp --extra semantic papyrus-mcp ...` and set the checkout as its
working directory.

Many clients use a JSON or JSONC file with separate `command` and `args` fields.
For clients that document an `mcpServers` object, the entry commonly looks like
this:

```json
{
    "mcpServers": {
        "papyrus-corpus": {
            "command": "uvx",
            "args": [
                "--from",
                "papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git",
                "papyrus-mcp",
                "--artifact",
                "/absolute/path/to/data/papyrus-corpus"
            ]
        }
    }
}
```

This is a portable pattern, not a universal schema: some clients use `mcp`
instead of `mcpServers`, store the command and arguments in one array, or expose
the same fields in a settings screen. Follow the client's documented field names
and config location. Prefer a separate argument list when supported; it avoids
shell-quoting problems in paths.

When a client offers both Command/STDIO and URL/HTTP registration, choose the
command form. `papyrus-mcp` is STDIO-only and does not expose a URL, host, or
port. A client that accepts only a remote MCP URL cannot connect directly; it
needs a trusted STDIO-to-HTTP bridge or tunnel.

### Codex

Register the local command with the Codex CLI:

```console
codex mcp add papyrus-corpus -- \
  uvx --from 'papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git' \
  papyrus-mcp --artifact /absolute/path/to/data/papyrus-corpus
```

Check the registration with `codex mcp list` or `codex mcp get papyrus-corpus`.
The Codex MCP configuration is normally `~/.codex/config.toml`; a trusted
project can use `.codex/config.toml` instead. The equivalent entry is:

```toml
[mcp_servers.papyrus-corpus]
command = "uvx"
args = [
  "--from",
  "papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git",
  "papyrus-mcp",
  "--artifact",
  "/absolute/path/to/data/papyrus-corpus",
]
startup_timeout_sec = 60
tool_timeout_sec = 120
```

The longer timeouts account for measured startup validation of about 8.3
seconds and a first semantic suggestion of about 11.3 seconds on the current
artifact. Restart Codex after changing its configuration, then use `/mcp` to
confirm that `papyrus-corpus` and its tools are available. See the
[Codex MCP documentation](https://developers.openai.com/codex/mcp/) for the
current CLI, desktop, IDE, and configuration-file options.

### ChatGPT

The ChatGPT desktop app can configure the same local STDIO server from its MCP
server settings:

1. Open Settings and select MCP servers.
2. Select Add server and choose STDIO.
3. Enter `papyrus-corpus` as the name and `uvx` as the command. Add `--from`,
   `papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git`,
   `papyrus-mcp`, `--artifact`, and the absolute artifact path as its arguments,
   in that order.
4. Save and restart the app. Use `/mcp` in a chat to verify the connection.

If the app does not inherit your shell PATH, enter the absolute path to `uvx`.
The desktop app, Codex CLI, and Codex IDE extension can share MCP configuration
on the same Codex host, so an existing Codex entry may already be available
after restarting the app.

ChatGPT web cannot launch this local STDIO process or read local Codex
configuration directly. To use the private server there, connect it through a
supported bridge such as
[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels),
which can forward requests to a local STDIO command. A separately deployed
Streamable HTTP wrapper is another option, but `papyrus-mcp` itself does not
provide that transport.

### OpenCode

Run `opencode mcp add` for interactive setup, or add a local server to a
project-level `opencode.json` or the global
`~/.config/opencode/opencode.json`:

```json
{
    "$schema": "https://opencode.ai/config.json",
    "mcp": {
        "papyrus-corpus": {
            "type": "local",
            "command": [
                "uvx",
                "--from",
                "papyrus-chat[mcp,semantic] @ git+https://github.com/rodrigo-pena/papyrus-chat.git",
                "papyrus-mcp",
                "--artifact",
                "/absolute/path/to/data/papyrus-corpus"
            ],
            "enabled": true,
            "timeout": 120000
        }
    }
}
```

OpenCode puts the executable and arguments in one `command` array and expresses
its MCP initialization timeout in milliseconds. Restart OpenCode after editing
its configuration, then run `opencode mcp list` to confirm that
`papyrus-corpus` is connected. See the
[OpenCode MCP server documentation](https://opencode.ai/docs/mcp-servers/) for
current configuration options.

### Other MCP clients

Look for an MCP, Tools, Integrations, or Developer settings page and choose a
local, STDIO, or command-based server. Map its fields to the table above, restart
or reload the client, and verify that it discovers exactly the six tools listed
in [Tool contract and workflow](#tool-contract-and-workflow).

Config-file locations and schemas are client-specific. User-level configuration
makes the corpus available across projects; project-level configuration is easier
to share but should be enabled only in repositories you trust. If a GUI provides
one command field rather than separate executable and argument fields, enter the
full command and quote the package specification and any paths containing spaces
according to that client's rules.

## Install the research skill

This checkout includes the instruction-only skill at
`skills/research-papyri/SKILL.md`. Install that directory with your Codex skill
installer, or copy it into the discoverable local skills directory:

```console
mkdir -p ~/.agents/skills
cp -R skills/research-papyri ~/.agents/skills/
```

The skill is automatically discoverable and contains no scripts, plugin
metadata, model configuration, or network behavior.

## Tool contract and workflow

The server exposes exactly six tools:

| Tool                | Purpose                                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_corpus_info`   | Return schema, builder/source provenance, collections, statistics, languages, hash, creation time, and semantic capability.                       |
| `suggest_subjects`  | Return bounded exact HGV subject labels, scoped prevalence/coverage, and semantic availability.                                                   |
| `search_documents`  | Accept `CorpusQuery`; return exact candidate counts and at most 100 lean hits with located snippets and canonical URLs.                           |
| `facet_documents`   | Count a bounded collection, language, subject, material, origin, or passage-kind facet with exact `total_values` and truncation.                  |
| `lookup_document`   | Normalize an identifier and return exact match counts plus bounded, deterministic lean matches.                                                   |
| `inspect_documents` | Inspect 1-20 selected IDs with 1-10 passages, 200-2000-character excerpts, up to 8 focus terms, HGV context, line references, and canonical URLs. |

For a question about an identifier, look it up first and inspect the returned
document IDs. For a conceptual question, inspect corpus information when
needed, build multilingual lexical alternatives, ask for subject suggestions,
evaluate refinements with facets, search, and then inspect selected records.
Disclose collections, inclusive date interval, language, passage kind, lexical
groups, subject filters, and whether semantic suggestions were available.

Keep three kinds of statements separate:

- corpus evidence: returned counts, metadata, excerpts, labels, and line references;
- external background: sources obtained by the host from outside the artifact;
- model synthesis: interpretation or historical context inferred from those inputs.

Cite document claims only with the canonical `papyri.info` URLs returned by the
tools. Never construct a URL from an identifier or from memory. Empty and
unknown results are successful results; invalid fields, schemas, date intervals,
or limits are tool errors that should be corrected.

Example requests to the connected host:

- "Look up TM 23944, then inspect the matching record and cite the returned
  papyri.info URL."
- "Within DDbDP, how many Greek documents dated 700-750 CE concern taxes?
  Try lexical alternatives such as φόρος, phoros, and tax, use subject facets
  if available, then inspect representative records and disclose the scope."
- "Find documentary texts with a monthly list structure. Search the relevant
  Greek and English terms, compare collection and passage-kind facets, and
  cite only inspected records."

## Rebuilding the artifact safely

The server holds a read-only SQLite connection for its entire client session.
Before rebuilding the same artifact path with `papyrus-corpus-build --force`,
stop the MCP client/server session. Rebuild, validate, and then start a new
server session. A connected process must not continue reading a path while the
builder replaces it.

## Troubleshooting

### "MCP support is optional" or import errors

Install the protocol extra: `[mcp]` for retrieval-only use, or
`[mcp,semantic]` for an artifact with semantic subject suggestions. With a
local checkout, rerun `uv run --extra mcp --extra semantic papyrus-mcp ...`.

### Invalid artifact or unsupported schema

Pass the artifact root, not `manifest.json`, and confirm that all three required
files are present. The artifact must be a supported schema-v3 build. Rebuild it
with the builder commands above if validation reports a missing file, bad
manifest, incompatible schema, or integrity problem.

### Startup or tool timeout

Use `startup_timeout_sec = 60` and `tool_timeout_sec = 120`. Keep the artifact
on local storage, use an absolute path, and restart the client after changing
the entry. If the first `uvx` launch times out while installing dependencies,
run the pre-warming command above and reconnect. The first semantic suggestion
can also be slower while the local encoder initializes.

### Semantic suggestions unavailable

Check `get_corpus_info` first. No semantic index means the artifact was built
without `--semantic-model-dir`; a missing runtime means the server was started
without the semantic extra. Neither state prevents lexical search, facets,
identifier lookup, or inspection.

### STDIO protocol errors or malformed JSON

Run the server directly and inspect stderr. Do not pipe banners, debug prints,
or application output to STDOUT; only MCP frames belong there. Use `--verbose`
for diagnostics, and confirm that the registered `uvx` argument list ends in
the `papyrus-mcp --artifact ...` command rather than using a shell wrapper that
prints startup text.

### Server is not visible after configuration

Use the client's MCP status command, such as `codex mcp get papyrus-corpus` or
`opencode mcp list`, then verify the executable and artifact paths. Restart the
client after changing configuration. URL-only clients, including ChatGPT web
without a tunnel, cannot launch this local STDIO process directly.
