# Local MCP integration

`papyrus-mcp` exposes one validated Papyrus Chat artifact as a deterministic,
read-only MCP server over STDIO. The MCP process does local retrieval only: it
makes no LLM, web-search, provider, credential, or other network calls. The
connected Codex or ChatGPT host interprets natural language, calls any model it
uses, and writes the final answer.

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

With the GitHub package:

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

## Connect Codex

Register the local STDIO command with the Codex CLI. Use an absolute artifact
path when the client may start from a different working directory:

```console
codex mcp add papyrus-corpus -- \
  papyrus-mcp --artifact /absolute/path/to/data/papyrus-corpus
```

Check the registration with `codex mcp list` or `codex mcp get papyrus-corpus`.
The Codex MCP configuration is normally `~/.codex/config.toml`; a trusted
project can use `.codex/config.toml` instead. The equivalent entry is:

```toml
[mcp_servers.papyrus-corpus]
command = "papyrus-mcp"
args = ["--artifact", "/absolute/path/to/data/papyrus-corpus"]
startup_timeout_sec = 60
tool_timeout_sec = 120
```

The longer timeouts account for measured startup validation of about 8.3
seconds and a first semantic suggestion of about 11.3 seconds on the current
artifact. Restart Codex after changing its configuration, then use `/mcp` to
confirm that `papyrus-corpus` and its tools are available.

## Connect the ChatGPT desktop app

The ChatGPT desktop app can configure the same local STDIO server from its MCP
server settings:

1. Open Settings and select MCP servers.
2. Select Add server and choose STDIO.
3. Enter `papyrus-corpus` as the name, `papyrus-mcp` as the command, and add
   `--artifact` followed by the absolute artifact path as its argument.
4. Save and restart the app. Use `/mcp` in a chat to verify the connection.

If the app does not inherit your shell PATH, enter the absolute path to the
`papyrus-mcp` executable or use a wrapper command from the environment where
the package was installed. ChatGPT web does not read local Codex configuration
files; use a local Codex client or the desktop app for this STDIO server.

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
the entry. The first semantic suggestion can be slower while the local encoder
initializes.

### Semantic suggestions unavailable

Check `get_corpus_info` first. No semantic index means the artifact was built
without `--semantic-model-dir`; a missing runtime means the server was started
without the semantic extra. Neither state prevents lexical search, facets,
identifier lookup, or inspection.

### STDIO protocol errors or malformed JSON

Run the server directly and inspect stderr. Do not pipe banners, debug prints,
or application output to STDOUT; only MCP frames belong there. Use `--verbose`
for diagnostics, and confirm that the configured command points to
`papyrus-mcp` rather than a shell wrapper that prints startup text.

### Server is not visible after configuration

Run `codex mcp get papyrus-corpus`, verify the artifact path and executable,
restart the Codex or ChatGPT desktop client, and check `/mcp` again. The web
ChatGPT client cannot launch this local STDIO process.
