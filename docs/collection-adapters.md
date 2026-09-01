# Implementing collection adapters

Papyrus Chat does not treat every top-level directory in
[`papyri/idp.data`](https://github.com/papyri/idp.data) as interchangeable.
Each supported collection has an explicit adapter that turns upstream records
into the stable, provenance-preserving artifact contract used by retrieval.

The current selectable adapters are `dclp`, `ddbdp`, and `translations`.
`HGV_meta_EpiDoc/` is a linked auxiliary source used only when `ddbdp` is
selected. The README's [support matrix](../README.md#supported-upstream-collections)
describes the user-visible boundary.

## How a record reaches the artifact

Source acquisition establishes reproducible bytes and provenance, adapters interpret one upstream format, and the artifact layer owns storage:

1. The CLI validates a lowercase collection name against
   [`SUPPORTED_COLLECTIONS`](../src/papyrus_chat/builder/pipeline.py).
2. A remote build maps that name to an upstream directory through
   [`RemoteGitSource.COLLECTION_DIRS`](../src/papyrus_chat/builder/source.py),
   then checks out only the required directories.
3. The build source lists `.xml` files at the resolved Git commit and returns
   their bytes. Adapters never read an uncommitted working-tree version.
4. The registered adapter parses each record into `DocumentRecord`,
   `IdentifierRecord`, and `PassageRecord` values. Specialized adapters may
   return additional typed component data.
5. The pipeline audits the complete normalized record graph for duplicate
   storage keys and broken document/component relationships.
6. It sorts and persists the verified values, computes the logical content
   hash, writes the manifest, and validates the staged artifact before it is
   published.

## Choose the adapter shape

### EpiDoc text collections

For a TEI EpiDoc collection whose editions and translations follow the
existing structure, start with the shared extractor in
[`collections/epidoc.py`](../src/papyrus_chat/builder/collections/epidoc.py).
A minimal adapter looks like this:

```python
from papyrus_chat.builder.collections.epidoc import ParsedRecord, parse_epidoc_record


def parse_record(
    data: bytes,
    *,
    collection: str,
    source_path: str,
    repository_url: str,
    commit: str,
) -> ParsedRecord:
    return parse_epidoc_record(
        data,
        collection=collection,
        source_path=source_path,
        repository_url=repository_url,
        commit=commit,
    )
```

Use the shared extractor only after checking representative upstream files.
Collection-specific language declarations, passage containers, identifiers,
or canonical URL rules may require adapter options or a dedicated extractor.
For example, DDbDP derives language from edition `xml:lang` attributes and
does not expose embedded translations as transcription passages.

### Metadata and linked sources

A metadata-only source need not become a selectable collection. If it enriches
another collection, model it as a typed component and join it through explicit
identifiers. The DDbDP/HGV implementation demonstrates this pattern in
[`collections/ddbdp.py`](../src/papyrus_chat/builder/collections/ddbdp.py),
[`collections/hgv.py`](../src/papyrus_chat/builder/collections/hgv.py), and
[`components.py`](../src/papyrus_chat/builder/components.py).

Keep missing and one-to-many links observable. Do not guess relationships from
titles, filenames, or insertion order when the source provides stable
identifiers.

Treat repeated upstream values according to their semantics. If a field is a
set in the artifact model, remove exact repeats while preserving first-seen
order, as the HGV metadata adapter does for subjects, commentary, and origins.
Do not silently collapse records whose order or multiplicity carries meaning.
The pipeline's pre-write integrity audit is the final guard: duplicate metadata,
identifier, date, language, component, passage, or link keys fail together with
their source paths before SQLite persistence begins.

### Non-EpiDoc sources

Collections such as bibliographic or RDF data may need a different parser and
possibly a new artifact representation. Keep their parsing in a dedicated
adapter package, but return the existing artifact record types when their
semantics fit. If they do not fit, design and test the artifact schema change
before registering a CLI collection.

## Required adapter contract

Every selectable adapter is called with the same keyword-only context:

- `data`: the source record bytes;
- `collection`: the canonical lowercase CLI name;
- `source_path`: the path within the resolved upstream commit;
- `repository_url`: the source Git URL recorded in provenance;
- `commit`: the full resolved source commit.

Adapters that need no extra collection-specific data return
[`ParsedRecord`](../src/papyrus_chat/builder/collections/epidoc.py), containing:

- one `DocumentRecord`, including a stable ID derived from collection and
  source path;
- zero or more `IdentifierRecord` values attached to that document;
- zero or more `PassageRecord` values with stable structural locators;
- warnings for lossy or unsupported source structures that should remain
  visible to the builder user.

Metadata-only records are valid and must retain their document and identifiers
even when they yield no passages. Specialized return types, such as
`ParsedDDbDP`, must still expose the same four attributes because the main
parse loop relies on that common shape.

Preserve these invariants:

- IDs derive from source identity and structure, never iteration or insertion
  order.
- Every record carries repository URL, exact commit, source path, and an
  optional locator.
- `display_text` preserves visible editorial uncertainty; `search_text` is a
  separately normalized search representation.
- Missing text does not silently delete discoverable metadata.
- Unsupported markup emits a warning unless the adapter handles it explicitly.
- Invalid records fail with a `ParseError` or `BuildError` that includes the
  collection and source path.

## XML safety

Reuse [`parse_xml()`](../src/papyrus_chat/builder/xml.py) for XML inputs. It
disables entity expansion, network resolution, DTD loading, recovery mode, and
huge-tree parsing. It deliberately does not build an XML ID lookup table because
real upstream records can repeat `xml:id` values and no current adapter performs
ID-based lookup.

Do not instantiate a more permissive `lxml` parser inside an adapter. If a new
source requires a parser-policy change, add a minimal failing safety test first
and keep external entities and network access disabled.

## Register the collection

Adding an adapter module is not enough. Update both registration points:

```python
# src/papyrus_chat/builder/pipeline.py
SUPPORTED_COLLECTIONS = {
    # Existing entries...
    "new_collection": ("UpstreamDirectory", parse_new_collection),
}

# src/papyrus_chat/builder/source.py
RemoteGitSource.COLLECTION_DIRS = {
    # Existing entries...
    "new_collection": "UpstreamDirectory",
}
```

The first mapping controls CLI validation and parser dispatch. The second
controls remote sparse checkout; omitting it can make local builds pass while
remote builds lack the required files. If the collection needs auxiliary
directories, include them during sparse-checkout preparation as DDbDP does for
HGV, and keep the auxiliary source out of `SUPPORTED_COLLECTIONS` unless users
can build and query it independently.

If the adapter returns a specialized result, also update `CollectionParser`
and the normalization step in
[`pipeline.py`](../src/papyrus_chat/builder/pipeline.py). Persist only typed
artifact records; do not pass collection-specific XML elements into the schema
layer.

## Fixtures and tests

Use small, real upstream fixtures pinned to an exact commit. Record their
origin and license in
[`tests/fixtures/idp.data/PROVENANCE.md`](../tests/fixtures/idp.data/PROVENANCE.md),
and extend [`scripts/refresh_fixtures.py`](../scripts/refresh_fixtures.py) when
the new fixtures should participate in the refresh workflow.

Cover at least:

1. A representative record with expected title, identifiers, languages,
   passages, locators, and canonical URL.
2. A meaningful edge case from that collection, such as metadata-only content,
   nested text parts, missing language declarations, or one-to-many links.
3. A pipeline build through `LocalGitSource` that writes and validates an
   artifact containing the new collection.
4. CLI listing, case normalization, unknown-collection errors, deterministic
   output, and remote sparse-checkout behavior affected by the new entry.
5. Failure behavior for malformed input, without weakening the shared XML
   safety guarantees.
6. Exact repeated values and any collection-specific uniqueness or relationship
   rules exercised through the pre-write integrity audit.

The default verification commands are:

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Run the network-marked upstream smoke test separately when the adapter changes
remote acquisition:

```console
uv run pytest -m network
```

## Completion checklist

- The adapter has a stable parse contract and preserves source provenance.
- The selectable name and exact upstream path are registered for both dispatch
  and sparse checkout.
- Auxiliary sources and link cardinality are modeled explicitly.
- Real pinned fixtures cover normal, empty, and malformed or unusual records.
- The focused tests and full offline quality gates pass.
- `papyrus-corpus-build --list-collections` shows the intended public name.
- The README support matrix states what is selectable, auxiliary, and still
  unsupported.
