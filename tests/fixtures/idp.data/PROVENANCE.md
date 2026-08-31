# Fixture provenance

The XML files under this directory are copied verbatim from the
[`papyri/idp.data`](https://github.com/papyri/idp.data) repository, which is
distributed under the [Creative Commons Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).

The original DCLP and Translations fixtures remain pinned to the commit below.
The DDbDP/HGV pair was added from the later upstream snapshot recorded in its
own table because the older snapshot does not contain that pair.

- **Pinned upstream commit:** `04568cb5ea3775d8113bb6e7edfd9c7168cf7e88`
- **Pinned on:** 2026-08-31

| Fixture file (this repository)     | Upstream source path          |
| ---------------------------------- | ----------------------------- |
| `DCLP/23/23944.xml`                | `DCLP/23/23944.xml`           |
| `DCLP/23/23702.xml`                | `DCLP/23/23702.xml`           |
| `Translations/3/3227-1.xml`        | `Translations/3/3227-1.xml`   |
| `Translations/3/3643-1.xml`        | `Translations/3/3643-1.xml`   |

Additional documentary fixtures:

| Fixture file (this repository) | Upstream source path | Upstream commit |
| ------------------------------ | -------------------- | --------------- |
| `DDbDP/27/27093.xml` | `DDbDP/27/27093.xml` | `027a4a3a2d8a669bed692ed5d918892bdb7ea1b3` |
| `HGV_meta_EpiDoc/HGV28/27093.xml` | `HGV_meta_EpiDoc/HGV28/27093.xml` | `027a4a3a2d8a669bed692ed5d918892bdb7ea1b3` |

Upstream contributing projects represented by these records: the Digital
Corpus of Literary Papyri (DCLP/LDAB, Trismegistos) and APIS-derived
translations (see the `<authority>` element inside each record).

## Project-created files (NOT from upstream)

| File                                    | Purpose                                                            |
| --------------------------------------- | ------------------------------------------------------------------ |
| `../idp.data-invalid/DCLP/99/broken-record.xml` | Deliberately malformed XML for failure-path testing. |

The `idp.data-invalid` tree is kept outside the buildable fixture corpus so
normal builds from `tests/fixtures/idp.data` always succeed.

## Refreshing fixtures

By default, fixtures are pinned to the commit recorded above. An informed
user may regenerate them from the same commit, from the HEAD of the upstream
default branch (`master`), or from any other commit SHA:

```console
# Re-fetch at the pinned commit (no-op unless files drifted):
uv run python scripts/refresh_fixtures.py

# Refresh from the upstream default branch HEAD:
uv run python scripts/refresh_fixtures.py --ref master

# Refresh from an arbitrary commit:
uv run python scripts/refresh_fixtures.py --ref <commit-sha>
```

The script rewrites the upstream-derived files listed above and updates the
pinned commit recorded here. It requires network access and is never part of
the default test suite. When refreshing from a moving ref, review the diff
and re-run the full test suite: adapter expectations may change upstream.
