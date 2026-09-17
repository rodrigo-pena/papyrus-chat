"""Curated names for supported collections and auxiliary metadata sources."""

# Project names: https://github.com/papyri/idp.data#repository-contents
COLLECTION_NAMES: dict[str, str] = {
    "ddbdp": "Duke Data Bank of Documentary Papyri",
    "dclp": "Digital Corpus of Literary Papyri",
    # Descriptive label for the upstream Translations directory, not a project name.
    "translations": "Papyri.info translations",
}

METADATA_SOURCE_NAMES: dict[str, str] = {
    "hgv": "Heidelberger Gesamtverzeichnis der griechischen Papyrusurkunden Ägyptens",
}
