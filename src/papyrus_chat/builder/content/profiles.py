"""Bounded document representations assembled only from attributed sources."""

from collections import defaultdict
from collections.abc import Iterator, Sequence

from papyrus_chat.artifact.records import (
    ComponentRecord,
    DocumentRecord,
    PassageRecord,
    ProfilePassageReference,
    SemanticProfileRecord,
)
from papyrus_chat.semantic.tokenization import ContentTokenizer, bounded_prefix

PROFILE_TOKENS = 448
METADATA_TOKENS = 128
_DESCRIPTIVE_KEYS = (
    "subject",
    "commentary",
    "origin",
    "origPlace",
    "material",
    "origDate",
    "term_overview",
    "term_religion",
)


def _metadata_parts(
    document: DocumentRecord,
    components: Sequence[ComponentRecord],
) -> Iterator[tuple[str, str | None]]:
    yield document.title, None
    ordered = sorted(components, key=lambda c: c.component_id)
    for component in ordered:
        if component.title != document.title:
            yield component.title, component.component_id
    for key in _DESCRIPTIVE_KEYS:
        if value := document.metadata.get(key):
            yield f"{key}: {value}", None
        for component in ordered:
            for value in sorted(set(component.metadata.get(key, ()))):
                yield f"{key}: {value}", component.component_id


def document_profile(
    document: DocumentRecord,
    passages: Sequence[PassageRecord],
    components: Sequence[ComponentRecord],
    tokenizer: ContentTokenizer,
) -> SemanticProfileRecord:
    profile_text = ""
    component_ids: set[str] = set()
    seen: set[str] = set()
    for value, component_id in _metadata_parts(document, components):
        if value in seen:
            continue
        seen.add(value)
        separator = "\n" if profile_text else ""
        available = METADATA_TOKENS - tokenizer.count(profile_text + separator)
        part = bounded_prefix(value, available, tokenizer).strip()
        while part and tokenizer.count(profile_text + separator + part) > METADATA_TOKENS:
            part = part[:-1]
        if not part:
            break
        profile_text += separator + part
        if component_id is not None:
            component_ids.add(component_id)

    groups: dict[tuple[str, str], list[PassageRecord]] = defaultdict(list)
    for passage in sorted(passages, key=lambda p: (p.sequence, p.passage_id)):
        if passage.display_text.strip():
            groups[(passage.kind, passage.language or "und")].append(passage)
    references: list[ProfilePassageReference] = []
    ordered_groups = sorted(groups.items())
    for group_index, ((kind, language), group) in enumerate(ordered_groups):
        group_budget = (PROFILE_TOKENS - tokenizer.count(profile_text)) // (
            len(ordered_groups) - group_index
        )
        for passage_index, passage in enumerate(group):
            budget = group_budget // (len(group) - passage_index)
            label = f"\n{kind} ({language}): "
            excerpt = bounded_prefix(
                passage.display_text,
                budget - tokenizer.count(label),
                tokenizer,
            )
            # Check the entire concatenation; tokenization need not be additive.
            while excerpt and tokenizer.count(profile_text + label + excerpt) > PROFILE_TOKENS:
                excerpt = excerpt[:-1]
            if not excerpt.strip():
                continue
            before = tokenizer.count(profile_text)
            profile_text += label + excerpt
            group_budget -= tokenizer.count(profile_text) - before
            references.append(
                ProfilePassageReference(
                    passage_id=passage.passage_id,
                    char_start=0,
                    char_end=len(excerpt),
                )
            )
    if not profile_text.strip():
        profile_text = bounded_prefix(document.document_id, METADATA_TOKENS, tokenizer)
    return SemanticProfileRecord(
        profile_id=f"{document.document_id}#profile-v1",
        document_id=document.document_id,
        profile_text=profile_text,
        metadata_only=not groups,
        passages=tuple(references),
        component_ids=tuple(sorted(component_ids)),
    )
