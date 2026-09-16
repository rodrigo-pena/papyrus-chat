"""Distinct-document reciprocal-rank fusion with explicit stable ordering."""

from collections.abc import Mapping, Sequence

from papyrus_chat.retrieval.discovery.models import DiscoveryChannel

RRF_CONSTANT = 60


def fuse_rankings(
    rankings: Mapping[DiscoveryChannel, Sequence[str]],
    identities: Mapping[str, tuple[str, str]],
) -> list[tuple[str, float, tuple[DiscoveryChannel, ...]]]:
    scores: dict[str, float] = {}
    channels: dict[str, list[DiscoveryChannel]] = {}
    for channel, documents in rankings.items():
        for rank, document_id in enumerate(tuple(dict.fromkeys(documents)), 1):
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (RRF_CONSTANT + rank)
            channels.setdefault(document_id, []).append(channel)
    return [
        (document_id, scores[document_id], tuple(channels[document_id]))
        for document_id in sorted(scores, key=lambda d: (-scores[d], identities[d]))
    ]
