"""Canonical external link generation."""

from papyrus_chat.web.links import papyri_info_url


def test_papyri_info_url_prefers_tm() -> None:
    assert (
        papyri_info_url([("ddb", "p.tebt;1;7"), ("TM", "3643")])
        == "https://papyri.info/current/3643"
    )


def test_papyri_info_url_is_case_insensitive() -> None:
    assert papyri_info_url([("tm", "7444")]) == "https://papyri.info/current/7444"


def test_papyri_info_url_returns_none_without_tm() -> None:
    assert papyri_info_url([("ddb", "p.tebt;1;7"), ("HGV", "3643")]) is None


def test_papyri_info_url_strips_whitespace() -> None:
    assert papyri_info_url([("TM", "  7444  ")]) == "https://papyri.info/current/7444"


def test_papyri_info_url_skips_empty_tm_value() -> None:
    assert papyri_info_url([("TM", "   "), ("HGV", "3643")]) is None
