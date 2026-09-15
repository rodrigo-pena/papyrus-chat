import shutil

import pytest

from papyrus_chat.corpus import CorpusService
from papyrus_chat.retrieval.structured import StructuredCorpusSearch


def test_passage_pages_reassemble_greek_and_validate_cursors(corpus_artifact, tmp_path):
    root = tmp_path / "corpus"
    shutil.copytree(corpus_artifact, root)
    search = StructuredCorpusSearch(root / "corpus.sqlite")
    text = "δραχμὰς δέκα\n" * 1100
    row = search._connection.execute(
        "SELECT passage_id, document_id FROM passages LIMIT 1"
    ).fetchone()
    search._connection.execute(
        "UPDATE passages SET display_text=? WHERE passage_id=?", (text, row[0])
    )
    service = CorpusService(search)
    cursor = None
    restored = ""
    first_cursor = None
    while True:
        page = service.read_document_passages(row[1], cursor=cursor)
        assert len(page.windows) <= 5
        for window in page.windows:
            assert len(window.text) <= 2000
            assert window.char_end - window.char_start == len(window.text)
            assert window.source.path
            if window.passage_id == row[0]:
                restored += window.text
        cursor = page.next_cursor
        first_cursor = first_cursor or cursor
        if cursor is None:
            break
    assert restored == text
    assert first_cursor
    with pytest.raises(ValueError, match="cursor"):
        service.read_document_passages("different-document", cursor=first_cursor)
    with pytest.raises(ValueError, match="cursor"):
        service.read_document_passages(row[1], cursor="broken")
    service.close()
