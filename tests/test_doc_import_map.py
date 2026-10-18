"""Tests that subsystem documentation import maps cite existing files and accurate edges."""

import pathlib
from scripts.doc_import_map import check_docs


def test_doc_import_map_and_files_exist():
    root = pathlib.Path(__file__).resolve().parents[1]
    assert check_docs(root, verbose=True) is True
