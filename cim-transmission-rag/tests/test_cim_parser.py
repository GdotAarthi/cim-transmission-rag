"""Unit tests for CIM XML parser."""
import pytest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.parser.cim_parser import CIMParser, CIMChunk

SAMPLE_XML = os.path.join(os.path.dirname(__file__), "../data/raw/sample_transmission.xml")


@pytest.fixture
def parser():
    return CIMParser(SAMPLE_XML)


def test_parser_loads(parser):
    assert len(parser.graph) > 0


def test_parser_returns_chunks(parser):
    chunks = list(parser.parse())
    assert len(chunks) > 0, "No chunks returned"


def test_chunk_has_required_fields(parser):
    chunks = list(parser.parse())
    for chunk in chunks:
        assert isinstance(chunk, CIMChunk)
        assert chunk.chunk_id
        assert chunk.cim_class
        assert chunk.text
        assert "CIM Class:" in chunk.text


def test_acline_segment_parsed(parser):
    chunks = list(parser.parse())
    line_chunks = [c for c in chunks if c.cim_class == "ACLineSegment"]
    assert len(line_chunks) >= 1, "Expected at least one ACLineSegment"


def test_substation_parsed(parser):
    chunks = list(parser.parse())
    sub_chunks = [c for c in chunks if c.cim_class == "Substation"]
    assert len(sub_chunks) >= 1, "Expected at least one Substation"


def test_transformer_parsed(parser):
    chunks = list(parser.parse())
    xfmr_chunks = [c for c in chunks if c.cim_class == "PowerTransformer"]
    assert len(xfmr_chunks) >= 1, "Expected at least one PowerTransformer"


def test_chunk_metadata(parser):
    chunks = list(parser.parse())
    for chunk in chunks:
        assert "cim_class" in chunk.metadata
        assert "object_id" in chunk.metadata
        assert "name" in chunk.metadata
        assert "source_file" in chunk.metadata


def test_no_duplicate_chunk_ids(parser):
    chunks = list(parser.parse())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"
