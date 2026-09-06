from pathlib import Path
import pytest
from src.stage1_face_id import FaceIdentifier
from src.stage2_search import GoogleVisionSearcher, SearchCandidate


def test_searcher_initialization():
    searcher = GoogleVisionSearcher()
    assert searcher.identifier is not None


def test_match_confirmation_threshold_pass():
    identifier = FaceIdentifier()
    sample_path = Path("samples/test_face.jpg")
    if not sample_path.exists():
        pytest.skip("Sample image not present")

    scan = identifier.scan_face(sample_path)
    searcher = GoogleVisionSearcher(face_identifier=identifier)

    result = searcher.find_and_confirm_match(
        input_face_scan=scan,
        threshold=0.65,
        fallback_candidate_image=sample_path
    )
    assert result.is_match is True
    assert result.similarity_score >= 0.65
    assert len(result.matched_image_hash) == 64


def test_match_confirmation_threshold_reject():
    identifier = FaceIdentifier()
    sample_path = Path("samples/test_face.jpg")
    if not sample_path.exists():
        pytest.skip("Sample image not present")

    scan = identifier.scan_face(sample_path)
    searcher = GoogleVisionSearcher(face_identifier=identifier)

    # With impossibly high threshold, match is rejected
    result = searcher.find_and_confirm_match(
        input_face_scan=scan,
        threshold=1.5,
        fallback_candidate_image=sample_path
    )
    assert result.is_match is False
