import numpy as np
from pathlib import Path
import pytest
from src.stage1_face_id import FaceIdentifier, FaceScanResult


def test_face_identifier_initialization():
    identifier = FaceIdentifier()
    assert identifier is not None
    assert identifier.resnet is not None


def test_cosine_similarity_identical():
    v = np.random.randn(512)
    sim = FaceIdentifier.compute_cosine_similarity(v, v)
    assert pytest.approx(sim, 0.0001) == 1.0


def test_cosine_similarity_orthogonal():
    v1 = np.zeros(512)
    v1[0] = 1.0
    v2 = np.zeros(512)
    v2[1] = 1.0
    sim = FaceIdentifier.compute_cosine_similarity(v1, v2)
    assert pytest.approx(sim, 0.0001) == 0.0


def test_embedding_hash_deterministic():
    v = np.random.randn(512)
    h1 = FaceIdentifier.calculate_embedding_hash(v)
    h2 = FaceIdentifier.calculate_embedding_hash(v)
    assert h1 == h2
    assert len(h1) == 64


def test_scan_face_on_sample_image():
    sample_path = Path("samples/test_face.jpg")
    if not sample_path.exists():
        pytest.skip("Sample image not present")
    identifier = FaceIdentifier()
    res = identifier.scan_face(sample_path)
    assert res.face_detected is True
    assert res.bounding_box is not None
    assert len(res.embedding) == 512
    assert res.embedding_hash is not None
