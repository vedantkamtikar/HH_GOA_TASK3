from datetime import datetime, timezone
import tempfile
from pathlib import Path
import pytest
from src.stage3_blockchain import CryptographicLedger, ProofPayload
from src.verifier import BlockchainVerifier


@pytest.fixture
def temp_ledger(tmp_path):
    ledger_file = tmp_path / "test_blockchain.json"
    return CryptographicLedger(ledger_file=ledger_file)


def test_genesis_block(temp_ledger):
    assert len(temp_ledger.chain) == 1
    genesis = temp_ledger.chain[0]
    assert genesis.index == 0
    assert genesis.prev_hash == "0" * 64
    valid, _ = temp_ledger.verify_chain_integrity()
    assert valid is True


def test_record_proof_and_verification(temp_ledger, tmp_path):
    payload = ProofPayload(
        source_url="https://example.com/test-post",
        matched_image_hash="a" * 64,
        face_similarity_score=0.89,
        face_encoding_hash="b" * 64,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

    block, record_path = temp_ledger.record_proof(payload)
    assert block.index == 1
    assert block.prev_hash == temp_ledger.chain[0].block_hash
    assert record_path.exists()

    verifier = BlockchainVerifier(ledger=temp_ledger)
    res = verifier.verify_record_file(record_path)
    assert res.is_verified is True
    assert res.status == "VERIFIED"
    assert res.on_chain_block_index == 1


def test_tamper_detection(temp_ledger, tmp_path):
    payload = ProofPayload(
        source_url="https://example.com/test-post",
        matched_image_hash="c" * 64,
        face_similarity_score=0.75,
        face_encoding_hash="d" * 64,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    block, record_path = temp_ledger.record_proof(payload)

    verifier = BlockchainVerifier(ledger=temp_ledger)
    tamper_result = verifier.demonstrate_tamper_detection(
        original_record_path=record_path,
        mutate_key="face_similarity_score",
        tampered_val=0.999
    )
    assert tamper_result["tamper_detected"] is True
    assert tamper_result["on_chain_found_for_tampered"] is False
