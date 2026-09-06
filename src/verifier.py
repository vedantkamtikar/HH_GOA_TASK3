import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from src.stage3_blockchain import CryptographicLedger, ProofPayload, Block


@dataclass
class VerificationResult:
    """Result of an independent blockchain re-verification audit."""
    is_verified: bool
    status: str
    recomputed_proof_hash: str
    on_chain_block_index: Optional[int]
    on_chain_timestamp: Optional[str]
    chain_integrity_valid: bool
    image_hash_valid: Optional[bool]
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "status": self.status,
            "recomputed_proof_hash": self.recomputed_proof_hash,
            "on_chain_block_index": self.on_chain_block_index,
            "on_chain_timestamp": self.on_chain_timestamp,
            "chain_integrity_valid": self.chain_integrity_valid,
            "image_hash_valid": self.image_hash_valid,
            "details": self.details,
        }


class BlockchainVerifier:
    """
    Independent verification and audit engine.
    Audits off-chain payloads against on-chain ledger anchors and provides
    tamper detection testing.
    """

    def __init__(self, ledger: Optional[CryptographicLedger] = None):
        self.ledger = ledger or CryptographicLedger()

    def verify_record_file(
        self,
        offchain_json_path: Path,
        candidate_image_path: Optional[Path] = None
    ) -> VerificationResult:
        """
        Independently re-verify an off-chain record against the blockchain ledger.
        """
        offchain_path = Path(offchain_json_path)
        if not offchain_path.exists():
            return VerificationResult(
                is_verified=False,
                status="FILE_NOT_FOUND",
                recomputed_proof_hash="",
                on_chain_block_index=None,
                on_chain_timestamp=None,
                chain_integrity_valid=False,
                image_hash_valid=None,
                details=f"Off-chain record file does not exist: {offchain_path}"
            )

        # 1. Read stored payload
        try:
            with open(offchain_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            payload = ProofPayload(**raw_data)
        except Exception as e:
            return VerificationResult(
                is_verified=False,
                status="CORRUPTED_JSON",
                recomputed_proof_hash="",
                on_chain_block_index=None,
                on_chain_timestamp=None,
                chain_integrity_valid=False,
                image_hash_valid=None,
                details=f"Malformed record JSON payload: {e}"
            )

        # 2. Recompute cryptographic proof hash
        recomputed_hash = payload.compute_hash()

        # 3. Verify candidate image hash if candidate image is provided
        image_hash_valid = None
        if candidate_image_path and Path(candidate_image_path).exists():
            with open(candidate_image_path, 'rb') as img_f:
                computed_img_hash = hashlib.sha256(img_f.read()).hexdigest()
            image_hash_valid = (computed_img_hash == payload.matched_image_hash)
            if not image_hash_valid:
                return VerificationResult(
                    is_verified=False,
                    status="IMAGE_TAMPERED",
                    recomputed_proof_hash=recomputed_hash,
                    on_chain_block_index=None,
                    on_chain_timestamp=None,
                    chain_integrity_valid=False,
                    image_hash_valid=False,
                    details=f"Candidate image SHA-256 ({computed_img_hash[:16]}...) does not match recorded hash ({payload.matched_image_hash[:16]}...)"
                )

        # 4. Check chain integrity
        chain_valid, chain_reason = self.ledger.verify_chain_integrity()
        if not chain_valid:
            return VerificationResult(
                is_verified=False,
                status="LEDGER_COMPROMISED",
                recomputed_proof_hash=recomputed_hash,
                on_chain_block_index=None,
                on_chain_timestamp=None,
                chain_integrity_valid=False,
                image_hash_valid=image_hash_valid,
                details=f"Underlying blockchain ledger failed integrity check: {chain_reason}"
            )

        # 5. Query blockchain for matching proof hash
        block = self.ledger.find_proof(recomputed_hash)
        if not block:
            return VerificationResult(
                is_verified=False,
                status="HASH_NOT_ANCHORED",
                recomputed_proof_hash=recomputed_hash,
                on_chain_block_index=None,
                on_chain_timestamp=None,
                chain_integrity_valid=True,
                image_hash_valid=image_hash_valid,
                details=f"Proof hash {recomputed_hash[:16]}... was not found anchored in any block"
            )

        # Success: fully verified
        return VerificationResult(
            is_verified=True,
            status="VERIFIED",
            recomputed_proof_hash=recomputed_hash,
            on_chain_block_index=block.index,
            on_chain_timestamp=block.timestamp,
            chain_integrity_valid=True,
            image_hash_valid=image_hash_valid,
            details=f"Record securely verified at Block #{block.index} [Timestamp: {block.timestamp}]"
        )

    def demonstrate_tamper_detection(
        self,
        original_record_path: Path,
        mutate_key: str = "face_similarity_score",
        tampered_val: Any = 0.9999
    ) -> Dict[str, Any]:
        """
        Simulate malicious tampering on an off-chain record and show that
        independent verification catches the violation.
        """
        offchain_path = Path(original_record_path)
        with open(offchain_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_val = data.get(mutate_key)
        original_payload = ProofPayload(**data)
        original_hash = original_payload.compute_hash()

        # Apply intentional tampering
        tampered_data = dict(data)
        tampered_data[mutate_key] = tampered_val
        tampered_payload = ProofPayload(**tampered_data)
        tampered_hash = tampered_payload.compute_hash()

        # Query ledger with tampered hash
        tampered_block = self.ledger.find_proof(tampered_hash)
        caught = (tampered_block is None)

        return {
            "tamper_detected": caught,
            "mutated_field": mutate_key,
            "original_value": original_val,
            "tampered_value": tampered_val,
            "original_proof_hash": original_hash,
            "tampered_proof_hash": tampered_hash,
            "on_chain_found_for_tampered": bool(tampered_block),
            "verdict": "TAMPER DETECTED: Modified payload produces an unanchored cryptographic hash" if caught else "VULNERABLE"
        }
