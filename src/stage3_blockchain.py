import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from src.config import LEDGER_FILE, RECORDS_DIR


@dataclass
class ProofPayload:
    """Canonical payload representation of a verified face match."""
    source_url: str
    matched_image_hash: str
    face_similarity_score: float
    face_encoding_hash: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def to_canonical_json(self) -> str:
        """Serialize payload into a deterministic, sorted canonical JSON string."""
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))

    def compute_hash(self) -> str:
        """Compute SHA-256 cryptographic digest of the canonical payload."""
        return hashlib.sha256(self.to_canonical_json().encode('utf-8')).hexdigest()


@dataclass
class Block:
    """A cryptographic block in the local blockchain ledger."""
    index: int
    timestamp: str
    proof_hash: str
    payload_summary: Dict[str, Any]
    prev_hash: str
    nonce: int
    block_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CryptographicLedger:
    """
    Immutable local blockchain ledger.
    Provides verifiable cryptographic blocks with SHA-256 hash chaining,
    previous block pointers, timestamps, and Merkle/payload digests.
    """

    def __init__(self, ledger_file: Optional[Path] = None):
        self.ledger_file = Path(ledger_file) if ledger_file else LEDGER_FILE
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        self.chain: List[Block] = []
        self._load_or_initialize()

    @staticmethod
    def calculate_block_hash(index: int, timestamp: str, proof_hash: str, prev_hash: str, nonce: int = 0) -> str:
        """Deterministic block hash computation."""
        header = f"{index}|{timestamp}|{proof_hash}|{prev_hash}|{nonce}"
        return hashlib.sha256(header.encode('utf-8')).hexdigest()

    def _create_genesis_block(self) -> Block:
        """Construct the genesis block (Block 0)."""
        timestamp = "2026-01-01T00:00:00Z"
        proof_hash = "0" * 64
        prev_hash = "0" * 64
        block_hash = self.calculate_block_hash(0, timestamp, proof_hash, prev_hash, 0)
        return Block(
            index=0,
            timestamp=timestamp,
            proof_hash=proof_hash,
            payload_summary={"genesis": "Identity Verification Genesis Block"},
            prev_hash=prev_hash,
            nonce=0,
            block_hash=block_hash,
        )

    def _load_or_initialize(self):
        """Load blockchain from disk or create fresh genesis block."""
        if self.ledger_file.exists():
            try:
                with open(self.ledger_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chain = [Block(**b) for b in data]
                # Validate chain upon load
                is_valid, reason = self.verify_chain_integrity()
                if not is_valid:
                    raise ValueError(f"Existing blockchain file is corrupted: {reason}")
                return
            except Exception as e:
                print(f"[Warning] Failed loading ledger ({e}). Initializing fresh chain.")

        # Initialize with genesis block
        genesis = self._create_genesis_block()
        self.chain = [genesis]
        self._save()

    def _save(self):
        """Persist chain to disk."""
        with open(self.ledger_file, 'w', encoding='utf-8') as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2)

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def record_proof(self, payload: ProofPayload, save_offchain: bool = True) -> Tuple[Block, Path]:
        """
        Anchor a verified match payload to the blockchain.
        Returns the created block and the path to the off-chain record file.
        """
        proof_hash = payload.compute_hash()

        # Check if already anchored
        existing = self.find_proof(proof_hash)
        if existing:
            offchain_path = RECORDS_DIR / f"{proof_hash}.json"
            return existing, offchain_path

        latest = self.get_latest_block()
        new_index = latest.index + 1
        now_ts = datetime.now(timezone.utc).isoformat()

        # Summary for fast ledger inspection without leaking full raw data
        summary = {
            "source_url": payload.source_url,
            "face_similarity_score": payload.face_similarity_score,
            "matched_image_hash": payload.matched_image_hash,
        }

        block_hash = self.calculate_block_hash(
            index=new_index,
            timestamp=now_ts,
            proof_hash=proof_hash,
            prev_hash=latest.block_hash,
            nonce=0,
        )

        new_block = Block(
            index=new_index,
            timestamp=now_ts,
            proof_hash=proof_hash,
            payload_summary=summary,
            prev_hash=latest.block_hash,
            nonce=0,
            block_hash=block_hash,
        )

        self.chain.append(new_block)
        self._save()

        # Save off-chain full record
        offchain_path = RECORDS_DIR / f"{proof_hash}.json"
        if save_offchain:
            with open(offchain_path, 'w', encoding='utf-8') as f:
                f.write(payload.to_canonical_json())

        return new_block, offchain_path

    def find_proof(self, proof_hash: str) -> Optional[Block]:
        """Locate block containing the given proof hash."""
        for block in self.chain:
            if block.proof_hash == proof_hash:
                return block
        return None

    def verify_chain_integrity(self) -> Tuple[bool, str]:
        """Verify hash pointer integrity across the entire blockchain."""
        if not self.chain:
            return False, "Chain is empty"

        # Verify genesis
        genesis = self.chain[0]
        if genesis.index != 0 or genesis.prev_hash != "0" * 64:
            return False, "Invalid genesis block parameters"

        for i in range(1, len(self.chain)):
            current = self.chain[i]
            prev = self.chain[i - 1]

            # Check link to previous block
            if current.prev_hash != prev.block_hash:
                return False, f"Broken chain pointer at block {current.index}: prev_hash mismatch"

            # Check block hash validity
            expected_hash = self.calculate_block_hash(
                index=current.index,
                timestamp=current.timestamp,
                proof_hash=current.proof_hash,
                prev_hash=current.prev_hash,
                nonce=current.nonce,
            )
            if current.block_hash != expected_hash:
                return False, f"Tampered block hash at block {current.index}"

        return True, "Chain integrity valid"
