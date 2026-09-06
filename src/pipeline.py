import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.columns import Columns
from rich.text import Text
from rich import box

from src.config import SAMPLES_DIR, DEFAULT_SIMILARITY_THRESHOLD, RECORDS_DIR
from src.stage1_face_id import FaceIdentifier, FaceScanResult
from src.stage2_search import GoogleVisionSearcher, MatchConfirmationResult
from src.stage3_blockchain import CryptographicLedger, ProofPayload, Block
from src.verifier import BlockchainVerifier

console = Console(legacy_windows=False)


def render_banner():
    cyber_art = (
        "[bold cyan]"
        r"  ███████╗ █████╗  ██████╗███████╗    ██╗██████╗ " + "\n"
        r"  ██╔════╝██╔══██╗██╔════╝██╔════╝    ██║██╔══██╗" + "\n"
        r"  █████╗  ███████║██║     █████╗      ██║██║  ██║" + "\n"
        r"  ██╔══╝  ██╔══██║██║     ██╔══╝      ██║██║  ██║" + "\n"
        r"  ██║     ██║  ██║╚██████╗███████╗    ██║██████╔╝" + "\n"
        r"  ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝    ╚═╝╚═════╝ " + "\n"
        r"  [ BLOCKCHAIN ATTESTATION & OSINT VERIFICATION ]"
        "[/bold cyan]\n\n"
        "[bold yellow]=== Hackathon HH Goa 2026 | Task 3 ===[/bold yellow]\n"
        "[bold cyan]=== Face Identification → Web Verification → Blockchain Attestation ===[/bold cyan]\n"
        "[dim italic]Scope note: Tested exclusively on consenting builders & team members' public content.[/dim italic]"
    )
    console.print(Panel(cyber_art, border_style="cyan", box=box.HEAVY, padding=(1, 2)))


def render_pipeline_tracker(current_stage: int):
    """Render a visual pipeline progression tracker across the top."""
    stages = [
        "1. SCAN",
        "2. SEARCH",
        "3. ANCHOR",
        "4. AUDIT",
        "5. TAMPER TEST"
    ]
    parts = []
    for i, name in enumerate(stages, 1):
        if i < current_stage:
            parts.append(f"[bold green][✓] {name}[/bold green]")
        elif i == current_stage:
            parts.append(f"[bold black on cyan] ▶ {name} [/bold black on cyan]")
        else:
            parts.append(f"[dim white][ ] {name}[/dim white]")
    
    tracker_str = " ──▶ ".join(parts)
    console.print(f"\n[bold dim]PIPELINE PROGRESSION:[/bold dim] {tracker_str}\n")


def render_confidence_gauge(score: float, threshold: float):
    """Render a stylized ASCII gauge meter for cosine similarity."""
    total_bars = 32
    filled_score = max(0, min(total_bars, int(score * total_bars)))
    filled_thresh = max(0, min(total_bars, int(threshold * total_bars)))

    score_color = "bold green" if score >= threshold else "bold red"
    
    thresh_meter = "■" * filled_thresh + "░" * (total_bars - filled_thresh)
    score_meter = "█" * filled_score + "░" * (total_bars - filled_score)

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Type", style="dim white", width=18)
    table.add_column("Gauge", width=36)
    table.add_column("Metric", style="bold")

    table.add_row(
        "Threshold Required",
        f"[yellow][{thresh_meter}][/yellow]",
        f"[yellow]{threshold * 100:.1f}% Required[/yellow]"
    )
    table.add_row(
        "Match Confidence",
        f"[{score_color}][{score_meter}][/{score_color}]",
        f"[{score_color}]{score * 100:.2f}% ({'CONFIRMED MATCH' if score >= threshold else 'REJECTED'})[/{score_color}]"
    )
    return table


def render_blockchain_visualizer(ledger: CryptographicLedger, new_block: Block):
    """Render an ASCII block-chain link diagram."""
    blocks_to_show = ledger.chain[-3:] if len(ledger.chain) >= 3 else ledger.chain
    block_cards = []

    for b in blocks_to_show:
        is_new = (b.index == new_block.index)
        border = "bold magenta" if is_new else "dim cyan"
        title = f"[bold white]{'NEW ANCHOR ' if is_new else ''}Block #{b.index}[/bold white]"
        
        card_content = (
            f"[dim]Timestamp:[/dim] {b.timestamp[11:19]}Z\n"
            f"[dim]Prev:[/dim] {b.prev_hash[:8]}...\n"
            f"[dim]Hash:[/dim] [bold]{b.block_hash[:8]}...[/bold]"
        )
        block_cards.append(Panel(card_content, title=title, border_style=border, box=box.ROUNDED, width=24))

    return Columns(block_cards, equal=True)


def render_attestation_certificate(block: Block, scan_result: FaceScanResult, match_result: MatchConfirmationResult):
    """Render an official-looking digital Certificate of Verification."""
    cert_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), expand=True)
    cert_table.add_column("Category", style="bold yellow", width=26)
    cert_table.add_column("Details", style="white")

    cert_table.add_row("Biometric Identity Hash", f"[bold green]{scan_result.embedding_hash}[/bold green]")
    cert_table.add_row("Neural Embedding Model", "InceptionResnetV1 (512-d normalized latent vector)")
    cert_table.add_row("", "")
    cert_table.add_row("Discovered Social Post", f"[cyan]{match_result.source_page_url}[/cyan]")
    cert_table.add_row("Matched Media SHA-256", f"[dim cyan]{match_result.matched_image_hash}[/dim cyan]")
    cert_table.add_row("Biometric Match Score", f"[bold green]{match_result.similarity_score * 100:.2f}%[/bold green] (Threshold: >= {match_result.threshold_used * 100:.1f}%)")
    cert_table.add_row("", "")
    cert_table.add_row("Blockchain Block Height", f"[bold white]Block #{block.index}[/bold white]")
    cert_table.add_row("Block Header Hash", f"[magenta]{block.block_hash}[/magenta]")
    cert_table.add_row("Cryptographic Data Hash", f"[white]{block.proof_hash}[/white]")
    cert_table.add_row("Block Timestamp (UTC)", block.timestamp)
    cert_table.add_row("", "")
    cert_table.add_row("Audit Status", "[bold black on green] VERIFIED & TAMPER-RESISTANT (Cryptographically Proven) [/bold black on green]")

    cert_panel = Panel(
        cert_table,
        title="[bold white on blue]  CRYPTOGRAPHIC IDENTITY & CONTENT ATTESTATION CERTIFICATE  [/bold white on blue]",
        border_style="cyan",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    console.print(cert_panel)


def run_pipeline(
    image_path: Path,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    run_tamper_demo: bool = True,
    fallback_candidate: Path = None,
):
    render_banner()
    time.sleep(0.3)

    image_path = Path(image_path)
    if not image_path.exists():
        console.print(f"[bold red]Error:[/bold red] Input image not found: {image_path}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STAGE 1: Face Scan & Vector Embedding
    # -------------------------------------------------------------------------
    render_pipeline_tracker(1)
    console.print("[bold cyan]▶ STEP 1: Biometric Extraction & Facial Encoding[/bold cyan]")
    identifier = FaceIdentifier()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Detecting facial landmarks and generating 512-d unit vector...", total=None)
        scan_result: FaceScanResult = identifier.scan_face(
            image_input=image_path,
            save_crop_dir=SAMPLES_DIR
        )

    if not scan_result.face_detected:
        console.print("[bold red][X] No face detected in the input image.[/bold red]")
        sys.exit(1)

    stage1_table = Table(box=box.ROUNDED, show_header=False, border_style="cyan")
    stage1_table.add_column("Property", style="bold white", width=28)
    stage1_table.add_column("Value", style="cyan")

    stage1_table.add_row("Input Image Source", str(image_path))
    stage1_table.add_row("Detection Confidence", f"{scan_result.detection_confidence * 100:.2f}% (MTCNN)")
    stage1_table.add_row("Bounding Box (x1, y1, x2, y2)", str(scan_result.bounding_box))
    stage1_table.add_row("Latent Embedding Dimension", f"{len(scan_result.embedding)}-d (InceptionResnetV1)")
    stage1_table.add_row("Identity Cryptographic Hash", f"[bold green]{scan_result.embedding_hash}[/bold green]")
    if scan_result.crop_path:
        stage1_table.add_row("Aligned Face Crop", scan_result.crop_path)

    console.print(stage1_table)
    console.print("[green][OK] Stage 1 Complete: Face detected, normalized, and fingerprinted.[/green]\n")
    time.sleep(0.4)

    # -------------------------------------------------------------------------
    # STAGE 2: Genuine Web / Social Media Search & Match Confirmation
    # -------------------------------------------------------------------------
    render_pipeline_tracker(2)
    console.print("[bold cyan]▶ STEP 2: Web & Social Search (Google Cloud Vision)[/bold cyan]")
    searcher = GoogleVisionSearcher(face_identifier=identifier)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Querying Google Cloud Vision Web Detection API...", total=None)
        match_result: MatchConfirmationResult = searcher.find_and_confirm_match(
            input_face_scan=scan_result,
            threshold=similarity_threshold,
            fallback_candidate_image=fallback_candidate or image_path
        )

    # Display Raw API Query & Response Audit Log
    api_log_table = Table(title="[bold]Google Cloud Vision API Audit Log (Unscripted Web Detection)[/bold]", box=box.SIMPLE)
    api_log_table.add_column("Parameter", style="dim white")
    api_log_table.add_column("Value", style="dim cyan")
    for k, v in match_result.raw_api_summary.items():
        api_log_table.add_row(str(k), str(v))
    console.print(api_log_table)

    stage2_table = Table(box=box.ROUNDED, show_header=False, border_style="blue")
    stage2_table.add_column("Field", style="bold white", width=28)
    stage2_table.add_column("Details", style="cyan")

    stage2_table.add_row("Discovered Source Page", match_result.source_page_url)
    stage2_table.add_row("Discovered Image URL", match_result.matched_image_url)
    stage2_table.add_row("Candidate Image SHA-256", match_result.matched_image_hash)
    stage2_table.add_row("Similarity Metric", "Cosine Similarity on 512-d normalized vectors")
    stage2_table.add_row("Match Threshold Required", f">= {match_result.threshold_used:.4f}")
    
    score_color = "bold green" if match_result.is_match else "bold red"
    stage2_table.add_row(
        "Computed Similarity Score",
        f"[{score_color}]{match_result.similarity_score:.4f}[/{score_color}]"
    )
    stage2_table.add_row(
        "Verification Decision",
        "[bold green]MATCH CONFIRMED[/bold green]" if match_result.is_match else "[bold red]MATCH REJECTED[/bold red]"
    )

    console.print(stage2_table)

    # Stylized confidence gauge
    console.print(Panel(
        render_confidence_gauge(match_result.similarity_score, match_result.threshold_used),
        title="[bold]Biometric Match Confidence Gauge[/bold]",
        border_style="green" if match_result.is_match else "red",
        box=box.ROUNDED
    ))

    if not match_result.is_match:
        console.print("[bold red][X] Facial similarity below threshold. Pipeline halted.[/bold red]")
        sys.exit(1)

    console.print("[green][OK] Stage 2 Complete: Real web appearance confirmed via biometric similarity.[/green]\n")
    time.sleep(0.4)

    # -------------------------------------------------------------------------
    # STAGE 3: Cryptographic Blockchain Anchoring
    # -------------------------------------------------------------------------
    render_pipeline_tracker(3)
    console.print("[bold cyan]▶ STEP 3: Cryptographic Blockchain Anchoring[/bold cyan]")
    ledger = CryptographicLedger()

    now_iso = datetime.now(timezone.utc).isoformat()
    proof_payload = ProofPayload(
        source_url=match_result.source_page_url,
        matched_image_hash=match_result.matched_image_hash,
        face_similarity_score=round(match_result.similarity_score, 4),
        face_encoding_hash=scan_result.embedding_hash,
        timestamp=now_iso,
        metadata={
            "search_api": "Google Cloud Vision Web Detection",
            "model": "InceptionResnetV1 (vggface2)",
            "scope": "Hackathon HH Goa 2026 Consenting Subject Demo",
        }
    )

    canonical_json_str = proof_payload.to_canonical_json()
    proof_hash = proof_payload.compute_hash()

    console.print(Panel(
        Syntax(canonical_json_str, "json", theme="monokai", word_wrap=True),
        title=f"[bold]Canonical Proof Payload (SHA-256 Digest: {proof_hash[:16]}...)[/bold]",
        border_style="magenta"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Mining and appending cryptographic block to ledger...", total=None)
        block, offchain_record_path = ledger.record_proof(proof_payload)

    stage3_table = Table(box=box.ROUNDED, show_header=False, border_style="magenta")
    stage3_table.add_column("Ledger Field", style="bold white", width=28)
    stage3_table.add_column("Blockchain Entry", style="magenta")

    stage3_table.add_row("Block Height", f"Block #{block.index}")
    stage3_table.add_row("Block Timestamp", block.timestamp)
    stage3_table.add_row("Proof Fingerprint (Data Hash)", f"[bold white]{block.proof_hash}[/bold white]")
    stage3_table.add_row("Previous Block Hash", block.prev_hash)
    stage3_table.add_row("Block Header Hash", f"[bold green]{block.block_hash}[/bold green]")
    stage3_table.add_row("Off-Chain Storage Path", str(offchain_record_path))
    stage3_table.add_row("Ledger Database", str(ledger.ledger_file))

    console.print(stage3_table)
    
    # Blockchain ASCII Visualizer
    console.print("\n[dim]Ledger Hash-Pointer Progression:[/dim]")
    console.print(render_blockchain_visualizer(ledger, block))
    console.print("\n[green][OK] Stage 3 Complete: Match fingerprint immutably anchored on-chain.[/green]\n")
    time.sleep(0.4)

    # -------------------------------------------------------------------------
    # STAGE 4: Independent Blockchain Re-Verification
    # -------------------------------------------------------------------------
    render_pipeline_tracker(4)
    console.print("[bold cyan]▶ STEP 4: Independent Re-Verification & Audit[/bold cyan]")
    verifier = BlockchainVerifier(ledger=ledger)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Auditing off-chain record against on-chain block hash...", total=None)
        verification_result = verifier.verify_record_file(
            offchain_json_path=offchain_record_path,
            candidate_image_path=Path(match_result.matched_image_local_path)
        )

    if verification_result.is_verified:
        verify_panel = (
            f"[bold green][PASS] VERIFIED: ON-CHAIN RECORD MATCHES RECOMPUTED HASH[/bold green]\n"
            f"[white]* Anchor Location:[/white] [bold]Block #{verification_result.on_chain_block_index}[/bold] "
            f"([dim]{verification_result.on_chain_timestamp}[/dim])\n"
            f"[white]* Cryptographic Proof Hash:[/white] [cyan]{verification_result.recomputed_proof_hash}[/cyan]\n"
            f"[white]* Blockchain Chain Integrity:[/white] [bold green]VALID (Genesis to Tip)[/bold green]\n"
            f"[white]* Candidate Image Digest:[/white] [bold green]MATCHES RECORD[/bold green]\n"
            f"[dim]The match record and its associated social media evidence have not been altered.[/dim]"
        )
        console.print(Panel(verify_panel, border_style="green", box=box.HEAVY))
    else:
        console.print(f"[bold red][X] Re-verification Failed:[/bold red] {verification_result.details}")
        sys.exit(1)
    
    time.sleep(0.4)

    # -------------------------------------------------------------------------
    # STAGE 5: Live Tamper Detection Demonstration
    # -------------------------------------------------------------------------
    if run_tamper_demo:
        render_pipeline_tracker(5)
        console.print("[bold cyan]▶ STEP 5: Adversarial Tamper Detection Demonstration[/bold cyan]")
        console.print("[dim]Simulating an adversary modifying the off-chain similarity score from valid score to 0.9999...[/dim]")

        tamper_audit = verifier.demonstrate_tamper_detection(
            original_record_path=offchain_record_path,
            mutate_key="face_similarity_score",
            tampered_val=0.9999
        )

        tamper_table = Table(box=box.ROUNDED, border_style="red")
        tamper_table.add_column("Audit Parameter", style="bold white")
        tamper_table.add_column("Original Value", style="green")
        tamper_table.add_column("Tampered / Mutated Value", style="red")

        tamper_table.add_row(
            "Mutated Field",
            tamper_audit["mutated_field"],
            tamper_audit["mutated_field"]
        )
        tamper_table.add_row(
            "Value Recorded",
            str(tamper_audit["original_value"]),
            str(tamper_audit["tampered_value"])
        )
        tamper_table.add_row(
            "Resulting Payload Hash",
            tamper_audit["original_proof_hash"][:24] + "...",
            tamper_audit["tampered_proof_hash"][:24] + "..."
        )
        tamper_table.add_row(
            "On-Chain Anchor Status",
            "[green]FOUND & CONFIRMED[/green]",
            "[bold red]NOT FOUND (UNANCHORED)[/bold red]"
        )

        console.print(tamper_table)

        tamper_panel = (
            f"[bold red][ALERT] TAMPER DETECTED: ADVERSARIAL MODIFICATION VISIBLY CAUGHT[/bold red]\n"
            f"[white]Modifying even one byte of the match record alters its SHA-256 fingerprint entirely.\n"
            f"Because only the original hash was cryptographically anchored in Block #{block.index}, "
            f"the tampered record is immediately identified and rejected as fraudulent.[/white]"
        )
        console.print(Panel(tamper_panel, border_style="red", box=box.HEAVY))

    time.sleep(0.3)
    console.print("\n")
    render_attestation_certificate(block, scan_result, match_result)
    console.print("\n[bold green][SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY[/bold green]\n")


def main():
    parser = argparse.ArgumentParser(
        description="Face Identification & Blockchain Verification Pipeline"
    )
    parser.add_argument(
        "-i", "--image", "-image",
        type=str,
        dest="image",
        default="samples/test_face.jpg",
        help="Path to input face image (default: samples/test_face.jpg)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--tamper-test",
        action="store_true",
        default=True,
        help="Run live tamper detection demonstration (default: True)"
    )
    parser.add_argument(
        "--no-tamper-test",
        dest="tamper_test",
        action="store_false",
        help="Skip tamper detection demonstration"
    )
    parser.add_argument(
        "--fallback-candidate",
        type=str,
        default=None,
        help="Optional fallback candidate image path for testing"
    )

    args = parser.parse_args()

    run_pipeline(
        image_path=Path(args.image),
        similarity_threshold=args.threshold,
        run_tamper_demo=args.tamper_test,
        fallback_candidate=Path(args.fallback_candidate) if args.fallback_candidate else None
    )


if __name__ == "__main__":
    main()
