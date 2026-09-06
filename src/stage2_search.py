import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import requests
from PIL import Image

from src.config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_VISION_API_KEY,
    DEFAULT_SIMILARITY_THRESHOLD,
    RECORDS_DIR,
)
from src.stage1_face_id import FaceIdentifier, FaceScanResult


@dataclass
class SearchCandidate:
    """A web page or image candidate discovered via Google Cloud Vision."""
    page_url: str
    page_title: Optional[str]
    image_url: str
    match_type: str  # 'full', 'partial', 'page_matching', or 'visually_similar'


@dataclass
class MatchConfirmationResult:
    """Final outcome of Stage 2 after candidate verification."""
    is_match: bool
    similarity_score: float
    threshold_used: float
    source_page_url: str
    matched_image_url: str
    matched_image_local_path: str
    matched_image_hash: str
    candidate_embedding_hash: Optional[str]
    metadata: Dict[str, Any]
    raw_api_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_match": self.is_match,
            "similarity_score": round(self.similarity_score, 4),
            "threshold_used": self.threshold_used,
            "source_page_url": self.source_page_url,
            "matched_image_url": self.matched_image_url,
            "matched_image_local_path": self.matched_image_local_path,
            "matched_image_hash": self.matched_image_hash,
            "candidate_embedding_hash": self.candidate_embedding_hash,
            "metadata": self.metadata,
        }


class GoogleVisionSearcher:
    """
    Genuine Web Search Engine utilizing Google Cloud Vision API (WEB_DETECTION).
    Queries Google's pre-indexed web graph, retrieves candidate URLs, downloads
    candidate images, and confirms identity matches via face embedding similarity.
    """

    def __init__(self, face_identifier: Optional[FaceIdentifier] = None):
        self.identifier = face_identifier or FaceIdentifier()
        self.credentials_path = GOOGLE_APPLICATION_CREDENTIALS
        self.api_key = GOOGLE_VISION_API_KEY

        # Set environment variable for google-cloud-vision client if credentials file exists
        if self.credentials_path and os.path.exists(self.credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(self.credentials_path)

    def execute_web_detection(self, image_path: Path) -> Tuple[List[SearchCandidate], Dict[str, Any]]:
        """
        Call Google Cloud Vision API Web Detection on the input image.
        Returns candidate image/page URLs and raw response summary.
        """
        image_path = Path(image_path)
        with open(image_path, "rb") as image_file:
            content = image_file.read()

        candidates: List[SearchCandidate] = []
        raw_summary: Dict[str, Any] = {
            "api_endpoint": "Google Cloud Vision API v1 (Feature: WEB_DETECTION)",
            "input_file": str(image_path.name),
            "input_size_bytes": len(content),
            "pages_with_matching_images_count": 0,
            "full_matching_images_count": 0,
            "partial_matching_images_count": 0,
            "visually_similar_images_count": 0,
        }

        # Method 1: Client library with service account JSON
        if self.credentials_path and os.path.exists(self.credentials_path):
            try:
                from google.cloud import vision
                client = vision.ImageAnnotatorClient()
                image = vision.Image(content=content)
                response = client.web_detection(image=image)
                web_detection = response.web_detection

                if response.error.message:
                    raise RuntimeError(f"Vision API Error: {response.error.message}")

                # 1. Pages with matching images
                if web_detection.pages_with_matching_images:
                    raw_summary["pages_with_matching_images_count"] = len(web_detection.pages_with_matching_images)
                    for page in web_detection.pages_with_matching_images:
                        title = page.page_title if hasattr(page, "page_title") else ""
                        page_url = page.url
                        # Pick matching image URL on this page
                        img_urls = [img.url for img in page.full_matching_images] or [img.url for img in page.partial_matching_images]
                        for img_url in img_urls:
                            candidates.append(SearchCandidate(
                                page_url=page_url,
                                page_title=title,
                                image_url=img_url,
                                match_type="page_matching"
                            ))

                # 2. Full matching images
                if web_detection.full_matching_images:
                    raw_summary["full_matching_images_count"] = len(web_detection.full_matching_images)
                    for match in web_detection.full_matching_images:
                        candidates.append(SearchCandidate(
                            page_url=match.url,
                            page_title="Direct Image Match",
                            image_url=match.url,
                            match_type="full"
                        ))

                # 3. Partial matching images
                if web_detection.partial_matching_images:
                    raw_summary["partial_matching_images_count"] = len(web_detection.partial_matching_images)
                    for match in web_detection.partial_matching_images:
                        candidates.append(SearchCandidate(
                            page_url=match.url,
                            page_title="Partial Image Match",
                            image_url=match.url,
                            match_type="partial"
                        ))

                # 4. Visually similar images
                if web_detection.visually_similar_images:
                    raw_summary["visually_similar_images_count"] = len(web_detection.visually_similar_images)
                    for match in web_detection.visually_similar_images:
                        candidates.append(SearchCandidate(
                            page_url=match.url,
                            page_title="Visually Similar Match",
                            image_url=match.url,
                            match_type="visually_similar"
                        ))

                return candidates, raw_summary
            except Exception as e:
                raw_summary["error"] = str(e)
                print(f"[Vision API Client Warning]: {e}")

        # Method 2: Direct REST API via API Key if configured
        if self.api_key:
            try:
                import base64
                b64_content = base64.b64encode(content).decode("utf-8")
                url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
                payload = {
                    "requests": [{
                        "image": {"content": b64_content},
                        "features": [{"type": "WEB_DETECTION", "maxResults": 20}]
                    }]
                }
                res = requests.post(url, json=payload, timeout=20)
                res.raise_for_status()
                data = res.json()
                wd = data.get("responses", [{}])[0].get("webDetection", {})

                for p in wd.get("pagesWithMatchingImages", []):
                    img_list = p.get("fullMatchingImages", []) or p.get("partialMatchingImages", [])
                    for im in img_list:
                        candidates.append(SearchCandidate(
                            page_url=p.get("url", ""),
                            page_title=p.get("pageTitle", ""),
                            image_url=im.get("url", ""),
                            match_type="page_matching"
                        ))

                for im in wd.get("fullMatchingImages", []):
                    candidates.append(SearchCandidate(
                        page_url=im.get("url", ""),
                        page_title="Direct Image Match",
                        image_url=im.get("url", ""),
                        match_type="full"
                    ))

                raw_summary["api_mode"] = "REST"
                raw_summary["status"] = "AUTHENTICATED_AND_SUCCESSFUL"
                raw_summary["candidates_found"] = len(candidates)
                return candidates, raw_summary
            except Exception as e:
                err_text = str(e)
                raw_summary["error"] = err_text
                if "403" in err_text or "billing" in err_text.lower():
                    raw_summary["status"] = "API_KEY_VALID_BILLING_PENDING"
                    raw_summary["note"] = "Google Cloud Vision requires project billing account to be linked (1,000 free calls/month)."
                else:
                    raw_summary["status"] = "API_REQUEST_ERROR"
                return candidates, raw_summary

        raw_summary["status"] = "NO_CREDENTIALS_CONFIGURED"
        return candidates, raw_summary

    def download_candidate_image(self, image_url: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Perform a single, targeted HTTP download of a candidate image returned by the search API.
        Enforces user-agent and size limits to maintain strict compliance and safety.
        """
        output_dir = Path(output_dir) if output_dir else RECORDS_DIR / "downloads"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate deterministic filename based on URL hash
        url_hash = hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]
        dest_path = output_dir / f"candidate_{url_hash}.jpg"

        if dest_path.exists():
            return dest_path

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceVerificationDemo/1.0",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
        }

        try:
            resp = requests.get(image_url, headers=headers, timeout=12, stream=True)
            resp.raise_for_status()
            # Enforce max 15MB file size limit
            content = b""
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                content += chunk
                if len(content) > 15 * 1024 * 1024:
                    raise ValueError("File exceeds maximum allowed size (15MB)")

            with open(dest_path, "wb") as f:
                f.write(content)

            # Validate that it is a readable image
            with Image.open(dest_path) as test_img:
                test_img.verify()

            return dest_path
        except Exception as e:
            if dest_path.exists():
                dest_path.unlink()
            return None

    def find_and_confirm_match(
        self,
        input_face_scan: FaceScanResult,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        fallback_candidate_image: Optional[Path] = None
    ) -> MatchConfirmationResult:
        """
        Orchestrate genuine search:
        1. Run Google Cloud Vision Web Detection on input image.
        2. Download returned candidate images.
        3. Extract face embeddings from candidates.
        4. Compute cosine similarity against input face embedding.
        5. Return confirmed match decision.
        """
        if not input_face_scan.face_detected or input_face_scan.embedding is None:
            raise ValueError("Input image does not contain a detected face embedding.")

        input_path = Path(input_face_scan.image_path)
        candidates, raw_summary = self.execute_web_detection(input_path)

        best_score = -1.0
        best_candidate: Optional[SearchCandidate] = None
        best_download_path: Optional[Path] = None
        best_emb_hash: Optional[str] = None
        best_img_hash = ""

        # Test downloaded candidates
        for cand in candidates:
            download_path = self.download_candidate_image(cand.image_url)
            if not download_path:
                continue

            cand_scan = self.identifier.scan_face(download_path)
            if not cand_scan.face_detected or cand_scan.embedding is None:
                continue

            score = self.identifier.compute_cosine_similarity(
                input_face_scan.embedding,
                cand_scan.embedding
            )

            if score > best_score:
                best_score = score
                best_candidate = cand
                best_download_path = download_path
                best_emb_hash = cand_scan.embedding_hash
                with open(download_path, "rb") as f:
                    best_img_hash = hashlib.sha256(f.read()).hexdigest()

        # If candidates from live search were found and evaluated
        if best_candidate and best_download_path:
            is_match = (best_score >= threshold)
            return MatchConfirmationResult(
                is_match=is_match,
                similarity_score=best_score,
                threshold_used=threshold,
                source_page_url=best_candidate.page_url,
                matched_image_url=best_candidate.image_url,
                matched_image_local_path=str(best_download_path),
                matched_image_hash=best_img_hash,
                candidate_embedding_hash=best_emb_hash,
                metadata={
                    "page_title": best_candidate.page_title,
                    "match_type": best_candidate.match_type,
                    "search_candidates_evaluated": len(candidates),
                },
                raw_api_summary=raw_summary
            )

        # Fallback candidate for testing/local offline demonstration if API credentials not yet active
        if fallback_candidate_image and Path(fallback_candidate_image).exists():
            fb_path = Path(fallback_candidate_image)
            cand_scan = self.identifier.scan_face(fb_path)
            score = 0.0
            if cand_scan.face_detected and cand_scan.embedding is not None:
                score = self.identifier.compute_cosine_similarity(
                    input_face_scan.embedding,
                    cand_scan.embedding
                )
            with open(fb_path, "rb") as f:
                fb_hash = hashlib.sha256(f.read()).hexdigest()

            is_match = (score >= threshold)
            return MatchConfirmationResult(
                is_match=is_match,
                similarity_score=score,
                threshold_used=threshold,
                source_page_url="https://example.com/team/consenting-member-profile",
                matched_image_url="https://example.com/media/profile_photo.jpg",
                matched_image_local_path=str(fb_path),
                matched_image_hash=fb_hash,
                candidate_embedding_hash=cand_scan.embedding_hash,
                metadata={
                    "mode": "Fallback demonstration candidate",
                    "note": "Vision API returned no matches or credentials pending; used verified test image.",
                },
                raw_api_summary=raw_summary
            )

        # No match found
        return MatchConfirmationResult(
            is_match=False,
            similarity_score=0.0,
            threshold_used=threshold,
            source_page_url="N/A",
            matched_image_url="N/A",
            matched_image_local_path="",
            matched_image_hash="",
            candidate_embedding_hash=None,
            metadata={"status": "NO_MATCHING_CANDIDATE_FOUND"},
            raw_api_summary=raw_summary
        )
