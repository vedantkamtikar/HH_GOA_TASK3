import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Suppress PyTorch weight-loading warning noise
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from facenet_pytorch import MTCNN, InceptionResnetV1


@dataclass
class FaceScanResult:
    """Output structure of Stage 1: Face Identification & Embedding."""
    image_path: str
    face_detected: bool
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    embedding: Optional[List[float]] = None                   # 512-d normalized vector
    embedding_hash: Optional[str] = None                      # SHA-256 of the vector
    detection_confidence: Optional[float] = None
    crop_path: Optional[str] = None
    notes: str = ""

    def to_dict(self):
        return {
            "image_path": self.image_path,
            "face_detected": self.face_detected,
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
            "embedding_dimension": len(self.embedding) if self.embedding else 0,
            "embedding_hash": self.embedding_hash,
            "detection_confidence": round(self.detection_confidence, 4) if self.detection_confidence else None,
            "crop_path": self.crop_path,
            "notes": self.notes,
        }


class FaceIdentifier:
    """
    Stage 1 Engine: Detects faces, crops and aligns them, computes 512-d normalized
    embeddings using InceptionResnetV1, and produces a cryptographic digest.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # MTCNN face detector
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            device=self.device,
            post_process=True
        )
        
        # 512-d embedding model (InceptionResnetV1 trained on VGGFace2)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @staticmethod
    def calculate_embedding_hash(embedding: Union[List[float], np.ndarray]) -> str:
        """Deterministically compute SHA-256 digest of normalized 32-bit float array."""
        arr = np.array(embedding, dtype=np.float32)
        # Normalize to unit length for invariant representation
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @staticmethod
    def compute_cosine_similarity(vec_a: Union[List[float], np.ndarray], vec_b: Union[List[float], np.ndarray]) -> float:
        """Calculate cosine similarity between two face embedding vectors."""
        a = np.array(vec_a, dtype=np.float32).flatten()
        b = np.array(vec_b, dtype=np.float32).flatten()
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to [-1.0, 1.0] to guard against floating point precision drift
        return max(-1.0, min(1.0, sim))

    def _load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[Image.Image, str]:
        """Convert various input formats to PIL Image and source path string."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img = Image.open(path_str).convert("RGB")
            return img, path_str
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB"), "memory_buffer.jpg"
        elif isinstance(image_input, np.ndarray):
            # Assumed BGR if OpenCV array
            rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if len(image_input.shape) == 3 else image_input
            return Image.fromarray(rgb), "numpy_array.jpg"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def scan_face(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        save_crop_dir: Optional[Path] = None
    ) -> FaceScanResult:
        """
        Detect face, extract aligned crop, compute 512-d embedding vector,
        and generate cryptographic SHA-256 fingerprint.
        """
        img, source_path = self._load_image(image_input)
        
        # 1. Primary detection via MTCNN
        boxes, probs = self.mtcnn.detect(img)

        box = None
        prob = 0.0
        face_tensor = None

        if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.50:
            box = tuple(map(int, boxes[0]))
            prob = float(probs[0])
            # MTCNN extract aligned face tensor
            face_tensor = self.mtcnn(img)

        if face_tensor is None:
            return FaceScanResult(
                image_path=source_path,
                face_detected=False,
                notes="No face detected in image."
            )

        # 2. Extract 512-d embedding
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        
        face_tensor = face_tensor.to(self.device)
        with torch.no_grad():
            raw_embedding = self.resnet(face_tensor).squeeze().cpu().numpy()

        # Normalize embedding vector
        norm = np.linalg.norm(raw_embedding)
        norm_embedding = (raw_embedding / norm) if norm > 0 else raw_embedding
        embedding_list = [float(x) for x in norm_embedding]

        # 3. Compute SHA-256 cryptographic digest of embedding
        emb_hash = self.calculate_embedding_hash(norm_embedding)

        # 4. Optionally save visual crop
        crop_path_str = None
        if save_crop_dir and box is not None:
            save_crop_dir = Path(save_crop_dir)
            save_crop_dir.mkdir(parents=True, exist_ok=True)
            crop_file = save_crop_dir / f"crop_{emb_hash[:12]}.jpg"
            cropped_img = img.crop(box)
            cropped_img.save(crop_file, "JPEG")
            crop_path_str = str(crop_file)

        return FaceScanResult(
            image_path=source_path,
            face_detected=True,
            bounding_box=box,
            embedding=embedding_list,
            embedding_hash=emb_hash,
            detection_confidence=prob,
            crop_path=crop_path_str,
            notes=f"Detected with confidence {prob:.2f}; 512-d embedding computed."
        )
