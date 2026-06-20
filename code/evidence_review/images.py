"""Image resolution, downscaling, and image-ID handling.

Resizing the long edge before upload keeps damage detail while capping the
per-image token cost (Gemini tiles images into 768px tiles).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageOps

from . import config


@dataclass
class LoadedImage:
    image_id: str          # filename without extension, e.g. img_1
    rel_path: str          # path as written in the CSV
    data: bytes            # processed JPEG bytes
    mime_type: str
    width: int             # processed dimensions
    height: int
    ok: bool               # False if the file was missing or unreadable
    error: str = ""

    def estimated_tiles(self) -> int:
        """Approximate Gemini image-tile count for cost estimation."""
        if not self.ok:
            return 0
        if self.width <= 384 and self.height <= 384:
            return 1
        import math

        return max(1, math.ceil(self.width / 768) * math.ceil(self.height / 768))


def image_id_from_path(rel_path: str) -> str:
    return Path(rel_path).stem


def resolve(rel_path: str) -> Path:
    return (config.IMAGE_BASE / rel_path).resolve()


def _encode(im: Image.Image) -> tuple[bytes, int, int]:
    """Honor EXIF orientation, convert to RGB, downscale, and JPEG-encode."""
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    long_edge = max(im.size)
    if long_edge > config.MAX_IMAGE_LONG_EDGE:
        scale = config.MAX_IMAGE_LONG_EDGE / long_edge
        new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        im = im.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=config.JPEG_QUALITY, optimize=True)
    return buf.getvalue(), im.width, im.height


def load_and_process(rel_path: str) -> LoadedImage:
    image_id = image_id_from_path(rel_path)
    abs_path = resolve(rel_path)
    if not abs_path.exists():
        return LoadedImage(image_id, rel_path, b"", "image/jpeg", 0, 0, False, "file not found")
    try:
        with Image.open(abs_path) as im:
            data, w, h = _encode(im)
            return LoadedImage(image_id, rel_path, data, "image/jpeg", w, h, True)
    except Exception as e:  # noqa: BLE001 - any decode error => unusable image
        return LoadedImage(image_id, rel_path, b"", "image/jpeg", 0, 0, False, str(e))


def process_bytes(data: bytes, image_id: str, rel_path: str = "") -> LoadedImage:
    """Process raw uploaded image bytes (no file on disk) into a LoadedImage."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            jpg, w, h = _encode(im)
            return LoadedImage(image_id, rel_path or image_id, jpg, "image/jpeg", w, h, True)
    except Exception as e:  # noqa: BLE001
        return LoadedImage(image_id, rel_path or image_id, b"", "image/jpeg", 0, 0, False, str(e))


def load_claim_images(image_paths: List[str]) -> List[LoadedImage]:
    return [load_and_process(p) for p in image_paths]
