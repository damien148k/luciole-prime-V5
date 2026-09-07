"""
OCR Module — Extraction de texte depuis images/PDFs scannés
Utilise EasyOCR avec support GPU CUDA
Mode OFFLINE : charge les modèles depuis le cache local
V3 : auto-device via resolve_device()
"""

import os
from typing import List, Optional
from pathlib import Path
import io
from loguru import logger

from ..utils.device import resolve_device

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    pymupdf = None

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class OCRProcessor:
    """
    Processeur OCR utilisant EasyOCR avec support GPU.

    Fonctionnalités:
    - OCR sur images (PNG, JPG, etc.)
    - OCR sur PDFs scannés (extraction page par page)
    - Détection automatique si une page PDF nécessite OCR
    - Support multilingue (français + anglais par défaut)
    """

    def _get_easyocr_model_dir(self) -> Optional[str]:
        """Cherche le dossier des modèles EasyOCR."""
        possible_paths = []

        easyocr_dir = os.environ.get("EASYOCR_MODEL_DIR")
        if easyocr_dir:
            possible_paths.append(Path(easyocr_dir))

        possible_paths.append(Path("/app/models/easyocr"))
        possible_paths.append(Path.home() / ".EasyOCR" / "model")

        for model_dir in possible_paths:
            if model_dir.exists():
                craft_model = model_dir / "craft_mlt_25k.pth"
                if craft_model.exists():
                    logger.info(f"Modèles EasyOCR trouvés dans: {model_dir}")
                    return str(model_dir)

        logger.warning(f"Modèles EasyOCR non trouvés dans: {possible_paths}")
        return None

    def __init__(
        self,
        languages: List[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.3
    ):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR n'est pas installé. Installez-le avec: pip install easyocr")

        self.languages = languages or ['fr', 'en']
        self.confidence_threshold = confidence_threshold

        resolved = resolve_device(device)
        gpu = resolved == "cuda"

        logger.info(f"Initialisation EasyOCR: langues={self.languages}, gpu={gpu}")

        model_dir = self._get_easyocr_model_dir()

        if model_dir:
            logger.info(f"Chargement EasyOCR en mode offline depuis: {model_dir}")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=gpu,
                verbose=False,
                model_storage_directory=model_dir,
                download_enabled=False
            )
        else:
            logger.warning("Mode online: EasyOCR va télécharger les modèles")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=gpu,
                verbose=False
            )

        logger.info("EasyOCR initialisé avec succès")

    def ocr_image(self, image_path: str) -> str:
        logger.debug(f"OCR sur image: {image_path}")
        results = self.reader.readtext(image_path)
        texts = []
        for (bbox, text, confidence) in results:
            if confidence >= self.confidence_threshold:
                texts.append(text)
        extracted_text = "\n".join(texts)
        logger.debug(f"OCR: {len(texts)} blocs de texte extraits")
        return extracted_text

    def ocr_image_bytes(self, image_bytes: bytes) -> str:
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow requis pour OCR sur bytes")
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        results = self.reader.readtext(image_array)
        texts = []
        for (bbox, text, confidence) in results:
            if confidence >= self.confidence_threshold:
                texts.append(text)
        return "\n".join(texts)

    def ocr_pdf(self, pdf_path: str, dpi: int = 200) -> str:
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF requis pour OCR sur PDF")
        logger.info(f"OCR sur PDF: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        all_text = []
        for page_num, page in enumerate(doc, 1):
            logger.debug(f"OCR page {page_num}/{len(doc)}")
            mat = pymupdf.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            page_text = self.ocr_image_bytes(img_bytes)
            if page_text.strip():
                all_text.append(f"--- Page {page_num} ---\n{page_text}")
        doc.close()
        result = "\n\n".join(all_text)
        logger.info(f"OCR PDF terminé: {len(doc)} pages traitées")
        return result

    def ocr_pdf_page(self, pdf_path: str, page_num: int, dpi: int = 200) -> str:
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF requis pour OCR sur PDF")
        doc = pymupdf.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            raise ValueError(f"Page {page_num} n'existe pas (PDF a {len(doc)} pages)")
        page = doc[page_num]
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return self.ocr_image_bytes(img_bytes)

    def needs_ocr(self, pdf_path: str, text_threshold: int = 50) -> List[int]:
        if not PYMUPDF_AVAILABLE:
            return []
        doc = pymupdf.open(pdf_path)
        pages_need_ocr = []
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) < text_threshold:
                pages_need_ocr.append(page_num)
        doc.close()
        if pages_need_ocr:
            logger.info(f"PDF {pdf_path}: {len(pages_need_ocr)} pages nécessitent OCR")
        return pages_need_ocr


def is_ocr_available() -> bool:
    """Vérifie si l'OCR est disponible."""
    return EASYOCR_AVAILABLE and PIL_AVAILABLE
