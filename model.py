from __future__ import annotations

import os
import logging
from typing import List, Dict, Optional
from uuid import uuid4

from PIL import Image, ImageOps

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import DATA_UNDEFINED_NAME

logger = logging.getLogger(__name__)


class NewModel(LabelStudioMLBase):
    """Surya OCR + detection integration for Label Studio

    Expects an OCR labeling config similar to Label Studio's OCR template:
      - an `Image` object
      - a `Polygon` object (for regions)
      - a `TextArea` control with `perRegion="true"` (for transcription)

    This backend detects text lines as polygons and fills transcription per region.
    """

    # Environment options
    DEVICE = os.getenv("DEVICE", "cuda")
    DISABLE_MATH = os.getenv("DISABLE_MATH", "false").lower() == "true"

    # These envs are used by get_local_path when LS serves files
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
        or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    _foundation = None
    _detector = None
    _recognizer = None

    def setup(self):
        # Set model version once
        self.set("model_version", f"surya-ocr-v0.0.1")

    # Lazy import + init to keep server start fast and optional
    def _lazy_init(self):
        if self._foundation is not None:
            return

        # Surya imports happen lazily to avoid heavy import time if server just needs to respond /health
        from surya.foundation import FoundationPredictor
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        logger.info("Initializing Surya predictors (foundation, detection, recognition)...")

        # FoundationPredictor decides device internally; DEVICE env is kept for future routing if needed
        self._foundation = FoundationPredictor()
        self._detector = DetectionPredictor()
        self._recognizer = RecognitionPredictor(self._foundation)

    @staticmethod
    def _image_size(path: str) -> tuple[int, int]:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.size

    @staticmethod
    def _to_ls_points(polygon_px: List[List[int]], img_w: int, img_h: int) -> List[List[float]]:
        # Convert pixel polygon [[x,y], ...] to Label Studio percentage points [[x%, y%], ...]
        pts = []
        for x, y in polygon_px:
            # Clamp to image bounds just in case
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            pts.append([x / img_w * 100.0, y / img_h * 100.0])
        return pts

    def _get_input_path(self, task: Dict, value_key: str) -> str:
        # Prefer explicit value key; fallback to $undefined$ if needed
        url = task["data"].get(value_key) or task["data"].get(DATA_UNDEFINED_NAME)
        if not url:
            raise FileNotFoundError("No input data URL found in task")
        return self.get_local_path(
            url,
            ls_host=self.LABEL_STUDIO_HOST,
            ls_access_token=self.LABEL_STUDIO_ACCESS_TOKEN,
            task_id=task.get("id"),
        )

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Run Surya OCR on incoming tasks and return polygon + transcription per region.

        Notes:
        - Only the first page is processed if a multi-page PDF is provided.
        - Requires labeling config to include `Polygon` and `TextArea` bound to the `Image`.
        """
        self._lazy_init()

        # Discover tag names based on project labeling config
        # We use Polygon for regions and TextArea(perRegion) for transcription
        from_name_poly, to_name, value_key = self.get_first_tag_occurence("Polygon", "Image")
        from_name_text, _, _ = self.get_first_tag_occurence("TextArea", "Image")

        if not (from_name_poly and to_name and value_key and from_name_text):
            logger.warning(
                "Label config missing required tags: need Polygon + TextArea bound to Image."
            )

        predictions: List[Dict] = []

        for task in tasks:
            try:
                input_path = self._get_input_path(task, value_key)
            except Exception as exc:
                logger.exception("Failed to resolve local path for task input: %s", exc)
                continue

            # Load image(s) using Surya helpers
            # Use same loader as Surya CLI to support PDFs and images
            from surya.input.load import load_from_file
            images, names = load_from_file(input_path)

            if not images:
                predictions.append({
                    "result": [],
                    "score": 0,
                    "model_version": str(self.model_version) if self.model_version else None,
                })
                continue

            # Only process first page for LS single-image task
            image = images[0]
            img_w, img_h = image.size

            # Run detection + recognition (with polygons)
            try:
                from surya.common.surya.schema import TaskNames
                task_names = [TaskNames.ocr_with_boxes]
                preds = self._recognizer(
                    [image],
                    task_names=task_names,
                    det_predictor=self._detector,
                    highres_images=None,
                    math_mode=not self.DISABLE_MATH,
                )
            except Exception as exc:
                logger.exception("Surya prediction failed: %s", exc)
                predictions.append({
                    "result": [],
                    "score": 0,
                    "model_version": str(self.model_version) if self.model_version else None,
                })
                continue

            page_pred = preds[0]
            results = []
            region_scores = []

            # Build LS results: one polygon region + one textarea per text line
            for line in page_pred.text_lines:
                # Use average of char confidences Surya provides at line level
                score = float(line.confidence or 0)
                region_scores.append(score)

                points = self._to_ls_points(line.polygon, img_w, img_h)
                region_id = str(uuid4())[:8]

                # Polygon region
                results.append(
                    {
                        "original_width": img_w,
                        "original_height": img_h,
                        "image_rotation": 0,
                        "value": {"points": points},
                        "id": region_id,
                        "from_name": from_name_poly,
                        "to_name": to_name,
                        "type": "polygon",
                        "origin": "manual",
                        "score": score,
                    }
                )

                # Transcription attached to the same region
                results.append(
                    {
                        "original_width": img_w,
                        "original_height": img_h,
                        "image_rotation": 0,
                        "value": {
                            "points": points,
                            # Some OCR templates include labels, but it's optional here
                            "labels": ["text"] if from_name_text == "transcription" else [],
                            "text": [line.text],
                        },
                        "id": region_id,
                        "from_name": from_name_text,
                        "to_name": to_name,
                        "type": "textarea",
                        "origin": "manual",
                        "score": score,
                    }
                )

            avg_score = sum(region_scores) / max(len(region_scores), 1)
            predictions.append(
                {
                    "result": results,
                    "score": avg_score,
                    "model_version": str(self.model_version) if self.model_version else None,
                }
            )

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        # This backend does not train; keep a simple example of using cache
        old_version = self.get("model_version")
        logger.debug("Received fit event=%s; current model_version=%s", event, old_version)
        # No-op training. If you want, bump version to reflect updates
        self.set("model_version", str(old_version))
        return {"status": "ok", "event": event}
