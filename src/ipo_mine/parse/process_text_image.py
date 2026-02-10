# parser/process_text_image.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
import base64
import re
import json
import os

# Optional deps guarded at import-time so the class still loads without them
try:
    from bs4 import BeautifulSoup, Tag  # type: ignore
except Exception:
    BeautifulSoup = None
    Tag = None

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:
    cv2 = None
    np = None

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


class LetterExtractor:
    """
    A small, focused helper for extracting CEO-style letters from S-1 filings.

    Main entry points:
      - extract_letter_from_html(html: str) -> str
      - ocr_images_tesseract(images: Iterable[str|Path]) -> str
      - ocr_images_openai(images: Iterable[str|Path]) -> str

    The class centralizes normalization heuristics, cue scoring, and boundaries.
    Optional dependencies are used only when you call those methods.
    """

    # --- Heuristics / cues -------------------------------------------------
    NEGATIVE_CUES = [
        "risk factors", "use of proceeds", "management’s discussion",
        "management's discussion", "md&a", "table of contents", "prospectus summary",
        "index", "financial statements", "selected financial data", "auditor’s report",
    ]
    LETTER_CUES = [
        r"\bdear\b",
        r"\bsincerely\b",
        r"\bchief executive officer\b",
        r"\bceo\b",
        r"\bto our shareholders\b",
        r"\bmessage to shareholders\b",
        r"\bletter\b",
    ]
    APOSTROPHE_FIX = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "\u00a0": " "})
    MAX_CHARS = 50_000

    def __init__(
        self,
        *,
        min_chars: int = 250,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            min_chars: Minimum length for HTML-extracted text before we call it a letter.
            openai_model: Default model used by ocr_images_openai.
            openai_api_key: If None, reads from OPENAI_API_KEY env var when using OpenAI OCR.
        """
        self.min_chars = min_chars
        self.openai_model = openai_model
        self.openai_api_key = openai_api_key

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def extract_letter_from_html(self, html: str) -> str:
        """
        Given targeted HTML (whole page or a narrowed snippet), return the cleaned letter text.

        Steps:
          1) Normalize punctuation/nbsp.
          2) Collect visible-ish text (skipping <script>/<style>).
          3) Trim after strong negative boundary cues (Risk Factors, TOC, etc.).
          4) Check plausibility: cue hits + length threshold.
        """
        if not html:
            return ""
        if BeautifulSoup is None:
            return self._clean_text(html)

        soup = BeautifulSoup(html, "lxml")
        raw = self._html_visible_text(soup)
        if not raw:
            return ""

        # Keep only the part before the first clear boundary token
        parts = re.split(
            r"(\bRisk Factors\b|\bTable of Contents\b|\bProspectus Summary\b)",
            raw,
            flags=re.IGNORECASE,
        )
        if parts:
            raw = parts[0] if parts[0] else raw

        txt = self._clean_text(raw)[: self.MAX_CHARS]
        if len(txt) < self.min_chars and self._plausibility_score(txt) < 1.5:
            # Too short and not convincingly a letter; still return for caller to decide
            return txt
        return txt

    def ocr_images_tesseract(self, images: Iterable[Path | str]) -> str:
        """
        Local OCR using Tesseract (no API). Best-effort rotations + PSM variants.
        Returns a single concatenated text blob.
        """
        if pytesseract is None or cv2 is None:
            return ""

        texts: List[str] = []
        for p in self._iter_files(images):
            img = cv2.imread(str(p))
            if img is None:
                continue
            proc = self._preprocess_for_ocr(img)
            best, best_len = "", 0
            for rot in (0, 90, 270):
                rot_img = proc if rot == 0 else cv2.rotate(
                    proc,
                    {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot],
                )
                for cfg in ("--oem 3 --psm 6", "--oem 3 --psm 4"):
                    try:
                        txt = pytesseract.image_to_string(rot_img, lang="eng", config=cfg, timeout=12)
                    except RuntimeError:
                        continue
                    txt = self._clean_text(txt)
                    if len(txt) > best_len:
                        best, best_len = txt, len(txt)
            if best:
                texts.append(best)
        return self._clean_text("\n\n".join(t for t in texts if t))

    def ocr_images_openai(
        self,
        images: Iterable[Path | str],
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """
        OCR using OpenAI Vision. Requires `openai` package and an API key.

        Returns concatenated text from all images where OCR succeeded.
        """
        if OpenAI is None:
            return ""
        key = api_key or self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return ""
        client = OpenAI(api_key=key)

        chosen_model = model or self.openai_model
        texts: List[str] = []

        for p in self._iter_files(images):
            b64 = self._img_to_b64(p)
            if not b64:
                continue
            try:
                msg = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all readable text from this image. Return raw text only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}  # png header is fine
                    ]
                }]
                resp = client.chat.completions.create(model=chosen_model, messages=msg, temperature=0)
                text = (resp.choices[0].message.content or "")
                text = self._clean_text(text)
                if text:
                    texts.append(text)
            except Exception:
                continue

        return self._clean_text("\n\n".join(t for t in texts if t))

    # ---------------------------------------------------------------------
    # Internals / helpers
    # ---------------------------------------------------------------------

    def _normalize_punct(self, s: str) -> str:
        return s.translate(self.APOSTROPHE_FIX)

    def _clean_text(self, s: str) -> str:
        s = self._normalize_punct(s or "")
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\.{3,}", "...", s)
        return s.strip()

    def _looks_like_negative_boundary(self, txt: str) -> bool:
        low = self._normalize_punct(txt).lower()
        return any(k in low for k in self.NEGATIVE_CUES)

    def _plausibility_score(self, txt: str) -> float:
        """Tiny heuristic: cue hits + a length bonus – a boundary penalty."""
        low = self._normalize_punct(txt).lower()
        hits = sum(bool(re.search(c, low)) for c in self.LETTER_CUES)
        length_bonus = min(len(low) / 2000.0, 1.0)  # up to +1 for longer letters
        penalty = 1.0 if (low.count("risk factors") > 2 or self._looks_like_negative_boundary(low)) else 0.0
        return max(0.0, hits + length_bonus - penalty)

    def _html_visible_text(self, soup) -> str:
        """Collect visible-ish text (skips <script>/<style>)."""
        out: List[str] = []
        for node in soup.find_all(text=True):
            if node.parent and node.parent.name in {"script", "style"}:
                continue
            t = str(node).strip()
            if t:
                out.append(t)
        return self._clean_text(" ".join(out))

    def _preprocess_for_ocr(self, img):
        """Light denoise + Otsu threshold for more stable OCR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thr

    def _iter_files(self, items: Iterable[Path | str]) -> List[Path]:
        """Resolve items to unique, existing files (globs allowed)."""
        found: List[Path] = []
        for it in items or []:
            p = Path(it)
            if p.is_file():
                found.append(p)
            else:
                # allow simple local globs like "pages/*.png"
                for g in sorted(Path(".").glob(str(it))):
                    if g.is_file():
                        found.append(g)
        uniq: List[Path] = []
        seen = set()
        for p in found:
            key = str(p.resolve())
            if key not in seen:
                uniq.append(p)
                seen.add(key)
        return uniq

    def _img_to_b64(self, p: Path) -> Optional[str]:
        try:
            return base64.b64encode(Path(p).read_bytes()).decode("ascii")
        except Exception:
            return None


# -----------------------------
# (Optional) tiny CLI wrapper
# -----------------------------
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="CEO Letter Extractor (HTML / Tesseract / OpenAI Vision)")
    ap.add_argument("--mode", choices=["html", "tesseract", "openai"], required=True)
    ap.add_argument("--html-file")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--openai-model", default="gpt-4o-mini")
    ap.add_argument("--openai-api-key", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ext = LetterExtractor(min_chars=args.min_chars, openai_model=args.openai_model, openai_api_key=args.openai_api_key)

    result = {"ok": False, "mode": args.mode, "text": ""}

    if args.mode == "html":
        html = Path(args.html_file).read_text(encoding="utf-8", errors="ignore") if args.html_file else ""
        text = ext.extract_letter_from_html(html)
        result.update(ok=bool(text), text=text)

    elif args.mode == "tesseract":
        text = ext.ocr_images_tesseract(args.images)
        result.update(ok=bool(text), text=text, images=[str(p) for p in args.images])

    elif args.mode == "openai":
        text = ext.ocr_images_openai(args.images)
        result.update(ok=bool(text), text=text, images=[str(p) for p in args.images], model=args.openai_model)

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
