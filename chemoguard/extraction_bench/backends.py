"""Adapters over the extraction backends, plus one that skips OCR entirely.

Three arms, and the third is the reason the other two can be interpreted:

``pattern``
    The production regex set in ``ocr_extraction.extract_parameters``, run
    over the exact text the generator drew. No OCR in the loop. Whatever it
    fails to read is a defect in the patterns or the alias table, because the
    glyphs were handed over perfectly. This arm needs no binaries, runs in
    seconds, and gives the other arms their ceiling: a field the pattern
    layer cannot read from perfect text is one Tesseract was never going to
    get either, and charging that loss to OCR would point a fix at the wrong
    module.

``tesseract``
    The full production path -- preprocess, OCR, regex, canonicalise. Needs
    the Tesseract binary. ``available()`` reports whether it is installed
    rather than failing at import, because it is not installed everywhere
    this runs and a benchmark that cannot start is worse than one that
    reports a missing arm.

``gpt4o_mini``
    Complete and **hard-guarded against making a network call**. Deferred by
    decision, not by omission: the adapter, the response cache and the
    parsing all exist, so enabling the arm is a flag rather than a rewrite.
    ``read`` raises unless network access is enabled explicitly at
    construction *and* ``CHEMOGUARD_BENCH_ALLOW_OPENAI=1`` is set in the
    environment. Two independent switches, because an accidental run spends
    real money against a real key.

Every arm returns the same ``Reading``, and every arm reaches its panel
through ``cbc.panel_from_extraction`` -- the same function the application
uses. The benchmark measures the production boundary, not a reimplementation
of it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

PATTERN = "pattern"
TESSERACT = "tesseract"
#: The same OCR path with the page-segmentation mode the code shipped with.
#: Scored alongside, not instead: see CLAUDE.md Finding 9 / M4.
TESSERACT_AS_SHIPPED = "tesseract_psm6"
GPT4O_MINI = "gpt4o_mini"


class NetworkDisabledError(RuntimeError):
    """The GPT arm was called while network access was switched off."""


@dataclass
class Reading:
    """One backend's attempt at one image.

    ``raw`` is the value-and-unit pair per analyte as the backend read it,
    before canonicalisation -- the only place the *printed* unit string
    survives, and therefore the only place a transcription error can be told
    apart from an alias-table gap.

    ``panel`` and ``skipped`` are what ``cbc.panel_from_extraction`` made of
    ``raw``: the canonicalised quantities that got through, and the readings
    that were refused with the reason why.
    """

    backend: str
    image_id: str
    raw: Dict[str, Optional[dict]] = field(default_factory=dict)
    panel: object = None
    skipped: Dict[str, dict] = field(default_factory=dict)
    ocr_text: Optional[str] = None
    confidence: Optional[float] = None
    #: Set when the backend raised. A failed read is recorded, not dropped:
    #: an image that crashed the extractor is a result.
    error: Optional[str] = None


def _empty_panel():
    from cbc import panel_from_extraction

    panel, _ = panel_from_extraction({})
    return panel


class PatternLayerBackend:
    """The production regex set, over the text the generator drew."""

    name = PATTERN

    @staticmethod
    def available() -> bool:
        return True

    def read(self, image, corpus_root) -> Reading:
        from cbc import panel_from_extraction
        from ocr_extraction import HematologyExtractor

        if not image.text_path:
            return Reading(
                backend=self.name,
                image_id=image.image_id,
                panel=_empty_panel(),
                error="no page text recorded for this image",
            )

        text = (Path(corpus_root) / image.text_path).read_text(encoding="utf-8")
        raw = HematologyExtractor().extract_parameters(text)
        panel, skipped = panel_from_extraction(raw)

        return Reading(
            backend=self.name,
            image_id=image.image_id,
            raw=raw,
            panel=panel,
            skipped=skipped,
            ocr_text=text,
        )


class TesseractBackend:
    """The full production OCR path, retaining the intermediate reading.

    Calls the same three steps ``HematologyExtractor.process_file`` calls, in
    the same order, rather than calling it directly -- only so that ``raw``
    survives for attribution. If ``process_file`` changes, this has to change
    with it; ``tests/test_extraction_bench.py`` pins the two together.

    ``psm`` overrides the page-segmentation mode so the as-shipped mode can
    be scored beside the corrected one. That comparison is the whole evidence
    for the change; asserting a 19x recall difference without showing it
    would be the kind of claim this benchmark exists to stop.
    """

    name = TESSERACT

    def __init__(self, tesseract_cmd: Optional[str] = None, psm: Optional[int] = None,
                 name: Optional[str] = None):
        self.tesseract_cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")
        self.psm = psm
        if name:
            self.name = name

    def available(self) -> bool:
        try:
            import pytesseract

            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def read(self, image, corpus_root) -> Reading:
        from cbc import panel_from_extraction
        from ocr_extraction import HematologyExtractor

        path = Path(corpus_root) / image.path
        extractor = HematologyExtractor(
            tesseract_cmd=self.tesseract_cmd, psm=self.psm
        )

        try:
            text, confidence = extractor.extract_text_from_image(str(path))
            raw = extractor.extract_parameters(text)
            panel, skipped = panel_from_extraction(raw)
        except Exception as exc:                      # noqa: BLE001
            return Reading(
                backend=self.name,
                image_id=image.image_id,
                panel=_empty_panel(),
                error="{0}: {1}".format(type(exc).__name__, exc),
            )

        return Reading(
            backend=self.name,
            image_id=image.image_id,
            raw=raw,
            panel=panel,
            skipped=skipped,
            ocr_text=text,
            confidence=confidence,
        )


class GPTBackend:
    """GPT-4o-mini vision, complete and switched off.

    The cache is keyed on the image bytes and the prompt text, so a rerun
    after the arm is enabled costs nothing and the reported numbers stay
    pinned to the exact responses they were computed from. That matters more
    here than for a deterministic backend: the call runs at temperature 0.1,
    which is not zero, so without a cache the numbers would not be
    reproducible even from the same corpus.
    """

    name = GPT4O_MINI

    #: Both must be true before a request is made.
    ENV_FLAG = "CHEMOGUARD_BENCH_ALLOW_OPENAI"

    def __init__(self, cache_dir, allow_network: bool = False):
        self.cache_dir = Path(cache_dir)
        self.allow_network = allow_network

    def available(self) -> bool:
        """Whether a run would be permitted. Deliberately not "is it installed"."""
        return self.allow_network and os.environ.get(self.ENV_FLAG) == "1"

    def _cache_path(self, image_bytes: bytes, prompt: str) -> Path:
        digest = hashlib.sha256(image_bytes)
        digest.update(b"\x00")
        digest.update(prompt.encode("utf-8"))
        return self.cache_dir / (digest.hexdigest()[:32] + ".json")

    def read(self, image, corpus_root) -> Reading:
        from cbc import panel_from_extraction
        from gpt_extraction import GPTExtractor

        path = Path(corpus_root) / image.path
        image_bytes = path.read_bytes()
        cached = self._cache_path(image_bytes, GPTExtractor.PROMPT)

        if cached.exists():
            raw = json.loads(cached.read_text(encoding="utf-8"))
        else:
            if not self.available():
                raise NetworkDisabledError(
                    "The GPT-4o-mini arm is deferred and no cached response "
                    "exists for {0}. Enable it deliberately: construct "
                    "GPTBackend(allow_network=True) and set {1}=1. Each "
                    "uncached image is a billable request against the key in "
                    ".env.".format(image.image_id, self.ENV_FLAG)
                )
            raw = GPTExtractor().extract_from_image(str(path))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(raw, indent=2), encoding="utf-8")

        panel, skipped = panel_from_extraction(raw)
        return Reading(
            backend=self.name,
            image_id=image.image_id,
            raw=raw,
            panel=panel,
            skipped=skipped,
        )


def build_backends(corpus_root, arms=None, allow_openai: bool = False):
    """The arms to run, with the deferred one left switched off by default."""
    available = {
        PATTERN: lambda: PatternLayerBackend(),
        TESSERACT: lambda: TesseractBackend(),
        TESSERACT_AS_SHIPPED: lambda: TesseractBackend(
            psm=6, name=TESSERACT_AS_SHIPPED
        ),
        GPT4O_MINI: lambda: GPTBackend(
            Path(corpus_root) / "gpt_cache", allow_network=allow_openai
        ),
    }
    chosen = arms or (PATTERN, TESSERACT)
    return [available[name]() for name in chosen]
