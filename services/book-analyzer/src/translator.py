import logging
import re
import threading
from typing import List, Optional

log = logging.getLogger(__name__)

CYRILLIC_RANGE = re.compile(r"[Ѐ-ӿ]")
LETTER_RANGE = re.compile(r"[^\W\d_]", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# opus-mt часто выдаёт "Chapter one, ..." или "Part two, ..."
# Детектор глав ждёт точку, двоеточие или тире после номера, поэтому нормализуем заголовки
_HEADING_COMMA_RE = re.compile(
    r"(?im)^(\s*(?:chapter|part|book|section|act|scene)\s+\S+?)\s*,\s+",
)


def normalize_chapter_headings(text: str) -> str:
    return _HEADING_COMMA_RE.sub(lambda m: m.group(1) + ". ", text)


def detect_language(text: str, sample_chars: int = 2000) -> str:
    """Return 'ru' if a meaningful share of the sample is Cyrillic, else 'en'."""
    sample = text[:sample_chars]
    if not sample:
        return "en"
    cyrillic = len(CYRILLIC_RANGE.findall(sample))
    letters = len(LETTER_RANGE.findall(sample))
    if letters == 0:
        return "en"
    return "ru" if cyrillic / letters > 0.3 else "en"


class Translator:
    """Lazy-loaded ru→en translator using Helsinki-NLP/opus-mt-tc-big-zle-en.

    The "tc-big" variant covers all East Slavic languages (ru/uk/be) and is
    ~230M params (~1.2GB on disk), trained on Tatoeba Challenge data. It
    produces dramatically better literary translations than the small
    opus-mt-ru-en (~75M) — fewer hallucinations and a richer English
    vocabulary. Trade-offs: ~1GB extra image size and ~50% slower inference.

    Beam search and repetition penalties below are essential: with greedy
    decoding (num_beams=1) and no penalties, MarianMT loops on long inputs
    and produces strings like "and the foot to the foot to the foot...".
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-tc-big-zle-en",
        batch_size: int = 8,
        num_beams: int = 4,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("loading translator model: %s", self.model_name)
        from transformers import MarianMTModel, MarianTokenizer

        self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self._model = MarianMTModel.from_pretrained(self.model_name)
        self._model.eval()
        log.info("translator loaded")

    def translate(self, text: str, cancel: Optional[threading.Event] = None) -> str:
        """Translate Russian text to English. If ``cancel`` is provided and gets
        set during execution, translation aborts at the next batch boundary by
        raising ``RuntimeError``."""
        self._ensure_loaded()
        paragraphs = text.split("\n\n")

        flat: List[tuple[int, int, str]] = []
        for para_idx, para in enumerate(paragraphs):
            stripped = para.strip()
            if not stripped:
                continue
            for sub_idx, chunk in enumerate(self._chunk(stripped)):
                flat.append((para_idx, sub_idx, chunk))

        translations: dict[int, list[tuple[int, str]]] = {}
        import time

        import torch

        total_batches = (len(flat) + self.batch_size - 1) // self.batch_size
        log.info(
            "translating %d chunks in %d batches (batch=%d, num_beams=%d)",
            len(flat), total_batches, self.batch_size, self.num_beams,
        )
        start_time = time.monotonic()

        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, len(flat), self.batch_size), start=1):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError(
                        f"translation cancelled after {batch_idx - 1}/{total_batches} batches"
                    )
                batch = flat[start : start + self.batch_size]
                texts = [b[2] for b in batch]
                inputs = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                outputs = self._model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=self.num_beams,
                    repetition_penalty=self.repetition_penalty,
                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                    early_stopping=True,
                )
                decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for (para_idx, sub_idx, _), out in zip(batch, decoded):
                    translations.setdefault(para_idx, []).append((sub_idx, out))
                if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
                    elapsed = time.monotonic() - start_time
                    rate = batch_idx / elapsed if elapsed > 0 else 0
                    eta = (total_batches - batch_idx) / rate if rate > 0 else 0
                    log.info(
                        "translation progress: %d/%d batches (%.1fs elapsed, eta %.0fs)",
                        batch_idx, total_batches, elapsed, eta,
                    )

        out_paragraphs: List[str] = []
        for para_idx, para in enumerate(paragraphs):
            if para_idx in translations:
                parts = sorted(translations[para_idx], key=lambda x: x[0])
                out_paragraphs.append(" ".join(p[1].strip() for p in parts))
            else:
                out_paragraphs.append(para)
        return normalize_chapter_headings("\n\n".join(out_paragraphs))

    def _chunk(self, paragraph: str, max_chars: int = 1500) -> List[str]:
        """Split a paragraph into chunks that comfortably fit in 512 tokens.

        We target characters because tokenization is expensive; ~1500 Russian
        characters typically tokenize to under 400 tokens for opus-mt.
        """
        if len(paragraph) <= max_chars:
            return [paragraph]
        sentences = SENTENCE_SPLIT.split(paragraph)
        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip() if current else sentence
        if current:
            chunks.append(current.strip())
        return chunks
