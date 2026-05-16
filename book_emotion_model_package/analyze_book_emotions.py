#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
GENERATED_OUTPUT_FILES = (
    "cleaned_main_text.txt",
    "chapter_detection_report.json",
    "chapter_emotions.json",
    "chapter_emotions.csv",
    "chapter_emotions_summary.txt",
    "chapter_emotions_plot.png",
    "block_emotions.json",
    "block_emotions.csv",
    "block_emotions_plot.png",
    "cleaned_text.txt",
    "boundary_detection.json",
    "chapters.json",
    "scenes.json",
    "raw_scene_emotions.json",
    "llm_scene_contexts.json",
    "normalized_scene_emotions.json",
    "chapter_emotion_profiles.json",
    "book_emotion_profile.json",
    "character_emotion_profiles.json",
    "calibration_report.json",
)
GENERATED_OUTPUT_DIRS = (
    "chapter_texts",
    "block_texts",
    "chapter_bar_charts",
)
STRUCTURED_HEADING_REGEX = re.compile(
    r"(?ix)^((?:chapter|section|part|book|act|scene)\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?:\b(?:\s*[:.-]\s*.*|\s+.*)?)?|(?:prologue|epilogue|interlude)(?:\b(?:\s*[:.-]\s*.*)?)?)$"
)
DASHED_CHAPTER_REGEX = re.compile(
    r"(?ix)^[\s\-–—]*chapter\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)[\s\-–—]*$"
)
SECTION_MARKER_REGEX = re.compile(
    r"(?ix)^(?:part|book|act)\s+(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\b(?:\s*[:.-]\s*.*|\s+.*)?)?$"
)
FRONT_MATTER_HEADING_REGEX = re.compile(
    r"(?i)^(foreword|preface|introduction|contents|table of contents|copyright|dedication|translator'?s note|note on the text|acknowledg(?:e)?ments?)\b"
)
BACK_MATTER_HEADING_REGEX = re.compile(
    r"(?i)^(afterword|appendix|appendices|notes|endnotes|bibliography|glossary|about the author|about this book|discussion questions|reading group guide|credits|index)\b"
)
DOMAIN_WATERMARK_REGEX = re.compile(r"(?i)\b[a-z0-9.-]+\.(com|net|org|ru|io|co)\b")
ROMAN_NUMERAL_REGEX = re.compile(r"(?i)^(?:[ivxlcdm]+)$")
ARABIC_NUMERAL_REGEX = re.compile(r"^\d{1,3}[.)]?$")
PAGE_COUNT_REGEX = re.compile(r"(?i)^(?:page\s+)?\d{1,4}\s+(?:of|/)\s+\d{1,4}$")
TITLE_ONLY_BAD_PREFIXES = (
    "copyright",
    "isbn",
    "library of congress",
    "printed in",
    "published by",
    "summary:",
)


@dataclass
class TextLine:
    text: str
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    page_width: float
    page_height: float
    global_index: int = -1

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class PageData:
    page_num: int
    width: float
    height: float
    lines: List[TextLine] = field(default_factory=list)

    def median_font(self) -> float:
        vals = [ln.font_size for ln in self.lines if ln.font_size > 0]
        return float(median(vals)) if vals else 12.0

    def median_gap(self) -> float:
        if len(self.lines) < 2:
            return 8.0
        gaps = [max(0.0, self.lines[i + 1].y0 - self.lines[i].y1) for i in range(len(self.lines) - 1)]
        gaps = [g for g in gaps if g >= 0]
        return float(median(gaps)) if gaps else 8.0


@dataclass
class HeadingCandidate:
    title: str
    kind: str  # chapter, front, back
    score: float
    page_num: int
    start_global: int
    end_global: int
    line_indices_on_page: Tuple[int, int]
    reasons: List[str] = field(default_factory=list)


@dataclass
class Chapter:
    index: int
    title: str
    text: str
    start_page: int
    end_page: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass
class ChunkPrediction:
    chunk_index: int
    text: str
    scores: Dict[str, float]


@dataclass
class BookText:
    raw_text: str
    cleaned_text: Optional[str]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Scene:
    chapter_index: int
    scene_index: int
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


@dataclass
class SceneContext:
    chapter_index: int
    scene_index: int
    summary: str = ""
    events: List[str] = field(default_factory=list)
    characters: List[str] = field(default_factory=list)
    speaker_emotions: Dict[str, List[str]] = field(default_factory=dict)
    character_emotions: Dict[str, List[str]] = field(default_factory=dict)
    narrator_emotions: List[str] = field(default_factory=list)
    narrator_tone: List[str] = field(default_factory=list)
    scene_tone: List[str] = field(default_factory=list)
    conflict: Optional[str] = None
    approval_interpretation: Dict[str, object] = field(default_factory=dict)
    dominant_perspective: str = "unknown"
    source: str = "none"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class EmotionScore:
    emotion: str
    score: float
    source: str
    explanation: Optional[str] = None


@dataclass
class SceneEmotionResult:
    chapter_index: int
    scene_index: int
    context: Optional[SceneContext]
    raw_classifier_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    scene_word_count: int = 0
    classifier_chunk_count: int = 0
    confidence: float = 0.0
    confidence_reasons: List[str] = field(default_factory=list)
    scene_weight: float = 1.0
    dominant_perspective: str = "unknown"
    character_emotion_scores: Dict[str, float] = field(default_factory=dict)
    narrator_emotion_scores: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


@dataclass
class ChapterEmotionProfile:
    chapter_index: int
    chapter_title: str
    scene_count: int
    average_scores: Dict[str, float]
    max_scores: Dict[str, float]
    dominant_emotions: List[Dict[str, float]]
    most_intense_scenes: List[Dict[str, object]]
    top_scenes_by_emotion: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    raw_average_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    use_llm_scene_analysis: bool = False
    use_context_normalization: bool = True
    save_intermediate_json: bool = True
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    llm_timeout_seconds: int = 30
    use_heuristic_context: bool = True
    scene_min_words: int = 250
    scene_target_words: int = 1000
    chapter_detection_fallback_block_count: int = 20
    exclude_neutral_from_profiles: bool = True
    use_corpus_calibration: bool = True
    context_window_scenes: int = 1
    classifier_peak_blend: float = 0.30


@dataclass
class AnalysisResult:
    book_title: Optional[str]
    chapters: List[Chapter]
    scenes: List[Scene]
    scene_results: List[SceneEmotionResult]
    chapter_profiles: List[ChapterEmotionProfile]
    diagnostics: Dict[str, object]
    output_files: Dict[str, str]


class BookEmotionAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)

        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_limit is None or tokenizer_limit > 100_000:
            tokenizer_limit = 512
        self.max_length = min(max_length or tokenizer_limit, tokenizer_limit)

        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in sorted(id2label.keys())]

    def predict_texts(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        outputs: List[Dict[str, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.sigmoid(logits).detach().cpu().tolist()
            for row in probs:
                outputs.append({label: float(score) for label, score in zip(self.labels, row)})
        return outputs


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def clean_line_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_counting(text: str) -> str:
    text = clean_line_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -—–_")


def is_page_number_line(text: str) -> bool:
    t = clean_line_text(text)
    if not t:
        return True
    return bool(
        re.fullmatch(r"(?:page\s+)?\d{1,4}", t, flags=re.I)
        or PAGE_COUNT_REGEX.fullmatch(t)
        or ROMAN_NUMERAL_REGEX.fullmatch(t)
    )


def is_numeric_only(text: str) -> bool:
    t = clean_line_text(text).strip(".)")
    return bool(ARABIC_NUMERAL_REGEX.fullmatch(t) or ROMAN_NUMERAL_REGEX.fullmatch(t))


def looks_like_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) >= 0.8


def looks_like_title_case(text: str) -> bool:
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return False
    good = 0
    checked = 0
    for word in words:
        if not re.search(r"[A-Za-z]", word):
            continue
        checked += 1
        stripped = re.sub(r"^[\W_]+|[\W_]+$", "", word)
        if stripped[:1].isupper():
            good += 1
    return checked > 0 and good / checked >= 0.7


def is_matter_heading(text: str, regex: re.Pattern[str]) -> bool:
    t = clean_line_text(text).rstrip(".")
    if not regex.match(t):
        return False
    words = t.split()
    if len(words) <= 4:
        return True
    return looks_like_all_caps(t) or looks_like_title_case(t)


def is_front_matter_heading(text: str) -> bool:
    return is_matter_heading(text, FRONT_MATTER_HEADING_REGEX)


def is_back_matter_heading(text: str) -> bool:
    return is_matter_heading(text, BACK_MATTER_HEADING_REGEX)


def is_section_marker_title(text: str) -> bool:
    return SECTION_MARKER_REGEX.match(clean_line_text(text)) is not None


def is_structured_heading_text(text: str) -> bool:
    t = clean_line_text(text)
    return STRUCTURED_HEADING_REGEX.match(t) is not None or DASHED_CHAPTER_REGEX.match(t) is not None


def should_prefix_with_section(title: str) -> bool:
    return bool(re.match(r"(?i)^chapter\b", clean_line_text(title)))


def prefixed_chapter_title(section_title: Optional[str], chapter_title: str) -> str:
    if section_title and should_prefix_with_section(chapter_title):
        return f"{section_title} - {chapter_title}"
    return chapter_title


def compile_optional_regex(pattern: Optional[str], label: str) -> Optional[re.Pattern[str]]:
    if not pattern:
        return None
    try:
        return re.compile(pattern, flags=re.I)
    except re.error as exc:
        raise ValueError(f"Invalid {label} regex: {exc}") from exc


def matches_optional_regex(text: str, regex: Optional[re.Pattern[str]]) -> bool:
    if regex is None:
        return True
    return regex.search(clean_line_text(text)) is not None


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def clean_generated_outputs(output_dir: Path) -> List[str]:
    removed: List[str] = []
    if not output_dir.exists():
        return removed

    for file_name in GENERATED_OUTPUT_FILES:
        path = output_dir / file_name
        if path.is_file():
            path.unlink()
            removed.append(str(path))

    for dir_name in GENERATED_OUTPUT_DIRS:
        path = output_dir / dir_name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))

    return removed


def looks_like_heading_line(text: str) -> bool:
    t = clean_line_text(text)
    if not t:
        return False
    if len(t) > 120:
        return False
    if t.lower().startswith(TITLE_ONLY_BAD_PREFIXES):
        return False
    words = t.split()
    if len(words) > 16:
        return False
    if t.endswith((".", "?", "!", ";")) and not STRUCTURED_HEADING_REGEX.match(t) and not is_numeric_only(t):
        trimmed = t.rstrip(".?!;")
        if len(words) > 8 and not looks_like_all_caps(trimmed):
            return False
    alpha_chars = sum(c.isalpha() for c in t)
    if alpha_chars == 0 and not is_numeric_only(t):
        return False
    return (
        is_structured_heading_text(t)
        or is_front_matter_heading(t)
        or is_back_matter_heading(t)
        or is_numeric_only(t)
        or looks_like_all_caps(t.rstrip(".?!;"))
        or looks_like_title_case(t.rstrip(".?!;"))
    )


def extract_pdf_pages(path: Path) -> List[PageData]:
    pages: List[PageData] = []
    with fitz.open(path) as doc:
        global_index = 0
        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict", sort=True)
            page_data = PageData(page_num=page_num, width=float(page.rect.width), height=float(page.rect.height))
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = clean_line_text("".join(span.get("text", "") for span in spans))
                    if not text:
                        continue
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    font_size = max(float(span.get("size", 12.0)) for span in spans)
                    line_obj = TextLine(
                        text=text,
                        page_num=page_num,
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                        font_size=font_size,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                        global_index=global_index,
                    )
                    page_data.lines.append(line_obj)
                    global_index += 1
            pages.append(page_data)
    return pages


def extract_txt_pages(path: Path) -> List[PageData]:
    raw = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raw = path.read_text(errors="replace")

    raw = normalize_whitespace(raw)
    lines = raw.split("\n")
    page = PageData(page_num=1, width=600, height=max(800, len(lines) * 14 + 40))
    y = 30.0
    global_index = 0
    for idx, raw_line in enumerate(lines):
        text = clean_line_text(raw_line)
        if text:
            page.lines.append(
                TextLine(
                    text=text,
                    page_num=1,
                    x0=50.0,
                    y0=y,
                    x1=550.0,
                    y1=y + 12.0,
                    font_size=12.0,
                    page_width=600.0,
                    page_height=page.height,
                    global_index=global_index,
                )
            )
            global_index += 1
        y += 18.0
    return [page]


def remove_repeated_headers_footers(pages: List[PageData]) -> Tuple[List[PageData], List[str]]:
    if not pages:
        return pages, []

    line_pages: Dict[str, set] = defaultdict(set)
    for page in pages:
        for line in page.lines:
            norm = normalize_for_counting(line.text)
            if not norm:
                continue
            near_edge = line.y0 < page.height * 0.15 or line.y1 > page.height * 0.85
            if near_edge or DOMAIN_WATERMARK_REGEX.search(norm):
                line_pages[norm].add(page.page_num)

    repeated: set[str] = set()
    threshold = max(3, math.ceil(len(pages) * 0.05))
    for norm, page_nums in line_pages.items():
        if len(page_nums) >= threshold and len(norm) <= 80:
            repeated.add(norm)
        if DOMAIN_WATERMARK_REGEX.search(norm):
            repeated.add(norm)

    removed_samples: Counter[str] = Counter()
    new_pages: List[PageData] = []
    for page in pages:
        kept_lines: List[TextLine] = []
        for idx, line in enumerate(page.lines):
            norm = normalize_for_counting(line.text)
            near_edge = line.y0 < page.height * 0.15 or line.y1 > page.height * 0.85
            if norm in repeated:
                removed_samples[line.text] += 1
                continue
            if near_edge and is_page_number_line(line.text):
                next_line = page.lines[idx + 1] if idx + 1 < len(page.lines) else None
                looks_like_section_number = (
                    line.y0 < page.height * 0.25
                    and next_line is not None
                    and looks_like_heading_line(next_line.text)
                    and len(clean_line_text(line.text)) <= 4
                    and (is_centered(line) or line.font_size >= page.median_font() * 1.1)
                )
                looks_like_chapter_number = looks_like_top_chapter_number(line, page, next_line)
                if not looks_like_section_number and not looks_like_chapter_number:
                    removed_samples[line.text] += 1
                    continue
            kept_lines.append(line)
        new_pages.append(PageData(page_num=page.page_num, width=page.width, height=page.height, lines=kept_lines))

    gi = 0
    for page in new_pages:
        for line in page.lines:
            line.global_index = gi
            gi += 1

    removed_preview = [f"{text} (x{count})" for text, count in removed_samples.most_common(10)]
    return new_pages, removed_preview


def gap_before(page: PageData, idx: int) -> float:
    if idx <= 0:
        return page.lines[idx].y0
    return max(0.0, page.lines[idx].y0 - page.lines[idx - 1].y1)


def gap_after(page: PageData, idx: int) -> float:
    if idx >= len(page.lines) - 1:
        return page.height - page.lines[idx].y1
    return max(0.0, page.lines[idx + 1].y0 - page.lines[idx].y1)


def is_centered(line: TextLine) -> bool:
    return abs(line.center_x - line.page_width / 2.0) <= line.page_width * 0.14


def body_like_line(text: str) -> bool:
    t = clean_line_text(text)
    if len(t) >= 80:
        return True
    words = t.split()
    return len(words) >= 14 and t.endswith((".", ",", ";", ":", "?", "!", '"'))


def looks_like_heading_subtitle(line: TextLine, page: PageData) -> bool:
    text = clean_line_text(line.text)
    if not text or body_like_line(text) or is_page_number_line(text):
        return False
    if is_structured_heading_text(text) or is_front_matter_heading(text) or is_back_matter_heading(text):
        return False
    if len(text.split()) > 12 or len(text) > 90:
        return False
    display_width = line.width <= page.width * 0.78
    large_display = line.font_size >= page.median_font() * 1.15
    compact_centered_title = is_centered(line) and display_width and looks_like_title_case(text)
    return large_display or compact_centered_title


def looks_like_top_chapter_number(line: TextLine, page: PageData, next_line: Optional[TextLine]) -> bool:
    if next_line is None:
        return False
    if not is_numeric_only(line.text):
        return False
    if line.y0 > page.height * 0.12:
        return False
    if not is_centered(line):
        return False
    if line.font_size < page.median_font() * 1.1:
        return False
    next_text = clean_line_text(next_line.text)
    return (
        looks_like_heading_line(next_text)
        or body_like_line(next_text)
        or (len(next_text.split()) >= 6 and not is_page_number_line(next_text))
    )


def assign_candidate_kind(title: str) -> str:
    t = clean_line_text(title)
    if is_front_matter_heading(t):
        return "front"
    if is_back_matter_heading(t):
        return "back"
    return "chapter"


def is_major_chapter_candidate(candidate: HeadingCandidate) -> bool:
    if candidate.kind != "chapter":
        return False
    reasons = set(candidate.reasons)
    title_norm = normalize_for_counting(candidate.title)
    words = len(candidate.title.split())
    if "structured-heading" in reasons or "numeric-heading" in reasons:
        return True
    if title_norm.startswith(("by ", "from ", "to ", "with ", "concerning ", "comments ")):
        return False
    if words <= 10 and candidate.score >= 7.0 and "followed-by-body" in reasons:
        return True
    return False


def filter_chapter_sequence_candidates(candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
    if not candidates:
        return []

    structured_non_section_count = sum(
        1
        for cand in candidates
        if "structured-heading" in cand.reasons and not is_section_marker_title(cand.title)
    )
    has_established_structured_pattern = structured_non_section_count >= 3

    filtered: List[HeadingCandidate] = []
    for cand in candidates:
        numeric_heading = "numeric-heading" in cand.reasons
        title_only = "title-only" in cand.reasons and not is_section_marker_title(cand.title)
        if has_established_structured_pattern and (numeric_heading or title_only):
            continue
        filtered.append(cand)

    return filtered


def find_toc_chapter_candidates(candidates: Sequence[HeadingCandidate]) -> List[HeadingCandidate]:
    chapter_candidates = [c for c in candidates if c.kind == "chapter"]
    contents_pages = {
        c.page_num
        for c in candidates
        if c.kind == "front" and re.match(r"(?i)^(contents|table of contents)\b", clean_line_text(c.title))
    }
    if not chapter_candidates:
        return []

    by_page: Dict[int, List[HeadingCandidate]] = defaultdict(list)
    for cand in chapter_candidates:
        by_page[cand.page_num].append(cand)

    toc_pages: set[int] = set()
    for page_num, page_candidates in by_page.items():
        structured_count = sum(
            1
            for cand in page_candidates
            if "structured-heading" in cand.reasons and re.match(r"(?i)^chapter\b", clean_line_text(cand.title))
        )
        dense_sequence = (
            len(page_candidates) >= 6
            and max(c.end_global for c in page_candidates) - min(c.start_global for c in page_candidates) <= len(page_candidates) * 3
        )
        near_contents = any(abs(page_num - contents_page) <= 1 for contents_page in contents_pages)
        if structured_count >= 3 and (near_contents or dense_sequence):
            toc_pages.add(page_num)

    return [c for c in chapter_candidates if c.page_num in toc_pages]


def score_candidate(page: PageData, start_idx: int, end_idx: int, group_lines: List[TextLine]) -> Optional[HeadingCandidate]:
    if not group_lines:
        return None

    title = " ".join(line.text for line in group_lines).strip()
    if not title:
        return None

    kind = assign_candidate_kind(title)
    score = 0.0
    reasons: List[str] = []
    median_font = page.median_font()
    median_gap = max(page.median_gap(), 1.0)
    first = group_lines[0]
    last = group_lines[-1]
    total_words = len(title.split())

    first_is_structured = is_structured_heading_text(first.text)
    title_is_structured = is_structured_heading_text(title)

    if first_is_structured or title_is_structured:
        score += 5.0
        reasons.append("structured-heading")
    elif is_front_matter_heading(title):
        score += 5.0
        reasons.append("front-heading")
    elif is_back_matter_heading(title):
        score += 5.0
        reasons.append("back-heading")
    elif is_numeric_only(first.text):
        score += 3.5
        reasons.append("numeric-heading")
        if len(group_lines) > 1:
            score += 1.2
            reasons.append("numeric-with-subtitle")
    else:
        score += 1.5
        reasons.append("title-only")

    if first.y0 <= page.height * 0.12:
        score += 2.0
        reasons.append("very-top")
    elif first.y0 <= page.height * 0.25:
        score += 1.3
        reasons.append("top")
    elif kind == "chapter" and first.y0 <= page.height * 0.45 and first_is_structured:
        score += 0.7
        reasons.append("midpage-structured")

    centered_ratio = sum(1 for ln in group_lines if is_centered(ln)) / len(group_lines)
    if centered_ratio >= 0.6:
        score += 1.3
        reasons.append("centered")

    if max(ln.font_size for ln in group_lines) >= median_font * 1.15:
        score += 1.2
        reasons.append("large-font")

    if total_words <= 14:
        score += 1.0
        reasons.append("short")
    elif total_words <= 24:
        score += 0.4

    if looks_like_all_caps(title) or looks_like_title_case(title):
        score += 0.8
        reasons.append("display-case")

    if gap_before(page, start_idx) >= median_gap * 1.5 or start_idx == 0:
        score += 1.0
        reasons.append("isolated-before")
    if gap_after(page, end_idx) >= median_gap * 1.5:
        score += 1.0
        reasons.append("isolated-after")

    next_idx = end_idx + 1
    if next_idx < len(page.lines) and body_like_line(page.lines[next_idx].text):
        score += 1.3
        reasons.append("followed-by-body")

    if len(group_lines) > 4 and not is_numeric_only(first.text):
        score -= 1.5
    if len(title) > 140 and not is_numeric_only(first.text):
        score -= 2.0
    if title.lower().startswith(TITLE_ONLY_BAD_PREFIXES):
        score -= 4.0
    if title.endswith((".", "?", "!", ";")) and not title_is_structured:
        score -= 1.5

    if kind == "chapter" and not first_is_structured and not is_numeric_only(first.text):
        if not ((centered_ratio >= 0.6) or (max(ln.font_size for ln in group_lines) >= median_font * 1.15)):
            score -= 1.2
        if first.y0 > page.height * 0.38:
            score -= 1.5
        if len(group_lines) == 1:
            score -= 2.5
        if total_words > 18:
            score -= 2.0

    threshold = 4.5 if kind in {"front", "back"} else 5.0
    if score < threshold:
        return None

    return HeadingCandidate(
        title=title,
        kind=kind,
        score=round(score, 3),
        page_num=page.page_num,
        start_global=group_lines[0].global_index,
        end_global=group_lines[-1].global_index,
        line_indices_on_page=(start_idx, end_idx),
        reasons=reasons,
    )


def generate_candidates_for_page(page: PageData) -> List[HeadingCandidate]:
    candidates: List[HeadingCandidate] = []
    if not page.lines:
        return candidates

    n = len(page.lines)
    top_scan_limit = max(8, min(n, 20))

    for i, line in enumerate(page.lines):
        line_text = line.text
        explicit = bool(
            is_structured_heading_text(line_text)
            or is_front_matter_heading(line_text)
            or is_back_matter_heading(line_text)
        )
        numeric = is_numeric_only(line_text)
        near_top = line.y0 <= page.height * 0.40

        if not explicit and not numeric and not near_top:
            continue
        if not looks_like_heading_line(line_text):
            continue

        if explicit or numeric:
            next_line = page.lines[i + 1] if i + 1 < n else None
            skip_single_numeric = bool(numeric and next_line is not None and looks_like_heading_line(next_line.text))
            if not skip_single_numeric:
                cand = score_candidate(page, i, i, [line])
                if cand:
                    candidates.append(cand)

        if numeric and i < n - 1 and i < top_scan_limit:
            group = [line]
            j = i + 1
            while j < n and len(group) < 5:
                nxt = page.lines[j]
                if not looks_like_heading_line(nxt.text):
                    break
                if body_like_line(nxt.text):
                    break
                if gap_before(page, j) > page.median_gap() * 2.5 and len(group) > 1:
                    break
                group.append(nxt)
                j += 1
            if len(group) >= 2:
                cand = score_candidate(page, i, i + len(group) - 1, group)
                if cand:
                    candidates.append(cand)

        if explicit and i < n - 1:
            group = [line]
            j = i + 1
            while j < n and len(group) < 4:
                nxt = page.lines[j]
                if not looks_like_heading_line(nxt.text) and not looks_like_heading_subtitle(nxt, page):
                    break
                if is_structured_heading_text(nxt.text) or is_front_matter_heading(nxt.text) or is_back_matter_heading(nxt.text):
                    break
                if body_like_line(nxt.text):
                    break
                group.append(nxt)
                j += 1
            if len(group) >= 2:
                cand = score_candidate(page, i, i + len(group) - 1, group)
                if cand:
                    candidates.append(cand)

        if near_top and i < top_scan_limit and not explicit and not numeric:
            group = [line]
            j = i + 1
            while j < n and len(group) < 3:
                nxt = page.lines[j]
                if not looks_like_heading_line(nxt.text):
                    break
                if body_like_line(nxt.text):
                    break
                if gap_before(page, j) > page.median_gap() * 2.2 and len(group) > 1:
                    break
                group.append(nxt)
                j += 1
            if len(group) >= 2:
                cand = score_candidate(page, i, i + len(group) - 1, group)
                if cand:
                    candidates.append(cand)

    candidates.sort(key=lambda c: (c.start_global, -(c.score)))
    selected: List[HeadingCandidate] = []
    for cand in candidates:
        if selected and cand.start_global <= selected[-1].end_global and cand.page_num == selected[-1].page_num:
            if cand.score > selected[-1].score:
                selected[-1] = cand
            continue
        selected.append(cand)
    return selected


def detect_candidates(pages: List[PageData]) -> List[HeadingCandidate]:
    all_candidates: List[HeadingCandidate] = []
    for page in pages:
        all_candidates.extend(generate_candidates_for_page(page))

    all_candidates.sort(key=lambda c: (c.start_global, -c.score))
    deduped: List[HeadingCandidate] = []
    for cand in all_candidates:
        if deduped:
            prev = deduped[-1]
            same_title = normalize_for_counting(cand.title) == normalize_for_counting(prev.title)
            close = abs(cand.page_num - prev.page_num) <= 1
            if same_title and close:
                if cand.score > prev.score:
                    deduped[-1] = cand
                continue
        deduped.append(cand)
    return deduped


def flatten_lines(pages: List[PageData]) -> List[TextLine]:
    return [line for page in pages for line in page.lines]


def select_body_window(
    lines: List[TextLine],
    candidates: List[HeadingCandidate],
    keep_front_matter: bool,
    keep_back_matter: bool,
    start_page_override: Optional[int],
    end_page_override: Optional[int],
    chapter_include_regex: Optional[re.Pattern[str]],
    chapter_exclude_regex: Optional[re.Pattern[str]],
) -> Tuple[int, int, List[HeadingCandidate], Dict[str, object]]:
    if not lines:
        return 0, -1, [], {"start_reason": "empty", "end_reason": "empty"}

    chapter_candidates_all = [c for c in candidates if c.kind == "chapter"]
    toc_chapter_candidates = find_toc_chapter_candidates(candidates)
    toc_candidate_starts = {c.start_global for c in toc_chapter_candidates}
    toc_chapter_titles = [c.title for c in toc_chapter_candidates]
    chapter_candidates_all = [c for c in chapter_candidates_all if c.start_global not in toc_candidate_starts]
    chapter_candidates = [c for c in chapter_candidates_all if is_major_chapter_candidate(c)] or chapter_candidates_all
    chapter_candidates = filter_chapter_sequence_candidates(chapter_candidates)
    if chapter_include_regex is not None:
        chapter_candidates = [c for c in chapter_candidates if matches_optional_regex(c.title, chapter_include_regex)]
    if chapter_exclude_regex is not None:
        chapter_candidates = [c for c in chapter_candidates if not matches_optional_regex(c.title, chapter_exclude_regex)]
    if len(toc_chapter_titles) == len(chapter_candidates):
        for cand, toc_title in zip(chapter_candidates, toc_chapter_titles):
            if is_numeric_only(cand.title):
                cand.title = toc_title
                if "toc-title-alias" not in cand.reasons:
                    cand.reasons.append("toc-title-alias")
    back_candidates = [c for c in candidates if c.kind == "back"]

    if start_page_override is not None:
        start_idx = next((ln.global_index for ln in lines if ln.page_num >= start_page_override), 0)
        start_reason = f"manual-start-page-{start_page_override}"
    elif keep_front_matter:
        start_idx = 0
        start_reason = "keep-front-matter"
    elif chapter_candidates:
        start_idx = chapter_candidates[0].start_global
        start_reason = "first-chapter-candidate"
    else:
        start_idx = 0
        start_reason = "no-chapters-detected"

    if end_page_override is not None:
        reversed_lines = list(reversed(lines))
        chosen = next((ln.global_index for ln in reversed_lines if ln.page_num <= end_page_override), lines[-1].global_index)
        end_idx = chosen
        end_reason = f"manual-end-page-{end_page_override}"
    elif keep_back_matter:
        end_idx = lines[-1].global_index
        end_reason = "keep-back-matter"
    else:
        last_idx = lines[-1].global_index
        if back_candidates:
            last_chapter_page = chapter_candidates[-1].page_num if chapter_candidates else 1
            total_pages = lines[-1].page_num
            usable = [
                c for c in back_candidates
                if c.page_num >= last_chapter_page and c.page_num >= max(last_chapter_page, math.ceil(total_pages * 0.55))
            ]
            if usable:
                end_idx = usable[0].start_global - 1
                end_reason = "first-back-matter-candidate-after-body"
            else:
                end_idx = last_idx
                end_reason = "no-usable-back-matter-candidate"
        else:
            end_idx = last_idx
            end_reason = "no-back-matter-candidate"

    body_candidates = [c for c in chapter_candidates if start_idx <= c.start_global <= end_idx]
    return start_idx, end_idx, body_candidates, {
        "start_reason": start_reason,
        "end_reason": end_reason,
        "toc_chapter_candidates_ignored": len(toc_chapter_candidates),
        "toc_chapter_titles": toc_chapter_titles,
    }


def reconstruct_text_from_lines(lines: Sequence[TextLine]) -> str:
    if not lines:
        return ""
    paragraphs: List[str] = []
    current = ""
    prev_line: Optional[TextLine] = None

    for line in lines:
        text = clean_line_text(line.text)
        if not text:
            if current:
                paragraphs.append(current.strip())
                current = ""
            prev_line = None
            continue

        new_paragraph = False
        if prev_line is None:
            new_paragraph = True
        elif line.page_num != prev_line.page_num:
            new_paragraph = True
        elif gap_before_of_pair(prev_line, line) > max(8.0, prev_line.page_height * 0.012):
            new_paragraph = True

        if new_paragraph:
            if current:
                paragraphs.append(current.strip())
            current = text
        else:
            if current.endswith("-") and text[:1].islower():
                current = current[:-1] + text
            else:
                current = current + " " + text
        prev_line = line

    if current:
        paragraphs.append(current.strip())
    return "\n\n".join(paragraphs).strip()


def gap_before_of_pair(prev_line: TextLine, current_line: TextLine) -> float:
    if prev_line.page_num != current_line.page_num:
        return current_line.y0
    return max(0.0, current_line.y0 - prev_line.y1)


def build_chapters_from_candidates(
    body_lines: List[TextLine],
    chapter_candidates: List[HeadingCandidate],
) -> List[Chapter]:
    if not body_lines:
        return []
    if not chapter_candidates:
        return [Chapter(index=1, title="Full Text", text=reconstruct_text_from_lines(body_lines), start_page=body_lines[0].page_num, end_page=body_lines[-1].page_num)]

    use_section_markers_as_context = any(not is_section_marker_title(c.title) for c in chapter_candidates)
    lines_by_global = {ln.global_index: idx for idx, ln in enumerate(body_lines)}
    chapters: List[Chapter] = []
    active_section_title: Optional[str] = None
    for idx, cand in enumerate(chapter_candidates):
        if use_section_markers_as_context and is_section_marker_title(cand.title):
            active_section_title = cand.title
            continue

        if cand.end_global not in lines_by_global:
            continue
        start_pos = lines_by_global[cand.end_global] + 1
        if idx + 1 < len(chapter_candidates):
            next_start = chapter_candidates[idx + 1].start_global
            if next_start in lines_by_global:
                end_pos = lines_by_global[next_start] - 1
            else:
                end_pos = len(body_lines) - 1
        else:
            end_pos = len(body_lines) - 1

        if start_pos > end_pos:
            continue
        chapter_lines = body_lines[start_pos : end_pos + 1]
        chapter_text = reconstruct_text_from_lines(chapter_lines)
        if not chapter_text.strip():
            continue
        chapters.append(
            Chapter(
                index=len(chapters) + 1,
                title=prefixed_chapter_title(active_section_title, cand.title),
                text=chapter_text,
                start_page=chapter_lines[0].page_num,
                end_page=chapter_lines[-1].page_num,
            )
        )

    if not chapters:
        return [Chapter(index=1, title="Full Text", text=reconstruct_text_from_lines(body_lines), start_page=body_lines[0].page_num, end_page=body_lines[-1].page_num)]
    return chapters


def split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paragraphs:
        return paragraphs
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def token_ids_no_warning(tokenizer: AutoTokenizer, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return list(encoded["input_ids"])


def token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(token_ids_no_warning(tokenizer, text))


def chunk_chapter(chapter_text: str, tokenizer: AutoTokenizer, max_tokens: int, overlap_tokens: int = 64) -> List[str]:
    paragraphs = split_paragraphs(chapter_text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = token_count(tokenizer, para)

        if para_tokens > max_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_tokens = 0

            para_ids = token_ids_no_warning(tokenizer, para)
            step = max(1, max_tokens - overlap_tokens)
            for start in range(0, len(para_ids), step):
                piece_ids = para_ids[start : start + max_tokens]
                piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
                if piece:
                    chunks.append(piece)
                if start + max_tokens >= len(para_ids):
                    break
            continue

        projected = current_tokens + para_tokens + (1 if current_parts else 0)
        if projected <= max_tokens:
            current_parts.append(para)
            current_tokens = projected
            continue

        prev_chunk = "\n\n".join(current_parts).strip()
        if prev_chunk:
            chunks.append(prev_chunk)

        overlap_prefix = ""
        if prev_chunk and overlap_tokens > 0:
            prev_ids = token_ids_no_warning(tokenizer, prev_chunk)
            overlap_prefix = tokenizer.decode(prev_ids[-overlap_tokens:], skip_special_tokens=True).strip()

        if overlap_prefix:
            current_parts = [overlap_prefix, para]
            current_tokens = token_count(tokenizer, overlap_prefix) + para_tokens
        else:
            current_parts = [para]
            current_tokens = para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def excerpt_text(text: str, max_chars: int = 900) -> str:
    clean = normalize_whitespace(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def extract_character_names(text: str, limit: int = 8) -> List[str]:
    candidates = re.findall(r"\b(?:Mr|Mrs|Miss|Madame|Princess|Prince|Doctor|Dr)\.?\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+|\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]{2,}\b", text)
    counts: Counter[str] = Counter()
    for raw in candidates:
        name = clean_line_text(raw).strip(".,;:!?\"'()[]")
        first = name.split()[0] if name else ""
        if not name or first in CHARACTER_NAME_STOPWORDS:
            continue
        if len(name) <= 2:
            continue
        counts[name] += 1
    return [name for name, _ in counts.most_common(limit)]


def lexical_emotion_cues(text: str) -> List[str]:
    lower = text.lower()
    return [emotion for emotion, pattern in CONTEXT_EMOTION_LEXICON.items() if re.search(pattern, lower)]


SCENE_BREAK_REGEX = re.compile(
    r"""(?ix)
    ^(?:\*[\s*]{1,}\*|[-_]{3,}|chapter\s+\d+|part\s+\w+)$
    |^(?:letter|telegram|diary|journal|memorandum)\b
    |^(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b
    |^(?:\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b)
    |^(?:(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b)
    """
)


TIME_SHIFT_REGEX = re.compile(
    r"(?i)\b(later|meanwhile|that evening|the next morning|the following day|at midnight|at dawn|afterwards|afterward)\b"
)


CHARACTER_NAME_STOPWORDS = {
    "I",
    "He",
    "She",
    "It",
    "We",
    "They",
    "The",
    "A",
    "An",
    "This",
    "That",
    "But",
    "And",
    "Then",
    "There",
    "When",
    "Where",
    "What",
    "Why",
    "How",
    "Chapter",
    "Part",
    "Book",
}


CONTEXT_EMOTION_LEXICON = {
    "sadness": r"\b(sad|sorrow|grief|tears|wept|weep|misery|pain|despair|mourn|dead|death|lonely)\b",
    "fear": r"\b(fear|afraid|terror|dread|trembling|anxious|anxiety|panic|fright|horror)\b",
    "remorse": r"\b(guilt|remorse|ashamed|shame|forgive|forgiveness|sin|regret)\b",
    "caring": r"\b(pity|compassion|mercy|kindness|tender|sympathy|comfort|protect)\b",
    "joy": r"\b(laughed|laugh|smiled|smile|joy|delight|happy|glad|pleasure|merry)\b",
    "love": r"\b(love|beloved|adore|kiss|heart|passion|longing|dear)\b",
    "anger": r"\b(angry|anger|rage|wrath|furious|hate|hatred|irritated)\b",
    "embarrassment": r"\b(embarrass|ashamed|blush|blushed|awkward|confusion|confused)\b",
    "surprise": r"\b(suddenly|amazement|astonished|surprised|unexpected|wonder)\b",
}


def is_scene_break_paragraph(paragraph: str) -> bool:
    text = clean_line_text(paragraph)
    if not text:
        return False
    if SCENE_BREAK_REGEX.search(text):
        return True
    return count_words(text) <= 18 and TIME_SHIFT_REGEX.search(text) is not None


def has_direct_approval_cue(text: str) -> bool:
    lower = text.lower()
    positive = re.search(r"\b(approve|approved|praise|praised|well done|right thing)\b", lower)
    if not positive:
        return False
    negated = re.search(
        r"\b(not|never|no|without|rather than|instead of|refuse(?:d)? to)\b.{0,35}\b(approve|approved|praise|praised)\b",
        lower,
    )
    return not bool(negated)


def segment_chapter_into_scenes(chapter: Chapter, config: PipelineConfig) -> List[Scene]:
    paragraphs = split_paragraphs(chapter.text)
    if not paragraphs:
        return [Scene(chapter_index=chapter.index, scene_index=1, text=chapter.text, start_char=0, end_char=len(chapter.text))]

    scenes: List[Scene] = []
    current_parts: List[str] = []
    current_words = 0
    current_start = 0
    search_from = 0

    for para in paragraphs:
        para_start = chapter.text.find(para, search_from)
        if para_start < 0:
            para_start = search_from
        para_end = para_start + len(para)
        para_words = count_words(para)

        boundary_break = bool(current_parts and current_words >= config.scene_min_words and is_scene_break_paragraph(para))
        should_close = boundary_break or (
            current_parts
            and current_words >= config.scene_min_words
            and current_words + para_words > config.scene_target_words
        )
        if should_close:
            scene_text = "\n\n".join(current_parts).strip()
            scenes.append(
                Scene(
                    chapter_index=chapter.index,
                    scene_index=len(scenes) + 1,
                    text=scene_text,
                    start_char=current_start,
                    end_char=current_start + len(scene_text),
                )
            )
            current_parts = []
            current_words = 0
            current_start = para_start

        if not current_parts:
            current_start = para_start
        current_parts.append(para)
        current_words += para_words
        search_from = para_end

    if current_parts:
        scene_text = "\n\n".join(current_parts).strip()
        scenes.append(
            Scene(
                chapter_index=chapter.index,
                scene_index=len(scenes) + 1,
                text=scene_text,
                start_char=current_start,
                end_char=current_start + len(scene_text),
            )
        )

    if len(scenes) > 1 and count_words(scenes[-1].text) < config.scene_min_words:
        last = scenes.pop()
        prev = scenes[-1]
        merged = f"{prev.text}\n\n{last.text}".strip()
        scenes[-1] = Scene(
            chapter_index=prev.chapter_index,
            scene_index=prev.scene_index,
            text=merged,
            start_char=prev.start_char,
            end_char=last.end_char,
        )

    return scenes


def segment_chapters_into_scenes(chapters: Sequence[Chapter], config: PipelineConfig) -> List[Scene]:
    scenes: List[Scene] = []
    for chapter in chapters:
        scenes.extend(segment_chapter_into_scenes(chapter, config))
    return scenes


class LLMSceneAnalyzer:
    def __init__(
        self,
        enabled: bool = False,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
        timeout_seconds: int = 30,
        use_heuristic_context: bool = True,
    ) -> None:
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.use_heuristic_context = use_heuristic_context
        self.llm_requested = bool(enabled)
        self.api_key_found = bool(os.environ.get("OPENAI_API_KEY"))
        self.stats: Counter[str] = Counter()
        self.errors: List[str] = []

    def analyze(
        self,
        scene: Scene,
        previous_scene: Optional[Scene] = None,
        next_scene: Optional[Scene] = None,
        chapter_title: Optional[str] = None,
    ) -> Optional[SceneContext]:
        if self.enabled and self.provider == "ollama":
            try:
                context = self._analyze_with_ollama(scene, previous_scene=previous_scene, next_scene=next_scene, chapter_title=chapter_title)
                self.stats["llm"] += 1
                return context
            except Exception as exc:
                message = f"Scene {scene.chapter_index}.{scene.scene_index}: {exc}"
                self.errors.append(message)
                if not self.use_heuristic_context:
                    self.stats["llm_error"] += 1
                    return SceneContext(
                        chapter_index=scene.chapter_index,
                        scene_index=scene.scene_index,
                        source="llm_error",
                        confidence=0.0,
                        warnings=[f"Local LLM analysis failed: {exc}"],
                    )
        elif self.enabled and self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            try:
                context = self._analyze_with_openai(scene, previous_scene=previous_scene, next_scene=next_scene, chapter_title=chapter_title)
                self.stats["llm"] += 1
                return context
            except Exception as exc:
                message = f"Scene {scene.chapter_index}.{scene.scene_index}: {exc}"
                self.errors.append(message)
                if not self.use_heuristic_context:
                    self.stats["llm_error"] += 1
                    return SceneContext(
                        chapter_index=scene.chapter_index,
                        scene_index=scene.scene_index,
                        source="llm_error",
                        confidence=0.0,
                        warnings=[f"LLM analysis failed: {exc}"],
                    )
        if self.use_heuristic_context:
            context = self._analyze_with_heuristics(scene, previous_scene=previous_scene, next_scene=next_scene, chapter_title=chapter_title)
            self.stats["heuristic"] += 1
            return context
        self.stats["none"] += 1
        return None

    def diagnostics(self) -> Dict[str, object]:
        return {
            "requested": self.llm_requested,
            "provider": self.provider,
            "model": self.model,
            "api_key_found": self.api_key_found,
            "ollama_url": self.ollama_url,
            "heuristic_fallback_enabled": self.use_heuristic_context,
            "llm_contexts": int(self.stats.get("llm", 0)),
            "heuristic_contexts": int(self.stats.get("heuristic", 0)),
            "llm_error_contexts": int(self.stats.get("llm_error", 0)),
            "no_contexts": int(self.stats.get("none", 0)),
            "errors": self.errors[:10],
        }

    def _scene_context_prompt(
        self,
        scene: Scene,
        previous_scene: Optional[Scene] = None,
        next_scene: Optional[Scene] = None,
        chapter_title: Optional[str] = None,
    ) -> str:
        context_parts = []
        if chapter_title:
            context_parts.append(f"Chapter title: {chapter_title}")
        if previous_scene is not None:
            context_parts.append(f"Previous scene excerpt: {excerpt_text(previous_scene.text, 900)}")
        if next_scene is not None:
            context_parts.append(f"Next scene excerpt: {excerpt_text(next_scene.text, 700)}")
        surrounding = "\n".join(context_parts)
        return (
            "Analyze this English literary scene as a scene, not as isolated keywords. "
            "Return valid JSON only with keys: summary, events, characters, speaker_emotions, "
            "character_emotions, narrator_emotions, narrator_tone, scene_tone, conflict, "
            "dominant_perspective, approval_interpretation. For approval_interpretation use "
            "is_present, meaning, is_direct_approval, explanation. Distinguish direct approval "
            "from compassion, forgiveness, moral acceptance, respect, irony, and unclear. "
            "Use surrounding excerpts only to interpret continuity; label emotions for the current scene.\n\n"
            f"{surrounding}\n\n"
            f"Scene:\n{scene.text[:6000]}"
        )

    def _analyze_with_ollama(
        self,
        scene: Scene,
        previous_scene: Optional[Scene] = None,
        next_scene: Optional[Scene] = None,
        chapter_title: Optional[str] = None,
    ) -> SceneContext:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": "You are a precise literary emotion analyst. Return valid JSON only."},
                {"role": "user", "content": self._scene_context_prompt(scene, previous_scene, next_scene, chapter_title)},
            ],
        }
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.ollama_url}: {exc.reason}") from exc
        content = data.get("message", {}).get("content", "")
        parsed = json.loads(content)
        return self._context_from_mapping(scene, parsed, source="llm", confidence=0.78)

    def _analyze_with_openai(
        self,
        scene: Scene,
        previous_scene: Optional[Scene] = None,
        next_scene: Optional[Scene] = None,
        chapter_title: Optional[str] = None,
    ) -> SceneContext:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise literary emotion analyst. Return valid JSON only."},
                {"role": "user", "content": self._scene_context_prompt(scene, previous_scene, next_scene, chapter_title)},
            ],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return self._context_from_mapping(scene, parsed, source="llm", confidence=0.85)

    def _context_from_mapping(
        self,
        scene: Scene,
        data: Dict[str, object],
        source: str,
        confidence: float,
    ) -> SceneContext:
        def list_of_str(value: object) -> List[str]:
            if isinstance(value, list):
                return [str(item) for item in value]
            if isinstance(value, str) and value:
                return [value]
            return []

        speaker = data.get("speaker_emotions", {})
        character = data.get("character_emotions", speaker if isinstance(speaker, dict) else {})
        return SceneContext(
            chapter_index=scene.chapter_index,
            scene_index=scene.scene_index,
            summary=str(data.get("summary", "")),
            events=list_of_str(data.get("events", [])),
            characters=list_of_str(data.get("characters", [])),
            speaker_emotions={str(k): list_of_str(v) for k, v in speaker.items()} if isinstance(speaker, dict) else {},
            character_emotions={str(k): list_of_str(v) for k, v in character.items()} if isinstance(character, dict) else {},
            narrator_emotions=list_of_str(data.get("narrator_emotions", [])),
            narrator_tone=list_of_str(data.get("narrator_tone", [])),
            scene_tone=list_of_str(data.get("scene_tone", [])),
            conflict=str(data.get("conflict")) if data.get("conflict") is not None else None,
            approval_interpretation=data.get("approval_interpretation", {}) if isinstance(data.get("approval_interpretation"), dict) else {},
            dominant_perspective=str(data.get("dominant_perspective", "unknown")),
            source=source,
            confidence=confidence,
        )

    def _analyze_with_heuristics(
        self,
        scene: Scene,
        previous_scene: Optional[Scene] = None,
        next_scene: Optional[Scene] = None,
        chapter_title: Optional[str] = None,
    ) -> SceneContext:
        text = scene.text
        lower = text.lower()
        tones: List[str] = lexical_emotion_cues(text)
        narrator_emotions: List[str] = []
        character_emotions: Dict[str, List[str]] = {}
        characters = extract_character_names(text)
        if previous_scene is not None:
            previous_characters = set(extract_character_names(previous_scene.text, limit=12))
            characters = sorted(characters, key=lambda name: (name not in previous_characters, name))[:8]
        quote_count = text.count('"') + text.count("'")
        dominant = "dialogue" if quote_count >= 4 else "narrator"
        if dominant == "narrator":
            narrator_emotions = tones[:]
        elif characters and tones:
            for name in characters[:4]:
                character_emotions[name] = tones[:4]
        approval_meaning = "unclear"
        direct_approval = has_direct_approval_cue(lower)
        if not direct_approval and any(t in tones for t in ("caring", "remorse", "sadness")):
            approval_meaning = "compassion/moral acceptance"
        elif direct_approval:
            approval_meaning = "direct approval"
        conflict = None
        if re.search(r"\b(against|quarrel|argue|angry|refuse|jealous|struggle|fight|fear|hate|threat)\b", lower):
            conflict = "Heuristic conflict cue detected."
        confidence = 0.45
        if tones:
            confidence += 0.15
        if quote_count:
            confidence += 0.05
        if previous_scene is not None or next_scene is not None:
            confidence += 0.05
        return SceneContext(
            chapter_index=scene.chapter_index,
            scene_index=scene.scene_index,
            summary="Heuristic context from lexical, character-name, dialogue, and neighboring-scene cues.",
            events=[],
            characters=characters,
            character_emotions=character_emotions,
            narrator_emotions=narrator_emotions,
            narrator_tone=tones if dominant == "narrator" else [],
            scene_tone=tones,
            conflict=conflict,
            approval_interpretation={
                "is_present": direct_approval or "caring" in tones or "remorse" in tones,
                "meaning": approval_meaning,
                "is_direct_approval": direct_approval,
                "explanation": "Heuristic fallback; use LLM for stronger evidence.",
            },
            dominant_perspective=dominant,
            source="heuristic",
            confidence=min(confidence, 0.75),
            warnings=["heuristic context fallback"],
        )


def normalize_scene_scores(
    scene: Scene,
    context: Optional[SceneContext],
    raw_scores: Dict[str, float],
    use_context_normalization: bool,
) -> Tuple[Dict[str, float], List[str]]:
    normalized = dict(raw_scores)
    notes: List[str] = []
    if not use_context_normalization:
        notes.append("context normalization disabled")
        return normalized, notes

    approval = normalized.get("approval", 0.0)
    if approval <= 0:
        return normalized, notes

    direct_approval = False
    approval_meaning = ""
    scene_tone: List[str] = []
    if context is not None:
        approval_info = context.approval_interpretation or {}
        direct_approval = bool(approval_info.get("is_direct_approval"))
        approval_meaning = str(approval_info.get("meaning", "")).lower()
        scene_tone = [tone.lower() for tone in context.scene_tone + context.narrator_tone]

    non_direct_meanings = ("compassion", "forgiveness", "moral acceptance", "respect")
    tragic_tones = ("sadness", "grief", "remorse", "fear", "anxiety", "despair", "tragic", "tense")
    tone_suggests_not_approval = any(tone in scene_tone for tone in tragic_tones)
    raw_suggests_not_approval = (
        normalized.get("sadness", 0.0) >= approval * 0.65
        or normalized.get("fear", 0.0) >= approval * 0.55
        or normalized.get("remorse", 0.0) >= approval * 0.45
    )
    meaning_suggests_not_approval = any(marker in approval_meaning for marker in non_direct_meanings)

    if not direct_approval and (meaning_suggests_not_approval or tone_suggests_not_approval or raw_suggests_not_approval):
        old_approval = approval
        normalized["approval"] = old_approval * 0.55
        if "caring" in normalized:
            normalized["caring"] = min(1.0, normalized.get("caring", 0.0) + old_approval * 0.22)
        if "sadness" in normalized:
            normalized["sadness"] = min(1.0, normalized.get("sadness", 0.0) + old_approval * 0.12)
        if "love" in normalized:
            normalized["love"] = min(1.0, normalized.get("love", 0.0) + old_approval * 0.06)
        notes.append("approval lowered: context or co-emotions suggest compassion/moral acceptance, not direct approval")

    if not direct_approval and normalized.get("joy", 0.0) > 0 and (
        tone_suggests_not_approval or normalized.get("sadness", 0.0) > normalized.get("joy", 0.0)
    ):
        normalized["joy"] = normalized["joy"] * 0.75
        notes.append("joy softened: scene appears tense or tragic")

    return normalized, notes


def score_confidence(
    scene: Scene,
    context: Optional[SceneContext],
    raw_scores: Dict[str, float],
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    word_count = count_words(scene.text)
    confidence = 0.62
    if word_count < 80:
        confidence -= 0.18
        reasons.append("short scene")
    elif word_count > 1200:
        confidence -= 0.08
        reasons.append("long scene")
    sorted_scores = sorted(
        [(label, score) for label, score in raw_scores.items() if label.lower() != "neutral"],
        key=lambda item: item[1],
        reverse=True,
    )
    if len(sorted_scores) >= 2:
        gap = sorted_scores[0][1] - sorted_scores[1][1]
        if gap < 0.025:
            confidence -= 0.12
            reasons.append("close competing emotions")
        elif gap > 0.12:
            confidence += 0.08
            reasons.append("clear dominant emotion")
    if context is not None:
        confidence += min(0.18, context.confidence * 0.18)
        if context.source == "heuristic":
            reasons.append("heuristic context")
        elif context.source == "llm":
            reasons.append("llm context")
    return max(0.05, min(1.0, confidence)), reasons


def emotion_intensity(scores: Dict[str, float], include_neutral: bool = False) -> float:
    values = [score for label, score in scores.items() if include_neutral or label.lower() != "neutral"]
    return max(values) if values else 0.0


def scene_weight(scene_word_count: int, scores: Dict[str, float], confidence: float) -> float:
    length_weight = max(1.0, float(scene_word_count))
    intensity_weight = 0.75 + min(0.75, emotion_intensity(scores) * 2.0)
    confidence_weight = 0.5 + confidence
    return length_weight * intensity_weight * confidence_weight


def apply_corpus_calibration(
    scene_results: Sequence[SceneEmotionResult],
    include_neutral: bool,
) -> Dict[str, object]:
    if not scene_results:
        return {"enabled": True, "baselines": {}, "method": "no scenes"}

    labels = sorted(scene_results[0].normalized_scores.keys())
    baselines: Dict[str, float] = {}
    for label in labels:
        values = sorted(result.normalized_scores.get(label, 0.0) for result in scene_results)
        mid = len(values) // 2
        if len(values) % 2:
            baselines[label] = values[mid]
        else:
            baselines[label] = (values[mid - 1] + values[mid]) / 2.0

    for result in scene_results:
        calibrated: Dict[str, float] = {}
        for label, score in result.normalized_scores.items():
            low = label.lower()
            if low == "neutral":
                value = score if include_neutral else score * 0.70
            else:
                baseline = baselines.get(label, 0.0)
                floor = baseline * 0.35
                if floor >= 0.95:
                    value = score
                else:
                    value = max(0.0, score - floor) / max(0.05, 1.0 - floor)
            calibrated[label] = round(min(1.0, max(0.0, value)), 6)
        result.normalized_scores = dict(sorted(calibrated.items()))
        result.scene_weight = round(scene_weight(result.scene_word_count, result.normalized_scores, result.confidence), 6)
        result.notes.append("corpus calibration applied: per-emotion baseline dampened, scene peaks preserved")

    return {
        "enabled": True,
        "method": "median-baseline dampening with peak-preserving rescale",
        "baselines": {k: round(v, 6) for k, v in sorted(baselines.items())},
        "scene_count": len(scene_results),
    }


def split_perspective_scores(
    scores: Dict[str, float],
    context: Optional[SceneContext],
) -> Tuple[Dict[str, float], Dict[str, float], str]:
    if context is None:
        return dict(scores), dict(scores), "unknown"
    dominant = context.dominant_perspective or "unknown"
    narrator_markers = {item.lower() for item in context.narrator_tone + context.narrator_emotions}
    character_markers = {
        emotion.lower()
        for emotions in list(context.character_emotions.values()) + list(context.speaker_emotions.values())
        for emotion in emotions
    }
    narrator_scores: Dict[str, float] = {}
    character_scores: Dict[str, float] = {}
    for label, score in scores.items():
        low = label.lower()
        narrator_factor = 0.75 if low in narrator_markers or dominant == "narrator" else 0.35
        character_factor = 0.75 if low in character_markers or dominant in {"dialogue", "character"} else 0.35
        narrator_scores[label] = round(score * narrator_factor, 6)
        character_scores[label] = round(score * character_factor, 6)
    return character_scores, narrator_scores, dominant


def aggregate_chunk_scores(
    chunks: Sequence[str],
    chunk_scores: Sequence[Dict[str, float]],
    peak_blend: float,
) -> Dict[str, float]:
    if not chunk_scores:
        return {}
    labels = list(chunk_scores[0].keys())
    weights = [max(1.0, float(count_words(chunk))) for chunk in chunks]
    total_weight = float(sum(weights) or len(chunk_scores))
    blend = max(0.0, min(0.75, peak_blend))
    aggregated: Dict[str, float] = {}
    for label in labels:
        weighted_mean = sum(row.get(label, 0.0) * weight for row, weight in zip(chunk_scores, weights)) / total_weight
        peak = max(row.get(label, 0.0) for row in chunk_scores)
        if label.lower() == "neutral":
            value = weighted_mean
        else:
            value = weighted_mean * (1.0 - blend) + peak * blend
        aggregated[label] = float(min(1.0, max(0.0, value)))
    return aggregated


def predict_scene_scores(
    scene: Scene,
    analyzer: BookEmotionAnalyzer,
    overlap_tokens: int,
    peak_blend: float = 0.30,
) -> Tuple[Dict[str, float], int]:
    chunks = chunk_chapter(scene.text, analyzer.tokenizer, analyzer.max_length, overlap_tokens=overlap_tokens)
    if not chunks:
        return {}, 0
    predictions = analyzer.predict_texts(chunks)
    return aggregate_chunk_scores(chunks, predictions, peak_blend=peak_blend), len(chunks)


def aggregate_scene_results_by_chapter(
    chapters: Sequence[Chapter],
    scene_results: Sequence[SceneEmotionResult],
    include_neutral: bool,
    exclude_neutral_from_profiles: bool = True,
) -> List[ChapterEmotionProfile]:
    by_chapter: Dict[int, List[SceneEmotionResult]] = defaultdict(list)
    for result in scene_results:
        by_chapter[result.chapter_index].append(result)

    profiles: List[ChapterEmotionProfile] = []
    for chapter in chapters:
        results = by_chapter.get(chapter.index, [])
        if not results:
            continue
        labels = sorted(results[0].normalized_scores.keys())
        if exclude_neutral_from_profiles and not include_neutral:
            labels = [label for label in labels if label.lower() != "neutral"]
        scene_weights = [max(1.0, result.scene_weight) for result in results]
        total_weight = float(sum(scene_weights) or len(results))
        average_scores = {
            label: sum(result.normalized_scores.get(label, 0.0) * weight for result, weight in zip(results, scene_weights)) / total_weight
            for label in labels
        }
        raw_average_scores = {
            label: sum(result.raw_classifier_scores.get(label, 0.0) * weight for result, weight in zip(results, scene_weights)) / total_weight
            for label in labels
        }
        max_scores = {label: max(result.normalized_scores.get(label, 0.0) for result in results) for label in labels}
        intensity_rows: List[Dict[str, object]] = []
        for result in results:
            non_neutral_scores = [
                score for label, score in result.normalized_scores.items() if include_neutral or label.lower() != "neutral"
            ]
            intensity = max(non_neutral_scores) if non_neutral_scores else 0.0
            intensity_rows.append(
                {
                    "chapter_index": result.chapter_index,
                    "scene_index": result.scene_index,
                    "intensity": round(float(intensity), 6),
                    "confidence": result.confidence,
                    "scene_weight": result.scene_weight,
                    "top_emotions": top_emotions(result.normalized_scores, top_k=3, include_neutral=include_neutral),
                }
            )
        top_scenes_by_emotion: Dict[str, List[Dict[str, object]]] = {}
        for label in labels:
            rows = [
                {
                    "chapter_index": result.chapter_index,
                    "scene_index": result.scene_index,
                    "score": result.normalized_scores.get(label, 0.0),
                    "confidence": result.confidence,
                }
                for result in results
            ]
            top_scenes_by_emotion[label] = sorted(rows, key=lambda row: row["score"], reverse=True)[:3]
        profiles.append(
            ChapterEmotionProfile(
                chapter_index=chapter.index,
                chapter_title=chapter.title,
                scene_count=len(results),
                average_scores={k: round(v, 6) for k, v in sorted(average_scores.items())},
                max_scores={k: round(v, 6) for k, v in sorted(max_scores.items())},
                dominant_emotions=top_emotions(average_scores, top_k=5, include_neutral=include_neutral),
                most_intense_scenes=sorted(intensity_rows, key=lambda row: row["intensity"], reverse=True)[:5],
                top_scenes_by_emotion=top_scenes_by_emotion,
                raw_average_scores={k: round(v, 6) for k, v in sorted(raw_average_scores.items())},
            )
        )
    return profiles


def build_book_emotion_profile(
    chapter_profiles: Sequence[ChapterEmotionProfile],
    scene_results: Sequence[SceneEmotionResult],
    include_neutral: bool,
) -> Dict[str, object]:
    if not chapter_profiles:
        return {}
    labels = sorted(chapter_profiles[0].average_scores.keys())
    if not include_neutral:
        labels = [label for label in labels if label.lower() != "neutral"]
    chapter_scores = [profile.average_scores for profile in chapter_profiles]
    book_scores = aggregate_scores(chapter_scores)
    arcs = []
    for profile in chapter_profiles:
        arcs.append(
            {
                "chapter_index": profile.chapter_index,
                "chapter_title": profile.chapter_title,
                "top_emotions": profile.dominant_emotions[:5],
                "intensity": round(emotion_intensity(profile.average_scores, include_neutral=include_neutral), 6),
            }
        )

    trend_rows = []
    for label in labels:
        first = chapter_profiles[0].average_scores.get(label, 0.0)
        last = chapter_profiles[-1].average_scores.get(label, 0.0)
        peak_profile = max(chapter_profiles, key=lambda profile: profile.average_scores.get(label, 0.0))
        trend_rows.append(
            {
                "label": label,
                "first_score": round(first, 6),
                "last_score": round(last, 6),
                "delta": round(last - first, 6),
                "peak_chapter": peak_profile.chapter_index,
                "peak_score": round(peak_profile.average_scores.get(label, 0.0), 6),
            }
        )

    narrator_scores = aggregate_scores([result.narrator_emotion_scores for result in scene_results]) if scene_results else {}
    character_scores = aggregate_scores([result.character_emotion_scores for result in scene_results]) if scene_results else {}
    confidence_values = [result.confidence for result in scene_results]
    return {
        "book_level_top_emotions": top_emotions(book_scores, top_k=10, include_neutral=include_neutral),
        "book_level_scores": {k: round(v, 6) for k, v in sorted(book_scores.items())},
        "chapter_arc": arcs,
        "rising_emotions": sorted(trend_rows, key=lambda row: row["delta"], reverse=True)[:8],
        "falling_emotions": sorted(trend_rows, key=lambda row: row["delta"])[:8],
        "emotion_peaks": sorted(trend_rows, key=lambda row: row["peak_score"], reverse=True)[:12],
        "perspective_profiles": {
            "narrator_top_emotions": top_emotions(narrator_scores, top_k=8, include_neutral=include_neutral),
            "character_top_emotions": top_emotions(character_scores, top_k=8, include_neutral=include_neutral),
        },
        "average_scene_confidence": round(sum(confidence_values) / len(confidence_values), 6) if confidence_values else 0.0,
    }


def build_character_emotion_profiles(
    scene_results: Sequence[SceneEmotionResult],
    include_neutral: bool,
) -> Dict[str, object]:
    character_counts: Counter[str] = Counter()
    character_emotions: Dict[str, Counter[str]] = defaultdict(Counter)
    character_scene_refs: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for result in scene_results:
        context = result.context
        if context is None:
            continue
        names = context.characters or list(context.character_emotions.keys()) or list(context.speaker_emotions.keys())
        for name in names:
            clean_name = clean_line_text(name)
            if not clean_name:
                continue
            character_counts[clean_name] += 1
            explicit_emotions = list(context.character_emotions.get(clean_name, [])) + list(context.speaker_emotions.get(clean_name, []))
            for emotion in explicit_emotions:
                character_emotions[clean_name][emotion.lower()] += 2
            for item in top_emotions(result.character_emotion_scores, top_k=3, include_neutral=include_neutral):
                character_emotions[clean_name][item["label"]] += max(1, int(round(item["score"] * 10)))
            if len(character_scene_refs[clean_name]) < 8:
                character_scene_refs[clean_name].append(
                    {
                        "chapter_index": result.chapter_index,
                        "scene_index": result.scene_index,
                        "top_emotions": top_emotions(result.character_emotion_scores, top_k=3, include_neutral=include_neutral),
                    }
                )

    profiles = []
    for name, count in character_counts.most_common():
        emotion_counter = character_emotions.get(name, Counter())
        total = float(sum(emotion_counter.values()) or 1.0)
        profiles.append(
            {
                "character": name,
                "scene_mentions": count,
                "top_emotions": [
                    {"label": label, "score": round(value / total, 6)}
                    for label, value in emotion_counter.most_common(8)
                    if include_neutral or label.lower() != "neutral"
                ],
                "sample_scenes": character_scene_refs.get(name, []),
            }
        )

    return {"characters": profiles, "character_count": len(profiles)}


def dataclass_list_to_dicts(items: Sequence[object]) -> List[Dict[str, object]]:
    return [asdict(item) for item in items]


def aggregate_scores(chunk_scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not chunk_scores:
        return {}
    labels = sorted({label for row in chunk_scores for label in row.keys()})
    return {label: float(sum(row.get(label, 0.0) for row in chunk_scores) / len(chunk_scores)) for label in labels}


def top_emotions(scores: Dict[str, float], top_k: int = 5, include_neutral: bool = False) -> List[Dict[str, float]]:
    items = list(scores.items())
    if not include_neutral:
        items = [(k, v) for k, v in items if k.lower() != "neutral"]
    pairs = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
    return [{"label": label, "score": round(score, 6)} for label, score in pairs]


def filter_plot_labels(score_dicts: List[Dict[str, float]], include_neutral: bool, top_n: int) -> List[str]:
    if not score_dicts:
        return []
    labels = list(score_dicts[0].keys())
    if not include_neutral:
        labels = [label for label in labels if label.lower() != "neutral"]
    averages = {label: sum(d.get(label, 0.0) for d in score_dicts) / len(score_dicts) for label in labels}
    return [label for label, _ in sorted(averages.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def create_line_chart(chapter_payloads: List[Dict], output_path: Path, include_neutral: bool, top_n: int) -> None:
    if not chapter_payloads:
        return
    score_dicts = [chapter["scores"] for chapter in chapter_payloads]
    labels = filter_plot_labels(score_dicts, include_neutral, top_n)
    if not labels:
        return
    x = [chapter["chapter_index"] for chapter in chapter_payloads]
    plt.figure(figsize=(12, 7))
    for label in labels:
        y = [chapter["scores"].get(label, 0.0) for chapter in chapter_payloads]
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.xlabel("Chapter")
    plt.ylabel("Emotion score")
    plt.title("Emotion trajectory across chapters")
    plt.xticks(x)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_block_results(
    scene_results: Sequence[SceneEmotionResult],
    threshold: float,
    include_neutral: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    block_results: List[Dict[str, object]] = []
    csv_rows: List[Dict[str, object]] = []
    for block_index, result in enumerate(scene_results, start=1):
        dominant = [
            {"label": label, "score": round(score, 6)}
            for label, score in sorted(result.normalized_scores.items(), key=lambda x: x[1], reverse=True)
            if score >= threshold and (include_neutral or label.lower() != "neutral")
        ]
        payload = {
            "block_index": block_index,
            "chapter_index": result.chapter_index,
            "scene_index": result.scene_index,
            "block_title": f"Block {block_index}",
            "scene_ref": f"{result.chapter_index}.{result.scene_index}",
            "scene_word_count": result.scene_word_count,
            "classifier_chunk_count": result.classifier_chunk_count,
            "confidence": result.confidence,
            "confidence_reasons": result.confidence_reasons,
            "scene_weight": result.scene_weight,
            "dominant_perspective": result.dominant_perspective,
            "top_emotions": top_emotions(result.normalized_scores, top_k=5, include_neutral=include_neutral),
            "dominant_emotions_above_threshold": dominant,
            "scores": result.normalized_scores,
            "raw_scores": result.raw_classifier_scores,
            "character_emotion_scores": result.character_emotion_scores,
            "narrator_emotion_scores": result.narrator_emotion_scores,
            "notes": result.notes,
        }
        block_results.append(payload)

        row = {
            "block_index": block_index,
            "chapter_index": result.chapter_index,
            "scene_index": result.scene_index,
            "scene_word_count": result.scene_word_count,
            "classifier_chunk_count": result.classifier_chunk_count,
            "confidence": result.confidence,
            "top_1_emotion": payload["top_emotions"][0]["label"] if payload["top_emotions"] else "",
            "top_1_score": payload["top_emotions"][0]["score"] if payload["top_emotions"] else "",
        }
        row.update({label: round(score, 6) for label, score in sorted(result.normalized_scores.items())})
        csv_rows.append(row)
    return block_results, csv_rows


def create_block_line_chart(block_payloads: List[Dict[str, object]], output_path: Path, include_neutral: bool, top_n: int) -> None:
    if not block_payloads:
        return
    score_dicts = [block["scores"] for block in block_payloads if isinstance(block.get("scores"), dict)]
    labels = filter_plot_labels(score_dicts, include_neutral, top_n)
    if not labels:
        return
    x = [int(block["block_index"]) for block in block_payloads]
    plt.figure(figsize=(13, 7))
    for label in labels:
        y = [block["scores"].get(label, 0.0) for block in block_payloads]  # type: ignore[union-attr]
        plt.plot(x, y, marker="o", linewidth=1.8, label=label)
    plt.xlabel("Text block")
    plt.ylabel("Emotion score")
    plt.title("Emotion trajectory across text blocks")
    plt.xticks(x if len(x) <= 30 else x[:: max(1, len(x) // 20)])
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def create_chapter_bar_charts(chapter_payloads: List[Dict], output_dir: Path, include_neutral: bool, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapter_payloads:
        items = list(chapter["scores"].items())
        if not include_neutral:
            items = [(k, v) for k, v in items if k.lower() != "neutral"]
        items = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
        if not items:
            continue
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        plt.figure(figsize=(11, 6))
        plt.bar(labels, values)
        plt.xlabel("Emotion")
        plt.ylabel("Score")
        plt.title(f"Chapter {chapter['chapter_index']}: {chapter['chapter_title']}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"chapter_{chapter['chapter_index']:03d}_{chapter['chapter_title']}")
        plt.savefig(output_dir / f"{safe_name}.png", dpi=180)
        plt.close()


def save_chapter_texts(chapters: List[Chapter], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"chapter_{chapter.index:03d}_{chapter.title}")
        (output_dir / f"{safe_name}.txt").write_text(chapter.text, encoding="utf-8")


def save_block_texts(scenes: Sequence[Scene], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for block_index, scene in enumerate(scenes, start=1):
        safe_name = f"block_{block_index:03d}_chapter_{scene.chapter_index:03d}_scene_{scene.scene_index:03d}"
        (output_dir / f"{safe_name}.txt").write_text(scene.text, encoding="utf-8")


def run_detection(
    input_path: Path,
    output_dir: Path,
    keep_front_matter: bool,
    keep_back_matter: bool,
    start_page: Optional[int],
    end_page: Optional[int],
    chapter_regex: Optional[str],
    exclude_chapter_regex: Optional[str],
    max_chapters: Optional[int],
    expected_chapters: Optional[int],
    strict_chapter_count: bool,
    force_single_chapter: bool = False,
) -> Dict[str, object]:
    if input_path.suffix.lower() == ".pdf":
        pages = extract_pdf_pages(input_path)
        split_method = "layout_scoring_pdf"
    elif input_path.suffix.lower() == ".txt":
        pages = extract_txt_pages(input_path)
        split_method = "text_scoring_txt"
    else:
        raise ValueError("Unsupported file type. Use .pdf or .txt")

    pages, removed_repeated = remove_repeated_headers_footers(pages)
    lines = flatten_lines(pages)
    if not lines:
        raise ValueError("No extractable text found after cleaning.")

    candidates = detect_candidates(pages)
    chapter_include_regex = compile_optional_regex(chapter_regex, "chapter")
    chapter_exclude_regex = compile_optional_regex(exclude_chapter_regex, "exclude-chapter")
    start_idx, end_idx, chapter_candidates, body_meta = select_body_window(
        lines=lines,
        candidates=candidates,
        keep_front_matter=keep_front_matter,
        keep_back_matter=keep_back_matter,
        start_page_override=start_page,
        end_page_override=end_page,
        chapter_include_regex=chapter_include_regex,
        chapter_exclude_regex=chapter_exclude_regex,
    )
    body_lines = [ln for ln in lines if start_idx <= ln.global_index <= end_idx]
    chapters = build_chapters_from_candidates(body_lines, chapter_candidates)
    if force_single_chapter and body_lines:
        chapters = [
            Chapter(
                index=1,
                title="Full Text",
                text=reconstruct_text_from_lines(body_lines),
                start_page=body_lines[0].page_num,
                end_page=body_lines[-1].page_num,
            )
        ]
    if max_chapters is not None:
        if max_chapters < 1:
            raise ValueError("--max_chapters must be greater than zero")
        chapters = chapters[:max_chapters]
    if max_chapters is not None and chapters:
        cleaned_main_text = "\n\n".join(f"{chapter.title}\n\n{chapter.text}" for chapter in chapters).strip()
    else:
        cleaned_main_text = reconstruct_text_from_lines(body_lines)
    warnings: List[str] = []
    if expected_chapters is not None and len(chapters) != expected_chapters:
        message = f"Expected {expected_chapters} chapters, detected {len(chapters)}."
        if strict_chapter_count:
            raise ValueError(message)
        warnings.append(message)

    cleaned_path = output_dir / "cleaned_main_text.txt"
    cleaned_path.write_text(cleaned_main_text, encoding="utf-8")
    (output_dir / "cleaned_text.txt").write_text(cleaned_main_text, encoding="utf-8")
    save_chapter_texts(chapters, output_dir / "chapter_texts")

    report = {
        "input_file": str(input_path),
        "split_method": split_method,
        "total_pages": len(pages),
        "total_lines_after_cleaning": len(lines),
        "removed_repeated_headers_footers": removed_repeated,
        "body_start_global": start_idx,
        "body_end_global": end_idx,
        "body_start_page": body_lines[0].page_num if body_lines else None,
        "body_end_page": body_lines[-1].page_num if body_lines else None,
        "body_start_reason": body_meta["start_reason"],
        "body_end_reason": body_meta["end_reason"],
        "toc_chapter_candidates_ignored": body_meta.get("toc_chapter_candidates_ignored", 0),
        "toc_chapter_titles": body_meta.get("toc_chapter_titles", []),
        "chapter_regex": chapter_regex,
        "exclude_chapter_regex": exclude_chapter_regex,
        "max_chapters": max_chapters,
        "expected_chapters": expected_chapters,
        "force_single_chapter": force_single_chapter,
        "warnings": warnings,
        "candidate_counts": Counter(c.kind for c in candidates),
        "all_candidates": [
            {
                "title": c.title,
                "kind": c.kind,
                "score": c.score,
                "page_num": c.page_num,
                "start_global": c.start_global,
                "end_global": c.end_global,
                "reasons": c.reasons,
            }
            for c in candidates
        ],
        "selected_chapter_candidates": [
            {
                "title": c.title,
                "score": c.score,
                "page_num": c.page_num,
                "reasons": c.reasons,
            }
            for c in chapter_candidates
        ],
        "chapters": [
            {
                "chapter_index": chapter.index,
                "chapter_title": chapter.title,
                "start_page": chapter.start_page,
                "end_page": chapter.end_page,
                "char_count": len(chapter.text),
            }
            for chapter in chapters
        ],
    }
    report_path = output_dir / "chapter_detection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "boundary_detection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "chapters.json").write_text(
        json.dumps(dataclass_list_to_dicts(chapters), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "pages": pages,
        "lines": lines,
        "chapters": chapters,
        "cleaned_text_path": str(cleaned_path),
        "report_path": str(report_path),
        "split_method": split_method,
        "warnings": warnings,
    }


def analyze_book(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    max_length: int,
    overlap_tokens: int,
    batch_size: int,
    threshold: float,
    include_neutral: bool,
    plot_top_n: int,
    chapter_bar_top_n: int,
    keep_front_matter: bool,
    keep_back_matter: bool,
    start_page: Optional[int],
    end_page: Optional[int],
    chapter_regex: Optional[str],
    exclude_chapter_regex: Optional[str],
    max_chapters: Optional[int],
    expected_chapters: Optional[int],
    strict_chapter_count: bool,
    force_single_chapter: bool,
    detect_only: bool,
    clean_output: bool,
    use_llm_scene_analysis: bool,
    use_context_normalization: bool,
    save_intermediate_json: bool,
    llm_provider: str,
    llm_model: str,
    ollama_url: str,
    llm_timeout_seconds: int,
    use_heuristic_context: bool,
    scene_min_words: int,
    scene_target_words: int,
    exclude_neutral_from_profiles: bool,
    use_corpus_calibration: bool,
    context_window_scenes: int,
    classifier_peak_blend: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_outputs = clean_generated_outputs(output_dir) if clean_output else []

    detection = run_detection(
        input_path=input_path,
        output_dir=output_dir,
        keep_front_matter=keep_front_matter,
        keep_back_matter=keep_back_matter,
        start_page=start_page,
        end_page=end_page,
        chapter_regex=chapter_regex,
        exclude_chapter_regex=exclude_chapter_regex,
        max_chapters=max_chapters,
        expected_chapters=expected_chapters,
        strict_chapter_count=strict_chapter_count,
        force_single_chapter=force_single_chapter,
    )
    chapters: List[Chapter] = detection["chapters"]  # type: ignore[assignment]

    if detect_only:
        return {
            "json_path": None,
            "csv_path": None,
            "summary_path": None,
            "plot_path": None,
            "num_chapters": len(chapters),
            "cleaned_text_path": detection["cleaned_text_path"],
            "report_path": detection["report_path"],
            "split_method": detection["split_method"],
            "removed_outputs": removed_outputs,
            "warnings": detection.get("warnings", []),
        }

    analyzer = BookEmotionAnalyzer(model_name=model_name, max_length=max_length, batch_size=batch_size)
    config = PipelineConfig(
        use_llm_scene_analysis=use_llm_scene_analysis,
        use_context_normalization=use_context_normalization,
        save_intermediate_json=save_intermediate_json,
        llm_provider=llm_provider,
        llm_model=llm_model,
        ollama_url=ollama_url,
        llm_timeout_seconds=llm_timeout_seconds,
        use_heuristic_context=use_heuristic_context,
        scene_min_words=scene_min_words,
        scene_target_words=scene_target_words,
        exclude_neutral_from_profiles=exclude_neutral_from_profiles,
        use_corpus_calibration=use_corpus_calibration,
        context_window_scenes=context_window_scenes,
        classifier_peak_blend=classifier_peak_blend,
    )
    scenes = segment_chapters_into_scenes(chapters, config)
    save_block_texts(scenes, output_dir / "block_texts")
    llm_analyzer = LLMSceneAnalyzer(
        enabled=config.use_llm_scene_analysis,
        provider=config.llm_provider,
        model=config.llm_model,
        ollama_url=config.ollama_url,
        timeout_seconds=config.llm_timeout_seconds,
        use_heuristic_context=config.use_heuristic_context,
    )

    scene_results: List[SceneEmotionResult] = []
    llm_contexts: List[Optional[Dict[str, object]]] = []
    chapters_by_index = {chapter.index: chapter for chapter in chapters}
    for scene_pos, scene in enumerate(tqdm(scenes, desc="Analyzing scenes")):
        raw_scores, chunk_count = predict_scene_scores(
            scene,
            analyzer,
            overlap_tokens=overlap_tokens,
            peak_blend=config.classifier_peak_blend,
        )
        if not raw_scores:
            continue
        previous_scene = scenes[scene_pos - 1] if config.context_window_scenes > 0 and scene_pos > 0 else None
        next_scene = scenes[scene_pos + 1] if config.context_window_scenes > 0 and scene_pos + 1 < len(scenes) else None
        chapter_title = chapters_by_index.get(scene.chapter_index).title if scene.chapter_index in chapters_by_index else None
        context = llm_analyzer.analyze(
            scene,
            previous_scene=previous_scene,
            next_scene=next_scene,
            chapter_title=chapter_title,
        )
        normalized_scores, notes = normalize_scene_scores(
            scene=scene,
            context=context,
            raw_scores=raw_scores,
            use_context_normalization=config.use_context_normalization,
        )
        confidence, confidence_reasons = score_confidence(scene, context, raw_scores)
        weight = scene_weight(count_words(scene.text), normalized_scores, confidence)
        character_scores, narrator_scores, dominant_perspective = split_perspective_scores(normalized_scores, context)
        scene_results.append(
            SceneEmotionResult(
                chapter_index=scene.chapter_index,
                scene_index=scene.scene_index,
                context=context,
                raw_classifier_scores={k: round(v, 6) for k, v in sorted(raw_scores.items())},
                normalized_scores={k: round(v, 6) for k, v in sorted(normalized_scores.items())},
                scene_word_count=count_words(scene.text),
                classifier_chunk_count=chunk_count,
                confidence=round(confidence, 6),
                confidence_reasons=confidence_reasons,
                scene_weight=round(weight, 6),
                dominant_perspective=dominant_perspective,
                character_emotion_scores=character_scores,
                narrator_emotion_scores=narrator_scores,
                notes=notes,
            )
        )
        llm_contexts.append(asdict(context) if context else None)

    if config.use_corpus_calibration:
        calibration_report = apply_corpus_calibration(scene_results, include_neutral=include_neutral)
        for result in scene_results:
            character_scores, narrator_scores, dominant_perspective = split_perspective_scores(result.normalized_scores, result.context)
            result.character_emotion_scores = character_scores
            result.narrator_emotion_scores = narrator_scores
            result.dominant_perspective = dominant_perspective
    else:
        calibration_report = {"enabled": False}

    llm_diagnostics = llm_analyzer.diagnostics()
    llm_warnings: List[str] = []
    if (
        llm_diagnostics["requested"]
        and llm_diagnostics["provider"] == "openai"
        and not llm_diagnostics["api_key_found"]
    ):
        llm_warnings.append("LLM requested but OPENAI_API_KEY was not found; heuristic fallback was used.")
    if (
        llm_diagnostics["requested"]
        and llm_diagnostics["provider"] == "ollama"
        and llm_diagnostics["llm_contexts"] == 0
    ):
        llm_warnings.append("Local LLM requested, but no scene received Ollama context; check that Ollama is running and the model is pulled.")
    if (
        llm_diagnostics["requested"]
        and llm_diagnostics["provider"] == "openai"
        and llm_diagnostics["llm_contexts"] == 0
        and llm_diagnostics["api_key_found"]
    ):
        llm_warnings.append("LLM requested and API key found, but no scene received LLM context; inspect llm_diagnostics.errors.")
    if llm_diagnostics["errors"]:
        first_error = str(llm_diagnostics["errors"][0])
        llm_warnings.append(f"First LLM error: {first_error}")

    chapter_profiles = aggregate_scene_results_by_chapter(
        chapters,
        scene_results,
        include_neutral=include_neutral,
        exclude_neutral_from_profiles=config.exclude_neutral_from_profiles,
    )
    book_profile = build_book_emotion_profile(chapter_profiles, scene_results, include_neutral=include_neutral)
    character_profiles = build_character_emotion_profiles(scene_results, include_neutral=include_neutral)
    chapter_results: List[Dict] = []
    csv_rows: List[Dict] = []
    for profile in chapter_profiles:
        chapter = chapters_by_index[profile.chapter_index]
        scores = profile.average_scores
        raw_scores = profile.raw_average_scores
        dominant = [
            {"label": label, "score": round(score, 6)}
            for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if score >= threshold and (include_neutral or label.lower() != "neutral")
        ]
        payload = {
            "chapter_index": chapter.index,
            "chapter_title": chapter.title,
            "start_page": chapter.start_page,
            "end_page": chapter.end_page,
            "chapter_char_count": len(chapter.text),
            "scene_count": profile.scene_count,
            "top_emotions": profile.dominant_emotions,
            "dominant_emotions_above_threshold": dominant,
            "scores": scores,
            "raw_scores": raw_scores,
            "max_scores": profile.max_scores,
            "most_intense_scenes": profile.most_intense_scenes,
        }
        chapter_results.append(payload)

        row = {
            "chapter_index": chapter.index,
            "chapter_title": chapter.title,
            "start_page": chapter.start_page,
            "end_page": chapter.end_page,
            "chapter_char_count": len(chapter.text),
            "scene_count": profile.scene_count,
            "top_1_emotion": payload["top_emotions"][0]["label"] if payload["top_emotions"] else "",
            "top_1_score": payload["top_emotions"][0]["score"] if payload["top_emotions"] else "",
        }
        row.update({label: round(score, 6) for label, score in sorted(scores.items())})
        csv_rows.append(row)

    block_results, block_csv_rows = build_block_results(scene_results, threshold=threshold, include_neutral=include_neutral)
    book_scores = aggregate_scores([profile.average_scores for profile in chapter_profiles]) if chapter_profiles else {}
    raw_book_scores = aggregate_scores([profile.raw_average_scores for profile in chapter_profiles]) if chapter_profiles else {}
    result = {
        "input_file": str(input_path),
        "model_name": model_name,
        "split_method": detection["split_method"],
        "num_chapters": len(chapter_results),
        "num_scenes": len(scenes),
        "max_length": max_length,
        "overlap_tokens": overlap_tokens,
        "threshold": threshold,
        "include_neutral": include_neutral,
        "pipeline_config": asdict(config),
        "llm_diagnostics": llm_diagnostics,
        "calibration_report": calibration_report,
        "book_level_top_emotions": top_emotions(book_scores, top_k=10, include_neutral=include_neutral),
        "book_level_scores": {k: round(v, 6) for k, v in sorted(book_scores.items())},
        "raw_book_level_scores": {k: round(v, 6) for k, v in sorted(raw_book_scores.items())},
        "book_emotion_profile": book_profile,
        "character_emotion_profiles": character_profiles,
        "chapters": chapter_results,
        "blocks": block_results,
        "scene_results": dataclass_list_to_dicts(scene_results),
    }

    json_path = output_dir / "chapter_emotions.json"
    csv_path = output_dir / "chapter_emotions.csv"
    block_json_path = output_dir / "block_emotions.json"
    block_csv_path = output_dir / "block_emotions.csv"
    summary_path = output_dir / "chapter_emotions_summary.txt"
    plot_path = output_dir / "chapter_emotions_plot.png"
    block_plot_path = output_dir / "block_emotions_plot.png"
    bar_dir = output_dir / "chapter_bar_charts"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8")
    block_json_path.write_text(json.dumps(block_results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(block_csv_rows).to_csv(block_csv_path, index=False, encoding="utf-8")
    if config.save_intermediate_json:
        (output_dir / "scenes.json").write_text(
            json.dumps(dataclass_list_to_dicts(scenes), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "raw_scene_emotions.json").write_text(
            json.dumps(
                [
                    {
                        "chapter_index": item.chapter_index,
                        "scene_index": item.scene_index,
                        "scene_word_count": item.scene_word_count,
                        "classifier_chunk_count": item.classifier_chunk_count,
                        "confidence": item.confidence,
                        "confidence_reasons": item.confidence_reasons,
                        "scene_weight": item.scene_weight,
                        "dominant_perspective": item.dominant_perspective,
                        "raw_classifier_scores": item.raw_classifier_scores,
                    }
                    for item in scene_results
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "llm_scene_contexts.json").write_text(
            json.dumps(llm_contexts, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "normalized_scene_emotions.json").write_text(
            json.dumps(dataclass_list_to_dicts(scene_results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "chapter_emotion_profiles.json").write_text(
            json.dumps(dataclass_list_to_dicts(chapter_profiles), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "book_emotion_profile.json").write_text(
            json.dumps(book_profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "character_emotion_profiles.json").write_text(
            json.dumps(character_profiles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "calibration_report.json").write_text(
            json.dumps(calibration_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary_lines = [
        f"Input: {input_path}",
        f"Model: {model_name}",
        f"Split method: {detection['split_method']}",
        f"Chapters analyzed: {len(chapter_results)}",
        f"Blocks analyzed: {len(block_results)}",
        f"Scenes analyzed: {len(scene_results)}",
        f"LLM requested: {llm_diagnostics['requested']}",
        f"LLM provider: {llm_diagnostics['provider']}",
        f"LLM model: {llm_diagnostics['model']}",
        f"OPENAI_API_KEY found: {llm_diagnostics['api_key_found']}",
        f"LLM contexts: {llm_diagnostics['llm_contexts']}",
        f"Heuristic contexts: {llm_diagnostics['heuristic_contexts']}",
        f"Corpus calibration: {calibration_report.get('enabled', False)}",
        f"Detection report: {detection['report_path']}",
        f"Cleaned text: {detection['cleaned_text_path']}",
        "",
        "Book-level top emotions:",
    ]
    for item in result["book_level_top_emotions"]:
        summary_lines.append(f"- {item['label']}: {item['score']:.4f}")
    summary_lines.append("")
    summary_lines.append("Per-chapter top emotions:")
    for chapter in chapter_results:
        pretty = ", ".join(f"{x['label']}={x['score']:.4f}" for x in chapter["top_emotions"][:3])
        summary_lines.append(f"- {chapter['chapter_title']}: {pretty}")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    create_line_chart(chapter_results, plot_path, include_neutral=include_neutral, top_n=plot_top_n)
    create_block_line_chart(block_results, block_plot_path, include_neutral=include_neutral, top_n=plot_top_n)
    create_chapter_bar_charts(chapter_results, bar_dir, include_neutral=include_neutral, top_n=chapter_bar_top_n)

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "block_json_path": str(block_json_path),
        "block_csv_path": str(block_csv_path),
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
        "block_plot_path": str(block_plot_path),
        "num_chapters": len(chapter_results),
        "num_blocks": len(block_results),
        "num_scenes": len(scene_results),
        "cleaned_text_path": detection["cleaned_text_path"],
        "report_path": detection["report_path"],
        "split_method": detection["split_method"],
        "removed_outputs": removed_outputs,
        "llm_diagnostics": llm_diagnostics,
        "warnings": list(detection.get("warnings", [])) + llm_warnings,
    }


def run_book_emotion_analysis(
    input_file: str | Path,
    output_dir: str | Path,
    config: Optional[PipelineConfig] = None,
    **kwargs: object,
) -> Dict[str, object]:
    config = config or PipelineConfig()
    params = {
        "input_path": Path(input_file),
        "output_dir": Path(output_dir),
        "model_name": kwargs.get("model_name", DEFAULT_MODEL_NAME),
        "max_length": int(kwargs.get("max_length", 512)),
        "overlap_tokens": int(kwargs.get("overlap_tokens", 64)),
        "batch_size": int(kwargs.get("batch_size", 8)),
        "threshold": float(kwargs.get("threshold", 0.30)),
        "include_neutral": bool(kwargs.get("include_neutral", False)),
        "plot_top_n": int(kwargs.get("plot_top_n", 8)),
        "chapter_bar_top_n": int(kwargs.get("chapter_bar_top_n", 10)),
        "keep_front_matter": bool(kwargs.get("keep_front_matter", False)),
        "keep_back_matter": bool(kwargs.get("keep_back_matter", False)),
        "start_page": kwargs.get("start_page"),
        "end_page": kwargs.get("end_page"),
        "chapter_regex": kwargs.get("chapter_regex"),
        "exclude_chapter_regex": kwargs.get("exclude_chapter_regex"),
        "max_chapters": kwargs.get("max_chapters"),
        "expected_chapters": kwargs.get("expected_chapters"),
        "strict_chapter_count": bool(kwargs.get("strict_chapter_count", False)),
        "force_single_chapter": bool(kwargs.get("force_single_chapter", False)),
        "detect_only": bool(kwargs.get("detect_only", False)),
        "clean_output": bool(kwargs.get("clean_output", True)),
        "use_llm_scene_analysis": config.use_llm_scene_analysis,
        "use_context_normalization": config.use_context_normalization,
        "save_intermediate_json": config.save_intermediate_json,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "ollama_url": config.ollama_url,
        "llm_timeout_seconds": config.llm_timeout_seconds,
        "use_heuristic_context": config.use_heuristic_context,
        "scene_min_words": config.scene_min_words,
        "scene_target_words": config.scene_target_words,
        "exclude_neutral_from_profiles": config.exclude_neutral_from_profiles,
        "use_corpus_calibration": config.use_corpus_calibration,
        "context_window_scenes": config.context_window_scenes,
        "classifier_peak_blend": config.classifier_peak_blend,
    }
    return analyze_book(**params)  # type: ignore[arg-type]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze chapter-level emotion trajectories from literary PDF/TXT files.")
    parser.add_argument("--input", required=True, help="Path to .pdf or .txt file")
    parser.add_argument("--output_dir", "--output", required=True, help="Directory for output files")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help=f"HF model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length per chunk")
    parser.add_argument("--overlap_tokens", type=int, default=64, help="Token overlap between chunks")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.30, help="Threshold for dominant emotions")
    parser.add_argument("--plot_top_n", type=int, default=8, help="How many emotions to show on the global line chart")
    parser.add_argument("--chapter_bar_top_n", type=int, default=10, help="How many emotions to show on each chapter bar chart")
    parser.add_argument("--include_neutral", action="store_true", help="Include 'neutral' on charts and top-emotion outputs")
    parser.add_argument("--keep_front_matter", action="store_true", help="Do not trim front matter before the main text")
    parser.add_argument("--keep_back_matter", action="store_true", help="Do not trim back matter after the main text")
    parser.add_argument("--start_page", type=int, default=None, help="Manual override: start analysis from this page number")
    parser.add_argument("--end_page", type=int, default=None, help="Manual override: end analysis at this page number")
    parser.add_argument("--chapter_regex", default=None, help="Only use detected headings matching this regex as chapter starts")
    parser.add_argument("--exclude_chapter_regex", default=None, help="Ignore detected headings matching this regex")
    parser.add_argument("--max_chapters", type=int, default=None, help="Analyze only the first N detected chapters after filtering")
    parser.add_argument("--expected_chapters", type=int, default=None, help="Expected number of chapters; recorded as a warning if it differs")
    parser.add_argument("--strict_chapter_count", action="store_true", help="Raise an error if --expected_chapters does not match")
    parser.add_argument("--force_single_chapter", action="store_true", help="Treat the body as one text, useful for short stories without chapters")
    parser.add_argument("--use_llm", type=parse_bool, default=False, help="Enable LLM scene analysis integration point (default: false)")
    parser.add_argument("--normalize", type=parse_bool, default=True, help="Enable context-aware score normalization (default: true)")
    parser.add_argument("--save_intermediate_json", type=parse_bool, default=True, help="Save scene and profile JSON files (default: true)")
    parser.add_argument("--llm_provider", choices=("ollama", "openai"), default="ollama", help="LLM provider for scene context")
    parser.add_argument("--llm_model", default="qwen2.5:7b", help="LLM model for scene context")
    parser.add_argument("--ollama_url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--llm_timeout_seconds", type=int, default=30, help="Timeout for each LLM scene request")
    parser.add_argument("--heuristic_context", type=parse_bool, default=True, help="Use local heuristic scene context when LLM is off/unavailable")
    parser.add_argument("--scene_min_words", type=int, default=250, help="Minimum target scene size in words")
    parser.add_argument("--scene_target_words", type=int, default=1000, help="Preferred scene size in words")
    parser.add_argument("--block_min_words", type=int, default=None, help="Alias for --scene_min_words when using block-level analysis")
    parser.add_argument("--block_target_words", type=int, default=None, help="Alias for --scene_target_words when using block-level analysis")
    parser.add_argument("--exclude_neutral_from_profiles", type=parse_bool, default=True, help="Exclude neutral from chapter profiles/charts unless --include_neutral is set")
    parser.add_argument("--corpus_calibration", type=parse_bool, default=True, help="Apply per-book emotion baseline calibration (default: true)")
    parser.add_argument("--context_window_scenes", type=int, default=1, help="Neighboring scenes included in context analysis")
    parser.add_argument("--classifier_peak_blend", type=float, default=0.30, help="Blend chunk mean with strongest chunk signal for long scenes")
    parser.add_argument("--detect_only", action="store_true", help="Only detect/trim/split chapters without running the emotion model")
    parser.add_argument("--no_clean_output", action="store_true", help="Keep previous generated files in the output directory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    result = analyze_book(
        input_path=input_path,
        output_dir=output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        overlap_tokens=args.overlap_tokens,
        batch_size=args.batch_size,
        threshold=args.threshold,
        include_neutral=args.include_neutral,
        plot_top_n=args.plot_top_n,
        chapter_bar_top_n=args.chapter_bar_top_n,
        keep_front_matter=args.keep_front_matter,
        keep_back_matter=args.keep_back_matter,
        start_page=args.start_page,
        end_page=args.end_page,
        chapter_regex=args.chapter_regex,
        exclude_chapter_regex=args.exclude_chapter_regex,
        max_chapters=args.max_chapters,
        expected_chapters=args.expected_chapters,
        strict_chapter_count=args.strict_chapter_count,
        force_single_chapter=args.force_single_chapter,
        detect_only=args.detect_only,
        clean_output=not args.no_clean_output,
        use_llm_scene_analysis=args.use_llm,
        use_context_normalization=args.normalize,
        save_intermediate_json=args.save_intermediate_json,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        llm_timeout_seconds=args.llm_timeout_seconds,
        use_heuristic_context=args.heuristic_context,
        scene_min_words=args.block_min_words if args.block_min_words is not None else args.scene_min_words,
        scene_target_words=args.block_target_words if args.block_target_words is not None else args.scene_target_words,
        exclude_neutral_from_profiles=args.exclude_neutral_from_profiles,
        use_corpus_calibration=args.corpus_calibration,
        context_window_scenes=args.context_window_scenes,
        classifier_peak_blend=args.classifier_peak_blend,
    )

    print("Done.")
    print(f"Split method: {result['split_method']}")
    print(f"Chapters analyzed: {result['num_chapters']}")
    if result.get("num_blocks") is not None:
        print(f"Blocks analyzed: {result['num_blocks']}")
    if result.get("num_scenes") is not None:
        print(f"Scenes analyzed: {result['num_scenes']}")
    if result.get("llm_diagnostics"):
        diag = result["llm_diagnostics"]
        print(
            "LLM diagnostics: "
            f"requested={diag['requested']}, "
            f"provider={diag['provider']}, "
            f"model={diag['model']}, "
            f"api_key_found={diag['api_key_found']}, "
            f"llm_contexts={diag['llm_contexts']}, "
            f"heuristic_contexts={diag['heuristic_contexts']}, "
            f"errors={len(diag['errors'])}"
        )
    if result.get("removed_outputs"):
        print(f"Removed old generated outputs: {len(result['removed_outputs'])}")
    for warning in result.get("warnings", []):
        print(f"Warning: {warning}")
    print(f"Cleaned text: {result['cleaned_text_path']}")
    print(f"Detection report: {result['report_path']}")
    if result.get("json_path"):
        print(f"JSON: {result['json_path']}")
        print(f"CSV: {result['csv_path']}")
        print(f"Block JSON: {result['block_json_path']}")
        print(f"Block CSV: {result['block_csv_path']}")
        print(f"Summary: {result['summary_path']}")
        print(f"Plot: {result['plot_path']}")
        print(f"Block plot: {result['block_plot_path']}")


if __name__ == "__main__":
    main()
