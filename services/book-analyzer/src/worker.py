import asyncio
import json
import logging
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import grpc
import nats
import redis.asyncio as redis

from analyze_book_emotions import (
    BookEmotionAnalyzer,
    PipelineConfig,
    run_book_emotion_analysis,
)

from .config import Config
from .proto import text_storage_pb2 as ts_pb
from .proto import text_storage_pb2_grpc as ts_grpc
from .translator import Translator, detect_language


log = logging.getLogger(__name__)

JOBS_SUBJECT = "jobs.process"
QUEUE_GROUP = "book-analyzers"


def _decode_txt(content: bytes) -> str:
    """Decode TXT bytes to UTF-8 with cp1251 fallback for legacy Russian files."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        log.info("txt is not UTF-8, falling back to cp1251")
        return content.decode("cp1251", errors="replace")


def _extract_pdf_text(content: bytes) -> str:
    """Extract clean plain text from PDF bytes for language detection /
    translation. We delegate to the model package's own pipeline because it
    handles the things naive ``page.get_text("text")`` does not:

      * line-grouping by font/position (better paragraph reconstruction),
      * repeated header/footer detection across pages,
      * isolated page-number lines stripped out,
      * dehyphenation of words broken across lines.

    Cleaner input = much less noise for the translator, which means fewer
    hallucinations on long literary texts.
    """
    import tempfile
    from pathlib import Path

    from analyze_book_emotions import (
        extract_pdf_pages,
        is_page_number_line,
        normalize_whitespace,
        remove_repeated_headers_footers,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        pages = extract_pdf_pages(tmp_path)
        pages, _ = remove_repeated_headers_footers(pages)

        out_pages: list[str] = []
        for page in pages:
            page_lines: list[str] = []
            for line in page.lines:
                text = normalize_whitespace(line.text)
                if not text or is_page_number_line(text):
                    continue
                if page_lines and page_lines[-1].endswith("-"):
                    page_lines[-1] = page_lines[-1][:-1] + text
                else:
                    page_lines.append(text)
            if page_lines:
                out_pages.append("\n".join(page_lines))
        return "\n\n".join(out_pages)
    finally:
        tmp_path.unlink(missing_ok=True)

# Поля слишком объёмные для хранения и отправки на фронтенд
_LARGE_RESULT_FIELDS = ("scene_results", "raw_book_level_scores")


class Worker:
    def __init__(self, config: Config):
        self.config = config
        self.nc: nats.aio.client.Client | None = None
        self.rdb: redis.Redis | None = None
        self.ts_channel: grpc.aio.Channel | None = None
        self.ts_client: ts_grpc.TextStorageServiceStub | None = None
        self.translator = Translator()
        # Один worker обрабатывает одну задачу за раз, потому что модель однопоточная
        self.lock = asyncio.Lock()
        self._stop = asyncio.Event()

    async def start(self) -> None:
        # Заранее прогреваем модель в HF-кеше, чтобы первая задача не ждала
        # загрузку, сам пайплайн создаёт собственный анализатор, поэтому это
        # только прогрев
        log.info("warming up emotion model: %s", self.config.model_name_or_path)
        await asyncio.to_thread(
            BookEmotionAnalyzer,
            model_name=self.config.model_name_or_path,
        )
        log.info("model warmed up")

        self.rdb = redis.from_url(f"redis://{self.config.redis_addr}", decode_responses=True)
        await self.rdb.ping()
        log.info("redis connected: %s", self.config.redis_addr)

        max_msg_size = 64 * 1024 * 1024  # match Go services
        self.ts_channel = grpc.aio.insecure_channel(
            self.config.text_storage_addr,
            options=[
                ("grpc.max_send_message_length", max_msg_size),
                ("grpc.max_receive_message_length", max_msg_size),
            ],
        )
        self.ts_client = ts_grpc.TextStorageServiceStub(self.ts_channel)
        log.info("text-storage connected: %s", self.config.text_storage_addr)

        self.nc = await nats.connect(self.config.nats_url)
        log.info("nats connected: %s", self.config.nats_url)

        await self.nc.subscribe(JOBS_SUBJECT, queue=QUEUE_GROUP, cb=self._on_message)
        log.info("subscribed: %s (queue=%s)", JOBS_SUBJECT, QUEUE_GROUP)

    async def wait(self) -> None:
        await self._stop.wait()

    async def stop(self) -> None:
        log.info("stopping worker")
        self._stop.set()
        if self.nc is not None:
            await self.nc.drain()
        if self.ts_channel is not None:
            await self.ts_channel.close()
        if self.rdb is not None:
            await self.rdb.aclose()

    async def _on_message(self, msg) -> None:
        job_id = msg.data.decode()
        log.info("received job: %s", job_id)
        async with self.lock:
            cancel_event = threading.Event()
            try:
                await asyncio.wait_for(
                    self._process_job(job_id, cancel_event),
                    timeout=self.config.job_timeout_seconds,
                )
            except asyncio.TimeoutError:
                # Сигнализируем  остановиться на границе следующего батча
                cancel_event.set()
                err = f"job exceeded {self.config.job_timeout_seconds}s timeout"
                log.error("%s: %s", job_id, err)
                await self._update_status(job_id, "failed", err)
            except Exception as e:
                log.exception("failed to process job %s", job_id)
                await self._update_status(job_id, "failed", str(e))

    async def _process_job(self, job_id: str, cancel_event: threading.Event) -> None:
        job = await self._get_job(job_id)
        if job is None:
            log.warning("job %s not found in redis, skipping", job_id)
            return

        await self._update_status(job_id, "processing")

        text_resp = await self.ts_client.GetText(ts_pb.GetTextRequest(text_id=job["text_id"]))
        content_bytes = text_resp.text.content
        fmt = (text_resp.text.format or "txt").lower()
        title = job.get("title") or text_resp.text.title or "input"
        use_llm = bool(job.get("use_llm", False))
        use_paragraph_analysis = bool(job.get("use_paragraph_analysis", False))
        log.info(
            "analyzing job=%s text_id=%s format=%s bytes=%d use_llm=%s use_paragraph_analysis=%s",
            job_id, job["text_id"], fmt, len(content_bytes), use_llm, use_paragraph_analysis,
        )

        # Перевод нужен только для текстовых форматов, PDF передаём как есть,
        # чтобы сохранить работу PyMuPDF-пайплайна с разметкой
        source_language = "en"
        translated = False
        analysis_input: bytes | str
        analysis_suffix: str

        if fmt == "pdf":
            # Язык определяем по фрагменту извлечённого текста, английские PDF
            # идут через нативный пайплайн модели с учётом разметки страниц
            # Русские PDF извлекаются, переводятся и анализируются как TXT:
            # жертвуем layout-детекцией ради корректного скоринга эмоций
            sampled = await asyncio.to_thread(_extract_pdf_text, content_bytes)
            source_language = detect_language(sampled)
            log.info("job=%s pdf sampled=%d chars, language=%s", job_id, len(sampled), source_language)
            if source_language == "ru":
                log.info("job=%s translating Russian PDF text", job_id)
                translated_text = await asyncio.to_thread(
                    self.translator.translate, sampled, cancel_event,
                )
                translated = True
                log.info("job=%s translated to %d chars", job_id, len(translated_text))
                analysis_input = translated_text.encode("utf-8")
                analysis_suffix = ".txt"
            else:
                analysis_input = content_bytes
                analysis_suffix = ".pdf"
        elif fmt == "txt":
            text = _decode_txt(content_bytes)
            log.info("job=%s decoded %d chars from txt", job_id, len(text))
            source_language = detect_language(text)
            if source_language == "ru":
                log.info("job=%s detected Russian source, translating to English", job_id)
                text = await asyncio.to_thread(self.translator.translate, text, cancel_event)
                translated = True
                log.info("job=%s translated to %d chars", job_id, len(text))
            analysis_input = text.encode("utf-8")
            analysis_suffix = ".txt"
        else:
            raise ValueError(f"unsupported format: {fmt}")

        result = await asyncio.to_thread(
            self._run_analysis,
            analysis_input,
            analysis_suffix,
            title,
            use_llm,
        )

        # Если детектор не нашёл главы в текстовом входе, добавляем синтетический
        # заголовок и запускаем анализ повторно, чтобы пользователь получил результат
        if (
            fmt != "pdf"
            and result.get("num_chapters", 0) == 0
            and isinstance(analysis_input, bytes)
            and analysis_input.strip()
        ):
            log.info("job=%s no chapters detected, retrying as single chapter", job_id)
            stripped = analysis_input.decode("utf-8", errors="replace").strip()
            fallback = f"Chapter 1. Full Text\n\n{stripped}".encode("utf-8")
            result = await asyncio.to_thread(
                self._run_analysis, fallback, ".txt", title, use_llm,
            )
            result["fallback_single_chapter"] = True

        if not use_paragraph_analysis:
            result.pop("blocks", None)
            result["num_blocks"] = 0

        for field in _LARGE_RESULT_FIELDS:
            result.pop(field, None)
        for ch in result.get("chapters", []):
            ch.pop("most_intense_scenes", None)

        # Отбрасываем слишком короткие главы, появившиеся из-за перевода или regex,
        # и перенумеровываем остальные, чтобы chapter_index оставался 1..N
        MIN_CHAPTER_CHARS = 100
        original_chapters = result.get("chapters", [])
        filtered = [
            ch for ch in original_chapters
            if (ch.get("chapter_char_count") or 0) >= MIN_CHAPTER_CHARS
        ]
        dropped = len(original_chapters) - len(filtered)
        if dropped:
            log.info("job=%s dropped %d short chapters (<%d chars)", job_id, dropped, MIN_CHAPTER_CHARS)
            for new_idx, ch in enumerate(filtered, start=1):
                ch["chapter_index"] = new_idx
            result["chapters"] = filtered
            result["num_chapters"] = len(filtered)

        result["source_language"] = source_language
        result["translated"] = translated
        result["use_llm"] = use_llm
        result["use_paragraph_analysis"] = use_paragraph_analysis

        result_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        log.info(
            "model produced %d chapters, %d scenes, %d bytes",
            result.get("num_chapters", 0),
            result.get("num_scenes", 0),
            len(result_bytes),
        )

        await self.ts_client.SaveAnalysisResult(
            ts_pb.SaveAnalysisResultRequest(
                text_id=job["text_id"],
                job_id=job_id,
                result_json=result_bytes,
            )
        )
        await self._update_status(job_id, "completed")
        log.info("job %s completed", job_id)

    def _run_analysis(
        self, content: bytes, suffix: str, title: str, use_llm: bool,
    ) -> dict:
        """Drive the new pipeline: write input to a temp dir, run the analysis,
        load the rich JSON the pipeline drops on disk, and clean up."""
        with tempfile.TemporaryDirectory(prefix="book-analyzer-") as tmp_root:
            tmp_root_path = Path(tmp_root)
            input_path = tmp_root_path / f"{_safe_filename(title)}{suffix}"
            output_dir = tmp_root_path / "out"
            input_path.write_bytes(content)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Настройки пайплайна живут в PipelineConfig и не передаются
            # напрямую в run_book_emotion_analysis
            pipeline_config = PipelineConfig(
                use_llm_scene_analysis=use_llm,
                save_intermediate_json=False,
                llm_provider=self.config.llm_provider,
                llm_model=self.config.llm_model,
                ollama_url=self.config.ollama_url,
                llm_timeout_seconds=self.config.llm_timeout_seconds,
            )

            run_book_emotion_analysis(
                input_file=input_path,
                output_dir=output_dir,
                config=pipeline_config,
                model_name=self.config.model_name_or_path,
            )

            result_path = output_dir / "chapter_emotions.json"
            return json.loads(result_path.read_text(encoding="utf-8"))

    async def _get_job(self, job_id: str) -> dict | None:
        raw = await self.rdb.get(f"job:{job_id}")
        if raw is None:
            return None
        return json.loads(raw)

    async def _update_status(self, job_id: str, status: str, error: str = "") -> None:
        job = await self._get_job(job_id)
        if job is None:
            return
        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        job["error"] = error
        await self.rdb.set(f"job:{job_id}", json.dumps(job))


def _safe_filename(name: str) -> str:
    """Strip filesystem-unfriendly characters so the temp file path is valid."""
    cleaned = re.sub(r"[^\w\-. ]+", "_", name).strip().strip(".")
    return cleaned or "input"
