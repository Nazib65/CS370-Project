"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(
        None,
        description="Reference WAV (relative to pipeline_data/speakers/) for "
        "voice cloning. Auto-resolved from the target language when omitted.",
    ),
    target_language: str = Query("es", description="Target language code."),
    per_speaker: bool = Query(
        True,
        description="If the translated segments carry diarization speaker "
        "labels, pick a different voice per speaker.",
    ),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).

    Voice selection:
      - ``speaker_wav`` overrides the default voice for the whole clip.
      - When *per_speaker* is true and the transcript has speaker labels,
        each speaker gets its own reference WAV resolved via
        ``resolve_speaker_wav()``.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = trans_dir / f"{title}.json"

    # Default voice — explicit override wins; otherwise auto-resolve a sensible
    # fallback under speakers/ for the target language.
    speakers_dir = settings.speakers_dir
    default_voice = speaker_wav or resolve_speaker_wav(speakers_dir, target_language)

    # Per-speaker voice map — only populated when the translation has speaker
    # labels (set by /api/diarize/{video_id}).
    speaker_voice_map: dict[str, str] = {}
    if per_speaker and source_path.exists():
        translated = json.loads(source_path.read_text())
        unique_speakers = {
            seg.get("speaker")
            for seg in translated.get("segments", [])
            if seg.get("speaker")
        }
        for spk in unique_speakers:
            speaker_voice_map[spk] = resolve_speaker_wav(
                speakers_dir, target_language, speaker_id=spk,
            )

    await _run_in_threadpool(
        None, svc.text_file_to_speech, str(source_path), str(audio_dir),
        alignment=alignment,
        speaker_voice_map=speaker_voice_map or None,
        default_speaker_wav=default_voice,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
        "default_voice": default_voice,
        "speaker_voice_map": speaker_voice_map,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
