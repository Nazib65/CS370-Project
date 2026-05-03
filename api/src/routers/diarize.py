"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    # Step 1 — Extract a 16 kHz mono PCM WAV that pyannote can ingest.
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source video missing — run /api/download/{video_id} first.",
        )

    audio_path = diar_dir / f"{title}.wav"
    if not audio_path.exists():
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            str(audio_path),
        ]
        proc = await asyncio.to_thread(
            subprocess.run, cmd,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg audio extraction failed: {proc.stderr[-400:]}",
            )

    # Step 2 — Run pyannote diarization on the extracted audio.
    diar_segments = await asyncio.to_thread(
        _alignment_service.diarize, str(audio_path)
    )

    # Step 3 — Collect unique speaker labels (deterministic order).
    speakers = sorted({s["speaker"] for s in diar_segments})

    # Step 4 — Persist the result so subsequent calls hit the cache.
    diar_path.write_text(json.dumps({
        "speakers": speakers,
        "segments": diar_segments,
    }))

    # Step 5 (Task 3 from the diarization notebook) — merge speaker labels
    # back into the cached transcription so downstream stages (translate, TTS)
    # see ``segment["speaker"]``. Skipped when the transcription is missing or
    # diarization returned nothing useful.
    if diar_segments:
        from foreign_whispers.diarization import assign_speakers
        transcript_path = settings.transcriptions_dir / f"{title}.json"
        if transcript_path.exists():
            transcript = json.loads(transcript_path.read_text())
            transcript["segments"] = assign_speakers(
                transcript.get("segments", []), diar_segments,
            )
            transcript_path.write_text(json.dumps(transcript))

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
    )
