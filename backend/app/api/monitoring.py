from __future__ import annotations

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from backend.app.schemas.payloads import StartMonitoringSchema
from backend.app.ai.worker import worker_instance

router = APIRouter(prefix="/api", tags=["monitoring"])


@router.post("/monitoring/start")
def start_monitoring(payload: StartMonitoringSchema):
    worker_instance.start(payload.classroom_id)
    return {"status": "started", "classroom_id": payload.classroom_id}


@router.post("/monitoring/stop")
def stop_monitoring():
    worker_instance.stop()
    return {"status": "stopped"}


@router.get("/monitoring/status")
def get_monitoring_status():
    return {
        "running": worker_instance.running,
        "classroom_id": worker_instance.classroom_id,
        "fps": worker_instance.fps,
        "seat_states": worker_instance.get_seat_states(),
    }


async def _mjpeg_generator(request: Request):
    try:
        while worker_instance.running:
            if await request.is_disconnected():
                break
            if worker_instance.latest_jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + worker_instance.latest_jpeg + b"\r\n"
                )
            await asyncio.sleep(0.06)
    except (asyncio.CancelledError, GeneratorExit):
        pass


@router.get("/stream")
async def stream_video(request: Request):
    return StreamingResponse(
        _mjpeg_generator(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

