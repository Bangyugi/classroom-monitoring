from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import STORAGE_DIR
from backend.app.core.database import Base, SessionLocal, engine
from backend.app.api import classrooms, monitoring, violations
from backend.app.api.ws import ws_manager
from backend.app.ai.worker import worker_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        classrooms.seed_default_classrooms(db)
    yield
    worker_instance.stop()


app = FastAPI(title="Classroom Phone Detection System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

app.include_router(classrooms.router)
app.include_router(monitoring.router)
app.include_router(violations.router)


@app.websocket("/ws/classrooms/{classroom_id}")
async def classroom_ws_endpoint(websocket: WebSocket, classroom_id: int):
    await ws_manager.connect(classroom_id, websocket)
    try:
        await websocket.send_json(
            {
                "event": "INITIAL_STATE",
                "classroom_id": classroom_id,
                "seat_states": worker_instance.get_seat_states(),
            }
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(classroom_id, websocket)

