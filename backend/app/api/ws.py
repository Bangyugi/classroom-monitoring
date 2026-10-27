from __future__ import annotations

import asyncio
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, classroom_id: int, websocket: WebSocket):
        await websocket.accept()
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        self.active_connections.setdefault(classroom_id, []).append(websocket)

    def disconnect(self, classroom_id: int, websocket: WebSocket):
        conns = self.active_connections.get(classroom_id, [])
        if websocket in conns:
            conns.remove(websocket)

    async def broadcast(self, classroom_id: int, payload: dict):
        conns = list(self.active_connections.get(classroom_id, []))
        for ws in conns:
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(classroom_id, ws)

    def broadcast_from_thread(self, classroom_id: int, payload: dict):
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(classroom_id, payload), self.loop)


ws_manager = ConnectionManager()

