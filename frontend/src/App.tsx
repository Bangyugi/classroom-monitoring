import React, { useEffect, useState } from "react";
import { Classroom, SeatStatus, ViolationLog } from "./types";
import { CalibrationModal } from "./components/CalibrationModal";

const STATUS_STYLE: Record<SeatStatus, string> = {
  EMPTY: "bg-slate-800/70 border-slate-700 text-slate-400",
  NORMAL: "bg-emerald-950/80 border-emerald-500 text-emerald-200",
  SUSPICIOUS: "bg-amber-950/90 border-amber-400 text-amber-200 shadow-lg shadow-amber-500/20",
  VIOLATION: "bg-rose-950 border-rose-500 text-rose-100 animate-pulse shadow-lg shadow-rose-600/40",
};

export const App: React.FC = () => {
  const [classrooms, setClassrooms] = useState<Classroom[]>([]);
  const [selectedRoom, setSelectedRoom] = useState<Classroom | null>(null);
  const [running, setRunning] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);
  const [seatStates, setSeatStates] = useState<Record<string, SeatStatus>>({});
  const [violations, setViolations] = useState<ViolationLog[]>([]);
  const [showCalib, setShowCalib] = useState<boolean>(false);
  const [previewImg, setPreviewImg] = useState<string | null>(null);
  const [streamRetry, setStreamRetry] = useState<number>(0);

  useEffect(() => {
    fetch("/api/classrooms")
      .then((r) => r.json())
      .then((data: Classroom[]) => {
        setClassrooms(data);
        if (data.length > 0) setSelectedRoom(data[0]);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedRoom) return;
    setSeatStates({});
    fetch(`/api/violations?classroom_id=${selectedRoom.id}`)
      .then((r) => r.json())
      .then(setViolations)
      .catch(() => {});

    let disposed = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connectWs = () => {
      if (disposed) return;
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${window.location.host}/ws/classrooms/${selectedRoom.id}`);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.event === "INITIAL_STATE" && msg.seat_states) {
          setSeatStates(msg.seat_states);
        } else if (msg.event === "SEAT_STATE_CHANGED" && msg.classroom_id === selectedRoom.id) {
          setSeatStates((prev) => ({ ...prev, [String(msg.seat_id)]: msg.status }));
          if (msg.status === "SUSPICIOUS" || msg.status === "VIOLATION") {
            fetch(`/api/violations?classroom_id=${selectedRoom.id}`)
              .then((r) => r.json())
              .then(setViolations)
              .catch(() => {});
          }
        }
      };
      ws.onclose = () => {
        if (!disposed) {
          reconnectTimer = setTimeout(connectWs, 1500);
        }
      };
    };

    connectWs();
    return () => {
      disposed = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, [selectedRoom?.id]);

  useEffect(() => {
    if (!selectedRoom) return;
    const timer = setInterval(() => {
      fetch("/api/monitoring/status")
        .then((r) => r.json())
        .then((st) => {
          const isCurrentRoom = Boolean(st.running && st.classroom_id === selectedRoom.id);
          setRunning(isCurrentRoom);
          setFps(isCurrentRoom ? st.fps || 0 : 0);
          if (isCurrentRoom && st.seat_states) {
            setSeatStates(st.seat_states);
          }
        })
        .catch(() => {});
    }, 1500);
    return () => clearInterval(timer);
  }, [selectedRoom?.id]);

  const toggleMonitoring = async () => {
    if (!selectedRoom) return;
    if (running) {
      await fetch("/api/monitoring/stop", { method: "POST" });
      setRunning(false);
    } else {
      await fetch("/api/monitoring/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ classroom_id: selectedRoom.id }),
      });
      setRunning(true);
    }
  };

  const updateRoomParams = async (patch: Partial<Classroom>) => {
    if (!selectedRoom) return;
    const res = await fetch(`/api/classrooms/${selectedRoom.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (res.ok) {
      const updated: Classroom = await res.json();
      setSelectedRoom(updated);
      setClassrooms((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <header className="border-b border-slate-800 bg-slate-900/80 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white">
            Hệ thống Giám sát & Cảnh báo Sử dụng Điện thoại trong Lớp học
          </h1>
          <p className="text-xs text-slate-400">
            Cascade AI Pipeline: YOLOv8n-pose (Mức 0–1) → Smart ROI Crop YOLOv8n (Mức 2)
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={selectedRoom?.id || ""}
            onChange={async (e) => {
              const found = classrooms.find((c) => c.id === Number(e.target.value));
              if (found) {
                if (running) {
                  await fetch("/api/monitoring/stop", { method: "POST" });
                  setRunning(false);
                }
                setSelectedRoom(found);
              }
            }}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm"
          >
            {classrooms.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} [{c.camera_profile}]
              </option>
            ))}
          </select>

          {selectedRoom && (
            <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-1.5 text-xs">
              <label className="text-slate-400">Giữ tư thế (s):</label>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="15"
                value={selectedRoom.suspicious_hold_sec}
                onChange={(e) =>
                  updateRoomParams({ suspicious_hold_sec: Number(e.target.value) })
                }
                className="w-14 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-center text-white"
              />
              <label className="text-slate-400 ml-1">Ngưỡng YOLO:</label>
              <input
                type="number"
                step="0.05"
                min="0.15"
                max="0.95"
                value={selectedRoom.phone_conf_thres}
                onChange={(e) =>
                  updateRoomParams({ phone_conf_thres: Number(e.target.value) })
                }
                className="w-14 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-center text-white"
              />
            </div>
          )}

          <button
            onClick={() => setShowCalib(true)}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 text-sm font-medium"
          >
            Hiệu chỉnh Bàn học
          </button>

          <button
            onClick={toggleMonitoring}
            className={`px-4 py-2 rounded-lg font-semibold text-sm ${
              running ? "bg-rose-600 hover:bg-rose-500" : "bg-emerald-600 hover:bg-emerald-500"
            }`}
          >
            {running ? "Dừng Giám sát" : "Bắt đầu Giám sát"}
          </button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 p-6">
        <section className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <span className="font-semibold text-sm text-slate-200">
              Luồng Camera Trực tiếp (Overlay AI Cascade)
            </span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-slate-800 text-emerald-400 font-mono">
              {running ? `LIVE • ${fps} FPS` : "STANDBY"}
            </span>
          </div>

          <div className="flex-1 bg-black rounded-lg overflow-hidden flex items-center justify-center min-h-[380px]">
            {running ? (
              <img
                src={`/api/stream?room=${selectedRoom?.id || 0}&r=${streamRetry}`}
                alt="Live AI Stream"
                onError={() => setTimeout(() => setStreamRetry((n) => n + 1), 1000)}
                className="w-full h-auto max-h-[540px] object-contain"
              />
            ) : (
              <div className="text-center text-slate-500 p-8">
                Bấm <b>"Bắt đầu Giám sát"</b> để khởi chạy luồng AI Cascade.
              </div>
            )}
          </div>
        </section>

        <section className="lg:col-span-5 flex flex-col gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold text-sm">Sơ đồ Chỗ ngồi Thời gian thực</h2>
              <div className="flex gap-2 text-xs">
                <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-600">
                  Bình thường
                </span>
                <span className="px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-500">
                  Nghi ngờ (Mức 1)
                </span>
                <span className="px-2 py-0.5 rounded bg-rose-950 text-rose-200 border border-rose-500">
                  Vi phạm (Mức 2)
                </span>
              </div>
            </div>

            {selectedRoom && (
              <div
                className="grid gap-3"
                style={{
                  gridTemplateColumns: `repeat(${selectedRoom.grid_cols}, minmax(0, 1fr))`,
                }}
              >
                {selectedRoom.seats.map((seat) => {
                  const st: SeatStatus =
                    (seat.id !== undefined && seatStates[String(seat.id)]) || "EMPTY";
                  return (
                    <div
                      key={seat.id || seat.label}
                      className={`p-3 rounded-xl border-2 transition-all ${STATUS_STYLE[st]}`}
                    >
                      <div className="text-xs opacity-75">{seat.label}</div>
                      <div className="text-sm font-bold mt-1">{st}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex-1 flex flex-col">
            <h2 className="font-semibold text-sm mb-3">Nhật ký Cảnh báo & Ảnh Bằng chứng</h2>
            <div className="space-y-2.5 overflow-y-auto max-h-72 pr-1">
              {violations.length === 0 ? (
                <div className="text-xs text-slate-500 py-8 text-center">
                  Chưa ghi nhận cảnh báo nào.
                </div>
              ) : (
                violations.map((v) => (
                  <div
                    key={v.id}
                    className="flex items-center justify-between p-2.5 rounded-lg bg-slate-800/70 border border-slate-700 text-xs"
                  >
                    <div>
                      <div className="font-semibold text-slate-100">
                        {v.seat_label} —{" "}
                        <span
                          className={
                            v.level === "VIOLATION" ? "text-rose-400" : "text-amber-400"
                          }
                        >
                          {v.level}
                        </span>
                      </div>
                      <div className="text-slate-400">
                        {new Date(v.created_at).toLocaleTimeString()}
                        {v.confidence ? ` • Độ tin cậy: ${(v.confidence * 100).toFixed(0)}%` : ""}
                      </div>
                    </div>
                    {v.snapshot_url && (
                      <button
                        onClick={() => setPreviewImg(v.snapshot_url)}
                        className="px-2.5 py-1 rounded bg-slate-700 hover:bg-slate-600 text-slate-200"
                      >
                        Xem ảnh
                      </button>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </section>
      </main>

      {showCalib && selectedRoom && (
        <CalibrationModal
          classroom={selectedRoom}
          onClose={() => setShowCalib(false)}
          onSaved={(updated) => {
            setSelectedRoom(updated);
            setClassrooms((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
          }}
        />
      )}

      {previewImg && (
        <div
          onClick={() => setPreviewImg(null)}
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-6"
        >
          <img
            src={previewImg}
            alt="Violation Evidence"
            className="max-w-4xl max-h-[85vh] rounded-xl border border-slate-700"
          />
        </div>
      )}
    </div>
  );
};
