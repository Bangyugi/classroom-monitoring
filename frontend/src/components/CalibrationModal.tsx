import React, { useState, useRef } from "react";
import { Classroom, SeatItem } from "../types";

interface Props {
  classroom: Classroom;
  onClose: () => void;
  onSaved: (updated: Classroom) => void;
}

export const CalibrationModal: React.FC<Props> = ({ classroom, onClose, onSaved }) => {
  const [seats, setSeats] = useState<SeatItem[]>(
    classroom.seats.map((s) => ({ ...s, polygon_norm: [...s.polygon_norm] }))
  );
  const [activeIdx, setActiveIdx] = useState<number>(0);
  const [draftPoints, setDraftPoints] = useState<number[][]>([]);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const nx = Number(((e.clientX - rect.left) / rect.width).toFixed(4));
    const ny = Number(((e.clientY - rect.top) / rect.height).toFixed(4));
    const next = [...draftPoints, [nx, ny]];
    if (next.length === 4) {
      const updated = [...seats];
      updated[activeIdx] = { ...updated[activeIdx], polygon_norm: next };
      setSeats(updated);
      setDraftPoints([]);
    } else {
      setDraftPoints(next);
    }
  };

  const handleSave = async () => {
    const res = await fetch(`/api/classrooms/${classroom.id}/seats`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ seats }),
    });
    if (res.ok) {
      onSaved(await res.json());
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/75 z-50 flex items-center justify-center p-6">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-5xl w-full p-6 text-slate-100">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">
            Hiệu chỉnh Đa giác Bàn học — {classroom.name}
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            Đóng ✕
          </button>
        </div>

        <p className="text-sm text-slate-300 mb-3">
          Chọn một bàn ở danh sách bên phải, sau đó click 4 điểm trên khung hình camera để khoanh vùng bàn học đó.
        </p>

        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2 relative border border-slate-700 rounded-lg overflow-hidden bg-black">
            <img
              src={`/api/classrooms/${classroom.id}/snapshot?t=${Date.now()}`}
              alt="Camera Snapshot"
              className="w-full h-auto block select-none"
            />
            <svg
              ref={svgRef}
              onClick={handleSvgClick}
              viewBox="0 0 1000 1000"
              preserveAspectRatio="none"
              className="absolute inset-0 w-full h-full cursor-crosshair"
            >
              {seats.map((seat, idx) => {
                const pts = seat.polygon_norm
                  .map(([x, y]) => `${x * 1000},${y * 1000}`)
                  .join(" ");
                const isSelected = idx === activeIdx;
                return (
                  <g key={idx}>
                    <polygon
                      points={pts}
                      fill={isSelected ? "rgba(59,130,246,0.35)" : "rgba(16,185,129,0.2)"}
                      stroke={isSelected ? "#60a5fa" : "#10b981"}
                      strokeWidth="4"
                    />
                  </g>
                );
              })}
              {draftPoints.map(([x, y], i) => (
                <circle key={i} cx={x * 1000} cy={y * 1000} r="8" fill="#facc15" />
              ))}
            </svg>
          </div>

          <div className="flex flex-col justify-between">
            <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
              {seats.map((seat, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setActiveIdx(idx);
                    setDraftPoints([]);
                  }}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm ${idx === activeIdx
                      ? "bg-blue-600/30 border-blue-400 text-white"
                      : "bg-slate-800 border-slate-700 text-slate-300"
                    }`}
                >
                  <div className="font-semibold">{seat.label}</div>
                  <div className="text-xs text-slate-400">
                    Đỉnh đã lưu: {seat.polygon_norm.length} (Đang vẽ: {idx === activeIdx ? draftPoints.length : 0}/4)
                  </div>
                </button>
              ))}
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={handleSave}
                className="flex-1 bg-emerald-600 hover:bg-emerald-500 py-2 rounded-lg font-semibold"
              >
                Lưu Cấu hình Bàn
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

