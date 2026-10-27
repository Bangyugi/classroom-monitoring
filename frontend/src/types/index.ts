export type SeatStatus = "EMPTY" | "NORMAL" | "SUSPICIOUS" | "VIOLATION";

export interface SeatItem {
  id?: number;
  label: string;
  row_idx: number;
  col_idx: number;
  polygon_norm: number[][];
}

export interface Classroom {
  id: number;
  name: string;
  camera_source: string;
  camera_profile: "FRONT_VIEW" | "REAR_VIEW";
  grid_rows: number;
  grid_cols: number;
  frame_skip: number;
  suspicious_hold_sec: number;
  phone_conf_thres: number;
  seats: SeatItem[];
}

export interface ViolationLog {
  id: number;
  classroom_id: number;
  seat_id: number;
  seat_label: string;
  track_id: number | null;
  level: SeatStatus;
  confidence: number | null;
  snapshot_url: string | null;
  created_at: string;
}

