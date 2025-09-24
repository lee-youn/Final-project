# ===== onepass_collision_highlight.py =====
# Single video -> YOLOv8 tracking (+camera shake) -> pick ego-collision time -> render short highlight MP4
# - Input : --video /path/to/video.mp4
# - Output: --out   /path/to/out_clip.mp4
# Notes:
#   * Vehicles: COCO {bicycle=1, car=2, motorcycle=3, bus=5, truck=7}
#   * Collision center time = camera shake peak near ego-other OR TTC / min-distance fallback
#   * Draws boxes/IDs, trails, ego anchor, min-distance lines, shake-blink, center frame border
#   * Example: python onepass_collision_highlight.py --video sample.mp4 --out sample_collision.mp4

import os, json, math, argparse, glob
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------- Config -----------------
YOLO_MODEL  = "yolov8s.pt"
TRACKER     = "bytetrack.yaml"
VEHICLE_CLS = {1, 2, 3, 5, 7}

EGO_X_RATIO = 0.5
EGO_Y_RATIO = 0.85

PAIR_COLLISION_W_RATIO = 0.04
NEAR_MISS_W_RATIO      = 0.17
NEAR_MISS_FOOT_Y       = 0.55

FOOT_Y_THRESH = 0.70
SMOOTH_K      = 3
TTC_MAX_SEC   = 5.0
MIN_SAMPLES   = 3

# camera shake
SHAKE_MAX_CORNERS  = 800
SHAKE_QUALITY      = 0.01
SHAKE_MIN_DISTANCE = 8
SHAKE_BLOCK_SIZE   = 7
SHAKE_MAD_K        = 3.2
SHAKE_MIN_MAG_PX   = 1.8
SHAKE_STRICT_RATIO = 1.5
SHAKE_NEAR_WIN_SEC = 0.30
SHAKE_PROMOTE_R_FACTOR = 2.0

# render
PRE_SEC  = 0.8
POST_SEC = 1.2
TRAIL    = 16
STABILIZE = True

# ----------------- Utils -----------------
def _color_for_id(tid: int):
    rng = np.random.default_rng(tid * 1234567)
    return tuple(int(x) for x in rng.integers(64, 255, size=3).tolist())  # (B,G,R)

def _draw_box(img, box, color, thick=2):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)

def _draw_trail(img, pts, color, tail=12):
    if len(pts) < 2: return
    start = max(0, len(pts)-tail)
    for i in range(start, len(pts)-1):
        p1 = (int(pts[i][0]),  int(pts[i][1]))
        p2 = (int(pts[i+1][0]),int(pts[i+1][1]))
        a  = 0.3 + 0.7*(i-start)/max(1,(len(pts)-1-start))
        c  = tuple(int((1-a)*32 + a*c) for c in color)
        cv2.line(img, p1, p2, c, 2, cv2.LINE_AA)

def _put_text(img, text, org, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

def angle_of(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))

def classify_turn(angles_deg):
    if len(angles_deg) < 5: return "straight"
    ang = np.unwrap(np.radians(angles_deg))
    d = np.diff(np.degrees(ang))
    rot = float(np.sum(d))
    if abs(rot) < 15: return "straight"
    return "left_turn" if rot > 0 else "right_turn"

def ttc_from_areas(areas, fps, k=SMOOTH_K):
    a = np.asarray(areas, dtype=np.float32)
    if a.size < 2: return None, None
    s = np.sqrt(a)
    ds = np.diff(s) * float(fps)
    if ds.size <= 0: return None, None
    k_eff = max(1, min(int(k), ds.size))
    if k_eff > 1:
        ker = np.ones(k_eff, dtype=np.float32) / float(k_eff)
        ds = np.convolve(ds, ker, mode="valid")
        s = s[1:1 + ds.size]
    else:
        s = s[1:]
    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = s / np.maximum(ds, eps)
    return ttc, 1

def detect_camera_shake(video_path, fps):
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return {"series": [], "peaks": [], "peak": None, "thr": None}
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    series = []
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # lk
        pts = cv2.goodFeaturesToTrack(prev_gray, SHAKE_MAX_CORNERS, SHAKE_QUALITY, SHAKE_MIN_DISTANCE, blockSize=SHAKE_BLOCK_SIZE)
        if pts is None or len(pts) < 8:
            series.append({"t": float(idx / (fps or 30.0)), "mag": 0.0, "dx": 0.0, "dy": 0.0})
            prev_gray = g; idx += 1; continue
        nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, g, pts, None)
        if nxt is None or st is None:
            series.append({"t": float(idx / (fps or 30.0)), "mag": 0.0, "dx": 0.0, "dy": 0.0})
            prev_gray = g; idx += 1; continue
        good_prev = pts[st[:,0]==1]; good_next = nxt[st[:,0]==1]
        if len(good_prev) < 6:
            series.append({"t": float(idx / (fps or 30.0)), "mag": 0.0, "dx": 0.0, "dy": 0.0})
            prev_gray = g; idx += 1; continue
        M, inl = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            flow = good_next - good_prev
            dx = float(np.median(flow[:,0,0])); dy = float(np.median(flow[:,0,1]))
        else:
            dx = float(M[0,2]); dy = float(M[1,2])
        mag = float(np.hypot(dx, dy))
        series.append({"t": float(idx / (fps or 30.0)), "mag": mag, "dx": dx, "dy": dy})
        prev_gray = g; idx += 1
    cap.release()
    if not series:
        return {"series": [], "peaks": [], "peak": None, "thr": None}
    mags = np.array([s["mag"] for s in series], dtype=np.float32)
    med = float(np.median(mags)); mad = float(np.median(np.abs(mags-med)) + 1e-6)
    thr = max(med + SHAKE_MAD_K * mad, SHAKE_MIN_MAG_PX)
    peaks = []
    for i in range(1, len(mags)-1):
        if (mags[i] > thr and mags[i] >= mags[i-1] and mags[i] >= mags[i+1] and mags[i] >= thr*SHAKE_STRICT_RATIO):
            peaks.append({"t": series[i]["t"], "mag": float(mags[i])})
    imax = int(np.argmax(mags))
    peak = {"t": series[imax]["t"], "mag": float(mags[imax])}
    return {"series": series, "peaks": peaks, "peak": peak, "thr": {"value": thr, "median": med, "mad": mad}}

# ----------------- Tracking + summarization -----------------
def track_video(video_path):
    model = YOLO(YOLO_MODEL)
    gen = model.track(source=video_path, tracker=TRACKER, stream=True, verbose=False)

    tracks = {}  # tid -> {cls, frames, centroids, areas, feet, boxes}
    fps, W, H = None, None, None
    fidx = 0

    for res in gen:
        if fps is None:
            try:
                fps = res.speed.get("fps", None)
            except Exception:
                fps = None
        if W is None or H is None:
            H, W = res.orig_shape  # (H, W)
        if res.boxes is None or res.boxes.id is None:
            fidx += 1; continue

        ids  = res.boxes.id.int().cpu().tolist()
        cls  = res.boxes.cls.int().cpu().tolist()
        xyxy = res.boxes.xyxy.cpu().numpy()

        for i, tid in enumerate(ids):
            c = int(cls[i])
            if c not in VEHICLE_CLS:
                continue
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
            w, h   = (x2 - x1), (y2 - y1)
            area   = max(1.0, float(w * h))
            foot   = (float((x1 + x2)/2.0), float(y2))
            if tid not in tracks:
                tracks[tid] = {"cls": c, "frames": [], "centroids": [], "areas": [], "feet": [], "boxes": []}
            tracks[tid]["frames"].append(int(fidx))
            tracks[tid]["centroids"].append([float(cx), float(cy)])
            tracks[tid]["areas"].append(area)
            tracks[tid]["feet"].append([foot[0], foot[1]])
            tracks[tid]["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
        fidx += 1

    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    return tracks, float(fps), int(W), int(H), int(fidx)

def summarize_tracks(tracks, W, H, fps):
    out_tracks = []
    for tid, d in tracks.items():
        fr = np.array(d["frames"], dtype=np.int32)
        C  = np.array(d["centroids"], dtype=np.float32)
        if len(C) < MIN_SAMPLES: continue
        v = np.diff(C, axis=0) * float(fps)
        angles = [angle_of(v_i) for v_i in v]
        move = classify_turn(angles)
        first_ts = float(fr[0] / fps)
        last_ts  = float(fr[-1] / fps)
        out_tracks.append({
            "id": int(tid),
            "cls": int(d["cls"]),
            "frames": d["frames"],
            "centroids": d["centroids"],
            "areas": d["areas"],
            "feet": d["feet"],
            "boxes": d["boxes"],
            "move": move,
            "first_ts": first_ts,
            "last_ts": last_ts
        })
    return out_tracks

def select_ego_collision(tracks_list, W, H, fps, cam_shake):
    if not tracks_list: return None
    C0 = np.array([W*EGO_X_RATIO, H*EGO_Y_RATIO], dtype=np.float32)
    best = None; best_fb = None
    thr_pair = PAIR_COLLISION_W_RATIO * W
    near_thr = NEAR_MISS_W_RATIO * W
    peaks = cam_shake.get("peaks", []) if cam_shake else []
    peak_ts = np.array([p["t"] for p in peaks], dtype=np.float32) if peaks else None

    for tr in tracks_list:
        feet = np.array(tr.get("feet", []), dtype=np.float32)
        frs  = np.array(tr.get("frames", []), dtype=np.int32)
        areas = np.array(tr.get("areas", []), dtype=np.float32)
        if len(feet) < MIN_SAMPLES or len(frs) != len(feet): continue
        dist = np.linalg.norm(feet - C0[None,:], axis=1)
        r_min = float(np.min(dist)); k_min = int(np.argmin(dist))
        t_min = float(frs[k_min] / fps)

        if r_min < thr_pair:
            cand = (r_min, int(tr["id"]), t_min, r_min, "contact_min", {})
            if (best is None) or (cand[0] < best[0]): best = cand
            continue

        y_ratio = feet[:,1] / float(H)
        if (r_min < near_thr) and (np.percentile(y_ratio, 75) >= NEAR_MISS_FOOT_Y):
            cand = (r_min, int(tr["id"]), t_min, r_min, "near_miss", {})
            if (best is None) or (cand[0] < best[0]): best = cand

        if np.percentile(y_ratio, 75) >= FOOT_Y_THRESH and len(areas) == len(feet):
            ttc, off = ttc_from_areas(areas, fps, k=SMOOTH_K)
            if ttc is not None and off is not None:
                ttc = np.asarray(ttc, dtype=np.float32)
                mask = (ttc > 0.0) & (ttc <= float(TTC_MAX_SEC))
                if np.any(mask):
                    idx_rel = int(np.argmin(ttc[mask]))
                    idx_seq = np.arange(len(ttc), dtype=np.int32)[mask][idx_rel] + off
                    idx_seq = int(min(idx_seq, len(frs)-1))
                    t_at = float(frs[idx_seq] / fps)
                    ttc_min = float(ttc[mask][idx_rel])
                    score = float(ttc_min + 0.002 * r_min)
                    cand = (score, int(tr["id"]), t_at, r_min, "ttc_min", {})
                    if (best is None) or (cand[0] < best[0]): best = cand

        if peak_ts is not None and peak_ts.size > 0:
            t_seq = frs.astype(np.float32) / float(fps)
            for j, t in enumerate(t_seq):
                k = int(np.argmin(np.abs(peak_ts - t)))
                dt = float(abs(peak_ts[k] - t))
                if dt <= SHAKE_NEAR_WIN_SEC:
                    r_here = float(np.linalg.norm(feet[j] - C0))
                    y_here = float(feet[j,1] / float(H))
                    mag = float(peaks[k]["mag"])
                    if (r_here < near_thr * SHAKE_PROMOTE_R_FACTOR) or (y_here >= 0.90):
                        score = r_here - 0.5 * mag
                        extra = {"t_shake": float(peak_ts[k]), "shake_mag": mag}
                        cand = (score, int(tr["id"]), float(t), r_here, "shake_peak", extra)
                        if (best is None) or (cand[0] < best[0]): best = cand
            # fallback candidate
        fb = (r_min, int(tr["id"]), t_min, r_min, "closest_min", {})
        if (best_fb is None) or (fb[0] < best_fb[0]): best_fb = fb

    if best is not None:
        score, tid, t_at, r_min_px, method, extra = best
        out = {"other_id": tid, "t_at": t_at, "score": score, "r_min_px": r_min_px, "method": method}
        out.update(extra)
        return out
    if best_fb is not None:
        score, tid, t_at, r_min_px, method, _ = best_fb
        return {"other_id": tid, "t_at": t_at, "score": score, "r_min_px": r_min_px, "method": method}
    return None

# ----------------- Render -----------------
def estimate_dxdy(prev, curr):
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(prev_g, 600, 0.01, 8)
    if pts is None: return 0.0, 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts, None)
    if nxt is None: return 0.0, 0.0
    good_prev = pts[st[:,0]==1]; good_next = nxt[st[:,0]==1]
    if len(good_prev) < 6: return 0.0, 0.0
    M, inl = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None: return 0.0, 0.0
    return float(M[0,2]), float(M[1,2])

def build_index(tracks):
    idx = {}
    for t in tracks:
        tid = int(t["id"])
        idx[tid] = {
            "frames": np.array(t.get("frames", []), dtype=np.int32),
            "boxes":  np.array(t.get("boxes",  []), dtype=np.float32),
            "feet":   np.array(t.get("feet",   []), dtype=np.float32),
            "cent":   np.array(t.get("centroids", []), dtype=np.float32),
        }
    return idx

def nearest_row(track, f):
    frames = track["frames"]
    if frames.size == 0: return None
    i = np.searchsorted(frames, f)
    if i < frames.size and frames[i] == f:
        return i
    return None

def render_clip(video_path, tracks_list, ego_col, cam_shake, out_path, pre=PRE_SEC, post=POST_SEC, trail=TRAIL, stabilize=STABILIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    if not ok: raise SystemExit("Failed to read video")
    H, W = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    center_t = float(ego_col.get("t_shake", ego_col.get("t_at", 0.0)))
    f1 = max(0, int(round((center_t - pre) * fps)))
    f2 = min(total-1, int(round((center_t + post) * fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    idx = build_index(tracks_list)
    focus_tid = int(ego_col["other_id"]) if "other_id" in ego_col else None
    draw_tids = [focus_tid] if focus_tid in idx else []
    for t in tracks_list:
        tid = int(t["id"])
        if tid not in draw_tids: draw_tids.append(tid)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f1)
    ok, prev = cap.read()
    if not ok:
        cap.release(); writer.release(); raise SystemExit("Seek failed")
    accum_dx, accum_dy = 0.0, 0.0
    C0 = (int(W*EGO_X_RATIO), int(H*EGO_Y_RATIO))
    peak_ts = [float(p["t"]) for p in (cam_shake.get("peaks", []) if cam_shake else [])]

    for f in range(f1, f2+1):
        frame = prev if f == f1 else cap.read()[1]
        if frame is None: break
        if stabilize and f > f1:
            dx, dy = estimate_dxdy(prev, frame)
            accum_dx += dx; accum_dy += dy
            M = np.float32([[1,0,-accum_dx],[0,1,-accum_dy]])
            frame = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        prev = frame.copy()
        canvas = frame.copy()

        ct_px = int(round(center_t * fps))
        if f == ct_px:
            cv2.rectangle(canvas, (0,0), (W-1,H-1), (0,255,255), 6)
        _put_text(canvas, f"t={f/fps:0.2f}s  center={center_t:0.2f}s", (12, 26), (255,255,0))
        if peak_ts:
            near = min(abs(f/fps - p) for p in peak_ts)
            if near <= 0.05:
                cv2.rectangle(canvas, (8,8), (W-9,H-9), (0,0,255), 4)
        cv2.circle(canvas, C0, 6, (0,255,255), -1)

        for tid in draw_tids:
            T = idx.get(tid); 
            if T is None: continue
            i = nearest_row(T, f)
            if i is None: continue
            color = _color_for_id(tid)
            if T["boxes"].size:
                _draw_box(canvas, T["boxes"][i].tolist(), color, 3)
            feet = T["feet"]; cent = T["cent"]
            if feet.size:
                hist = []
                for k in range(max(0,i-trail), i+1):
                    if k < feet.shape[0]: hist.append(feet[k].tolist())
                _draw_trail(canvas, hist, color, tail=trail)
                fx, fy = int(feet[i,0]), int(feet[i,1])
                cv2.circle(canvas, (fx,fy), 5, color, -1)
                cv2.line(canvas, C0, (fx,fy), color, 1, cv2.LINE_AA)
            _put_text(canvas, f"ID {tid}", (int(cent[i,0])+6, int(cent[i,1])-6), color)

        if focus_tid is not None and focus_tid in idx:
            T = idx[focus_tid]; i = nearest_row(T, f)
            if i is not None and T["boxes"].size:
                _draw_box(canvas, T["boxes"][i].tolist(), (0,255,255), 4)
                _put_text(canvas, "FOCUS", (12, H-16), (0,255,255))

        writer.write(canvas)

    cap.release(); writer.release()
    print(f"[OK] saved clip: {out_path}")

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="One-pass: track + pick collision + render highlight clip")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   required=True)
    args = ap.parse_args()

    video = args.video
    out   = args.out
    if not os.path.exists(video):
        raise SystemExit(f"Video not found: {video}")

    print("[1/3] Tracking…")
    tracks, fps, W, H, nframes = track_video(video)
    print(f"   fps={fps:.2f}, size={W}x{H}, frames={nframes}, tracks={len(tracks)}")

    print("[2/3] Camera shake…")
    cam_shake = detect_camera_shake(video, fps)
    n_peaks = len(cam_shake.get("peaks", []))
    print(f"   peaks={n_peaks}, peak_t={cam_shake.get('peak',{}).get('t', None)}")

    print("[3/3] Pick collision & render…")
    tracks_list = summarize_tracks(tracks, W, H, fps)
    ego_col = select_ego_collision(tracks_list, W, H, fps, cam_shake)
    if ego_col is None:
        # fallback: pick longest track
        if not tracks_list:
            raise SystemExit("No tracks found")
        tid = max(tracks_list, key=lambda t: len(t["frames"]))["id"]
        ego_col = {"other_id": int(tid), "t_at": float(tracks_list[0]["first_ts"])}

    print(f"   focus_id={ego_col.get('other_id')}  center_t={ego_col.get('t_shake', ego_col.get('t_at'))}")
    render_clip(video, tracks_list, ego_col, cam_shake, out)

if __name__ == "__main__":
    main()
