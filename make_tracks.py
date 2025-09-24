# make_tracks.py
import os, json, math, numpy as np, cv2
from ultralytics import YOLO

YOLO_MODEL = "yolov8s.pt"     # 필요시 yolov8s.pt
TRACKER = "bytetrack.yaml"    # 또는 "ocsort.yaml"
VEHICLE_CLS = {1,2,3,5,7}     # bicycle, car, motorcycle, bus, truck

def angle_of(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))

def classify_turn(angles_deg):
    if len(angles_deg) < 5: return "straight"
    ang = np.unwrap(np.radians(angles_deg))
    d = np.diff(np.degrees(ang))
    rot = float(np.sum(d))
    if abs(rot) < 15: return "straight"
    return "left_turn" if rot > 0 else "right_turn"

def entry_side(first_x, W, tol=0.33):
    x = first_x / float(W)
    if x < tol: return "side_left"
    if x > (1.0 - tol): return "side_right"
    return "main"  # 중앙

def run_one(video_path, out_json):
    model = YOLO(YOLO_MODEL)
    gen = model.track(source=video_path, tracker=TRACKER, stream=True, verbose=False)
    tracks = {}  # tid -> {cls, frames, centroids}
    fps, W, H = None, None, None
    fidx = 0

    for res in gen:
        if fps is None:
            try:
                fps = res.speed.get('fps', None)
            except:
                fps = None
        if W is None or H is None:
            H, W = res.orig_shape

        if res.boxes is None or res.boxes.id is None:
            fidx += 1
            continue

        ids = res.boxes.id.int().cpu().tolist()
        cls = res.boxes.cls.int().cpu().tolist()
        xyxy = res.boxes.xyxy.cpu().numpy()

        for i, tid in enumerate(ids):
            c = int(cls[i])
            if c not in VEHICLE_CLS: 
                continue
            x1, y1, x2, y2 = xyxy[i]
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            if tid not in tracks:
                tracks[tid] = {"cls": c, "frames": [], "centroids": []}
            tracks[tid]["frames"].append(fidx)
            tracks[tid]["centroids"].append([float(cx), float(cy)])
        fidx += 1

    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    # 특징 추출
    out_tracks = []
    for tid, d in tracks.items():
        fr = np.array(d["frames"])
        C  = np.array(d["centroids"])
        if len(C) < 3: 
            continue
        v = np.diff(C, axis=0) * fps   # px/s
        angles = [angle_of(v_i) for v_i in v]
        move = classify_turn(angles)
        side = entry_side(C[0,0], W)
        first_ts = float(fr[0] / fps)
        last_ts  = float(fr[-1] / fps)
        out_tracks.append({
            "id": int(tid),
            "cls": int(d["cls"]),
            "frames": d["frames"],
            "centroids": d["centroids"],
            "move": move,                         # straight / left_turn / right_turn
            "entry_side": side,                   # side_left / side_right / main
            "first_ts": first_ts,
            "last_ts":  last_ts
        })

    # 충돌(근접) 대략 탐지: 거리 최소 시점
    def pair_collision(a, b):
        Ca, Cb = np.array(a["centroids"]), np.array(b["centroids"])
        L = min(len(Ca), len(Cb))
        if L < 3: return None
        D = np.linalg.norm(Ca[:L] - Cb[:L], axis=1)
        k = int(np.argmin(D))
        # 화면 폭 5% 이하 근접이면 충돌 근접으로 취급
        thr = 0.05 * W
        if D[k] < thr:
            t = float(k / fps)
            # 근방 5프레임 범위로 충돌 구간 추정
            w = int(round(0.15 * fps))
            return {"t1": max(0.0, t - w / fps), "t2": t + w / fps, "dmin": float(D[k])}
        return None

    primary_pair = None
    best = None
    for i in range(len(out_tracks)):
        for j in range(i+1, len(out_tracks)):
            col = pair_collision(out_tracks[i], out_tracks[j])
            if col and (best is None or col["dmin"] < best["dmin"]):
                best, primary_pair = col, (out_tracks[i]["id"], out_tracks[j]["id"])

    # 교차로 진입 순서(중앙 60% 영역 진입 첫 시각)
    def first_enter_ts(track):
        C = np.array(track["centroids"])
        xs, ys = C[:,0], C[:,1]
        lx, rx = W*0.2, W*0.8
        ty, by = H*0.2, H*0.8
        for k,(x,y) in enumerate(zip(xs,ys)):
            if lx <= x <= rx and ty <= y <= by:
                return float(track["frames"][k] / fps)
        return float(track["first_ts"])  # fallback

    entry_order = None
    if primary_pair:
        A = next(t for t in out_tracks if t["id"]==primary_pair[0])
        B = next(t for t in out_tracks if t["id"]==primary_pair[1])
        ta, tb = first_enter_ts(A), first_enter_ts(B)
        entry_order = {"earlier_id": int(A["id"] if ta<=tb else B["id"]),
                       "later_id":   int(B["id"] if ta<=tb else A["id"])}

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "video": os.path.basename(video_path),
            "fps": fps, "W": W, "H": H,
            "tracks": out_tracks,
            "primary_pair": primary_pair,
            "collision": best,                 # {"t1","t2","dmin"}
            "entry_order": entry_order         # {"earlier_id","later_id"}
        }, f, indent=2)

def run_dir(video_dir, out_dir):
    for name in os.listdir(video_dir):
        if os.path.splitext(name)[1].lower() not in {".mp4",".mov",".mkv",".avi",".webm"}:
            continue
        video_path = os.path.join(video_dir, name)
        out_json = os.path.join(out_dir, os.path.splitext(name)[0] + ".tracks.json")
        print("→", name)
        run_one(video_path, out_json)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    run_dir(args.video_dir, args.out_dir)
