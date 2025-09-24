# -*- coding: utf-8 -*-
"""
make_tracks_ego.py  (camera-shake detection 포함 버전)

YOLOv8 + ByteTrack/OCSORT로 차량 트래킹한 뒤:
- 각 트랙 요약(centroids/areas/feet, move, entry_side, first_ts/last_ts)
- 자차(블랙박스) 기준: 가장 위험한 차량(ego_collision) 선별 (TTC/접근성 + 전구간 최소접근 허들 0.04*W)
- 페어(Pairwise) 기준: 가장 근접한 차량쌍(primary_pair) + 충돌구간(collision) + 선후진입(entry_order)
- 카메라 셰이크(프레임간 글로벌 모션 스파이크) 검출(camera_shake)
  * robust median + MAD로 스파이크 임계(thr) 설정
  * 전구간 피크 목록과 전역 최댓값, ego_collision 근처의 t_shake를 ego_collision에 추가

실행 예시:
python make_tracks_ego.py --video_dir data/raw/videos/training_reencoded --out_dir data/tracks/raw/train
또는 단일 파일:
python make_tracks_ego.py --video_path data/raw/videos/...mp4 --out_json data/tracks/...json
"""

import os
import json
import math
import glob
import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# ----------------- 설정 -----------------
YOLO_MODEL  = "yolov8s.pt"       # 필요시 yolov8n.pt / yolov8m.pt 등
TRACKER     = "bytetrack.yaml"   # 또는 "ocsort.yaml"
VEHICLE_CLS = {1, 2, 3, 5, 7}    # COCO: bicycle, car, motorcycle, bus, truck

# 자차 기준점(C0) = 화면 하단 중앙 근처 (카메라 설치 가정)
EGO_X_RATIO = 0.5   # 가로 중앙
EGO_Y_RATIO = 0.85  # 하단 85% 지점

# TTC/접근성 하이퍼파라미터
FOOT_Y_THRESH = 0.70   # y/H 70% 이상 구간이 있어야(하단 근접)
TREND_WINDOW  = 6      # 기준점-바닥중점 거리 감소 추세 체크 윈도우(프레임)
SMOOTH_K      = 3      # s(t)=sqrt(area) 미분 평활 윈도우
TTC_MAX_SEC   = 5.0    # 현실적 TTC 상한(초)
MIN_SAMPLES   = 3      # 최소 샘플 수


# 페어 충돌 판단 임계 (두 중심점 최소 거리 < 화면폭 4%)
PAIR_COLLISION_W_RATIO = 0.04
PAIR_HALF_WINDOW_SEC   = 0.15  # 충돌구간 ±0.15초
POST_COLLISION_PAD_SEC = 1.0

SHAKE_PROMOTE_R_FACTOR = 2.0


# ---------- 카메라 셰이크 파라미터 ----------
SHAKE_MAX_CORNERS   = 800
SHAKE_QUALITY       = 0.01
SHAKE_MIN_DISTANCE  = 8
SHAKE_BLOCK_SIZE    = 7
SHAKE_MAD_K         = 3.2    # median + K * MAD
SHAKE_MIN_MAG_PX    = 1.8    # 절대 최소 허들(px)
SHAKE_NEAR_WIN_SEC  = 0.30   # ego_collision 주변에서 셰이크 피크를 찾는 윈도(±초)

NEAR_MISS_W_RATIO = 0.17   # near-miss 판정: 화면폭 12%
NEAR_MISS_FOOT_Y  = 0.55   # near-miss 시 바닥근접 허들
DEBUG_EGO_SELECT  = True

BOTTOM_DOMINANT_Y = 0.90   # 화면 하단 90% 이상까지 들어오면 '매우 근접'
BOTTOM_DOMINANT_P = 90     # y 퍼센타일(예: 90퍼센타일)
DEBUG_EGO_SELECT = True    # 디버그

SHAKE_STRICT_RATIO  = 1.5

# ----------------- 유틸 -----------------
# 파일 상단 근처에 추가
def to_native(x):
    if isinstance(x, dict):
        return {k: to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    # numpy 스칼라
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    # numpy 배열
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def angle_of(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))

def classify_turn(angles_deg):
    if len(angles_deg) < 5:
        return "straight"
    ang = np.unwrap(np.radians(angles_deg))
    d = np.diff(np.degrees(ang))
    rot = float(np.sum(d))
    if abs(rot) < 15:
        return "straight"
    return "left_turn" if rot > 0 else "right_turn"

# --- 화면 비율 기반 ROI(하단 60%를 좌/우로 나눔; 필요하면 수치 조정) ---
RIGHT_ROI_R = (0.45, 1.00, 0.60, 1.00)  # ← 기존 0.62 대신 0.45로
LEFT_ROI_R  = (0.00, 0.38, 0.60, 1.00)

from shapely.geometry import Point, Polygon

def _rect_poly(r, W, H):
    x1, x2 = int(r[0]*W), int(r[1]*W)
    y1, y2 = int(r[2]*H), int(r[3]*H)
    return Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])

def entry_side_roi(centroids, feet, W, H, k=8, tol=0.33):
    """
    - 초반 k프레임의 발(feet) 좌표가 있으면 그걸, 없으면 센트로이드로 ROI 판정
    - ROI에 안 걸리면 x-기반 룰로 fallback
    """
    right_roi = _rect_poly(RIGHT_ROI_R, W, H)
    left_roi  = _rect_poly(LEFT_ROI_R,  W, H)

    use = feet if (feet is not None and len(feet) > 0) else centroids
    pts = [Point(float(x), float(y)) for (x, y) in use[:k]]

    if any(right_roi.contains(p) for p in pts):
        return "side_right"
    if any(left_roi.contains(p) for p in pts):
        return "side_left"
    return entry_side(centroids[0][0], W, tol)  # fallback

def entry_side(first_x, W, tol=0.4):
    x = first_x / float(W)
    if x < tol:
        return "side_left"
    if x > (1.0 - tol):
        return "side_right"
    return "main"  # 중앙

def ttc_from_areas(areas, fps, k=SMOOTH_K):
    """
    area(t)로부터 s = sqrt(area), ds/dt > 0 이면 접근으로 해석.
    TTC ≈ s / max(ds/dt, eps)
    returns (ttc_array, offset)
    """
    a = np.asarray(areas, dtype=np.float32)
    n = a.size
    if n < 2:
        return None, None

    s = np.sqrt(a)
    ds = np.diff(s) * float(fps)

    offset = 1  # diff로 한 스텝 앞당겨짐
    if ds.size <= 0:
        return None, None

    k_eff = int(k) if (k is not None) else 1
    k_eff = max(1, min(k_eff, ds.size))

    if k_eff > 1:
        ker = np.ones(k_eff, dtype=np.float32) / float(k_eff)
        ds = np.convolve(ds, ker, mode="valid")
        s = s[1:1 + ds.size]
    else:
        s = s[1:]  # len(s) == len(ds)

    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = s / np.maximum(ds, eps)

    return ttc, offset

def first_enter_ts(track, W, H, fps, roi=(0.2, 0.8, 0.2, 0.8)):
    """
    중앙 ROI(기본: 가로 20~80%, 세로 20~80%)에 최초로 진입한 시각(초).
    못 찾으면 first_ts 반환(보수).
    """
    C = np.asarray(track.get("centroids", []), dtype=np.float32)
    frs = np.asarray(track.get("frames", []), dtype=np.int32)
    if C.size == 0 or frs.size == 0:
        return float(track.get("first_ts", 0.0))
    xs, ys = C[:, 0], C[:, 1]
    lx, rx = W * roi[0], W * roi[1]
    ty, by = H * roi[2], H * roi[3]
    for k, (x, y) in enumerate(zip(xs, ys)):
        if lx <= x <= rx and ty <= y <= by:
            return float(frs[k] / fps)
    return float(track.get("first_ts", 0.0))

def pair_collision_aligned(a, b, W, fps):
    """
    두 트랙의 공통 프레임에 맞춰 중심점 거리 시퀀스를 정렬한 뒤,
    최소 거리 시점을 찾고, 임계(화면폭 * 0.04) 아래면 충돌/근접으로 본다.
    """
    fa = np.asarray(a.get("frames", []), dtype=np.int32)
    fb = np.asarray(b.get("frames", []), dtype=np.int32)
    Ca = np.asarray(a.get("centroids", []), dtype=np.float32)
    Cb = np.asarray(b.get("centroids", []), dtype=np.float32)
    if fa.size == 0 or fb.size == 0 or Ca.size == 0 or Cb.size == 0:
        return None

    common = sorted(set(fa).intersection(set(fb)))
    if len(common) < 3:
        return None

    ia = np.searchsorted(fa, common)
    ib = np.searchsorted(fb, common)
    A = Ca[ia]
    B = Cb[ib]
    D = np.linalg.norm(A - B, axis=1)
    k = int(np.argmin(D))

    thr = PAIR_COLLISION_W_RATIO * W
    if D[k] >= thr:
        return None
    

    t = float(common[k] / fps)
    w = int(round(PAIR_HALF_WINDOW_SEC * fps))
    k1 = max(0, k - w)
    k2 = min(len(common) - 1, k + w)
    return {"t1": float(common[k1] / fps), "t2": float(common[k2] / fps), "dmin": float(D[k])}

# ---------- 카메라 셰이크(글로벌 모션 스파이크) ----------
def _estimate_motion(prev_gray, gray):
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=SHAKE_MAX_CORNERS,
        qualityLevel=SHAKE_QUALITY,
        minDistance=SHAKE_MIN_DISTANCE,
        blockSize=SHAKE_BLOCK_SIZE,
        useHarrisDetector=False
    )
    if pts is None or len(pts) < 8:
        return 0.0, 0.0  # 잡을 점이 없으면 0

    nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None or st is None:
        return 0.0, 0.0

    good_prev = pts[st[:, 0] == 1]
    good_next = nxt[st[:, 0] == 1]
    if len(good_prev) < 6:
        return 0.0, 0.0

    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        # RANSAC 실패 시, 중위수 이동량으로 근사
        flow = good_next - good_prev
        dx = float(np.median(flow[:, 0, 0]))
        dy = float(np.median(flow[:, 0, 1]))
        return dx, dy

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    return dx, dy

def detect_camera_shake(video_path, fps):
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return {"series": [], "peaks": [], "peak": None, "thr": None}

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    series = []  # [{"t":..., "mag":..., "dx":..., "dy":...}, ...]

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy = _estimate_motion(prev_gray, g)
        mag = float(np.hypot(dx, dy))
        series.append({"t": float(idx / (fps or 30.0)), "mag": mag, "dx": dx, "dy": dy})
        prev_gray = g
        idx += 1
    cap.release()

    if not series:
        return {"series": [], "peaks": [], "peak": None, "thr": None}

    mags = np.array([s["mag"] for s in series], dtype=np.float32)
    med = float(np.median(mags))
    mad = float(np.median(np.abs(mags - med)) + 1e-6)
    thr = max(med + SHAKE_MAD_K * mad, SHAKE_MIN_MAG_PX)

    # 간단한 로컬 피크 찾기
    peaks = []
    for i in range(1, len(mags) - 1):
        if (
            mags[i] > thr
            and mags[i] >= mags[i - 1]
            and mags[i] >= mags[i + 1]
            and mags[i] >= thr * SHAKE_STRICT_RATIO   # ← 추가: 충분히 큰 피크만
        ):
            peaks.append({"t": series[i]["t"], "mag": float(mags[i])})

    # 전역 최대
    imax = int(np.argmax(mags))
    peak = {"t": series[imax]["t"], "mag": float(mags[imax])}

    return {
        "series": series,  # 원하면 나중에 downsample해서 저장해도 됨
        "peaks": peaks,
        "peak": peak,
        "thr": {"value": thr, "median": med, "mad": mad}
    }

# ----------------- 핵심 처리 -----------------
def run_one(video_path, out_json):
    model = YOLO(YOLO_MODEL)
    gen = model.track(source=video_path, tracker=TRACKER, stream=True, verbose=False)

    tracks = {}  # tid -> {cls, frames, centroids, areas, feet}
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
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = (x2 - x1), (y2 - y1)
            area = max(1.0, float(w * h))
            foot = (float((x1 + x2) / 2.0), float(y2))  # bbox 바닥중점

            if tid not in tracks:
                tracks[tid] = {"cls": c, "frames": [], "centroids": [], "areas": [], "feet": []}
            tracks[tid]["frames"].append(int(fidx))
            tracks[tid]["centroids"].append([float(cx), float(cy)])
            tracks[tid]["areas"].append(area)
            tracks[tid]["feet"].append([foot[0], foot[1]])
        fidx += 1

    if fps is None:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    # 트랙 요약
    out_tracks = []
    for tid, d in tracks.items():
        fr = np.array(d["frames"], dtype=np.int32)
        C = np.array(d["centroids"], dtype=np.float32)
        if len(C) < MIN_SAMPLES:
            continue
        v = np.diff(C, axis=0) * float(fps)  # px/s
        angles = [angle_of(v_i) for v_i in v]
        move = classify_turn(angles)
        # side = entry_side_roi(C[0, 0], W, H)
        # 트랙 요약 루프 내부
        C = np.array(d["centroids"], dtype=np.float32)
        F = np.array(d["feet"], dtype=np.float32) if d.get("feet") else None

        # 잘못: side = entry_side_roi(C[0, 0], W, H)
        side = entry_side_roi(C, F, W, H, k=16, tol=0.4)
        first_ts = float(fr[0] / fps)
        last_ts = float(fr[-1] / fps)
        out_tracks.append({
            "id": int(tid),
            "cls": int(d["cls"]),
            "frames": d["frames"],
            "centroids": d["centroids"],
            "areas": d["areas"],
            "feet": d["feet"],
            "move": move,
            "entry_side": side,
            "first_ts": first_ts,
            "last_ts": last_ts
        })

    # -------- 자차 기준(ego) 위험 차 한 대 선택 --------
    def select_primary_other(tracks_list, W, H, fps, cam_shake=None):
        if not tracks_list:
            return None
        C0 = np.array([W * EGO_X_RATIO, H * EGO_Y_RATIO], dtype=np.float32)
        best = None  # (score, tid, t_at, r_min_px, method, extra)
        best_fb = None
        thr_pair = PAIR_COLLISION_W_RATIO * W
        near_thr = NEAR_MISS_W_RATIO * W

        # 쉐이크 피크 시각/크기 목록
        peaks = cam_shake.get("peaks", []) if cam_shake else []
        peak_ts = np.array([p["t"] for p in peaks], dtype=np.float32) if peaks else None

        for tr in tracks_list:
            feet = np.array(tr.get("feet", []), dtype=np.float32)
            frs  = np.array(tr.get("frames", []), dtype=np.int32)
            areas = np.array(tr.get("areas", []), dtype=np.float32)
            if len(feet) < MIN_SAMPLES or len(frs) != len(feet):
                continue

            # 기준점 거리 시퀀스
            dist = np.linalg.norm(feet - C0[None, :], axis=1)
            r_min = float(np.min(dist)); k_min = int(np.argmin(dist))
            t_min = float(frs[k_min] / fps)

            # 0) contact
            if r_min < thr_pair:
                cand = (r_min, int(tr["id"]), t_min, r_min, "contact_min", {})
                if (best is None) or (cand[0] < best[0]): best = cand
                continue

            # 1) near-miss
            y_ratio = feet[:, 1] / float(H)
            if (r_min < near_thr) and (np.percentile(y_ratio, 75) >= NEAR_MISS_FOOT_Y):
                cand = (r_min, int(tr["id"]), t_min, r_min, "near_miss", {})
                if (best is None) or (cand[0] < best[0]): best = cand

            # 2) TTC (기존 로직 그대로)
            if np.percentile(y_ratio, 75) >= FOOT_Y_THRESH and len(areas) == len(feet):
                ttc, off = ttc_from_areas(areas, fps, k=SMOOTH_K)
                if ttc is not None and off is not None:
                    ttc = np.asarray(ttc, dtype=np.float32)
                    mask = (ttc > 0.0) & (ttc <= float(TTC_MAX_SEC))
                    if np.any(mask):
                        idx_rel = int(np.argmin(ttc[mask]))
                        idx_seq = np.arange(len(ttc), dtype=np.int32)[mask][idx_rel] + off
                        idx_seq = int(min(idx_seq, len(frs) - 1))
                        t_at = float(frs[idx_seq] / fps)
                        ttc_min = float(ttc[mask][idx_rel])
                        score = float(ttc_min + 0.002 * r_min)
                        cand = (score, int(tr["id"]), t_at, r_min, "ttc_min", {})
                        if (best is None) or (cand[0] < best[0]): best = cand

            # 3) SHAKE: 피크 근처에 있으면 강하게 프로모션
            if peak_ts is not None and peak_ts.size > 0:
                # 트랙 각 프레임의 t와 쉐이크 피크의 최근접 시차
                t_seq = frs.astype(np.float32) / float(fps)
                for j, t in enumerate(t_seq):
                    # 가장 가까운 피크와의 시간차
                    k = int(np.argmin(np.abs(peak_ts - t)))
                    dt = float(abs(peak_ts[k] - t))
                    if dt <= SHAKE_NEAR_WIN_SEC:
                        r_here = float(np.linalg.norm(feet[j] - C0))
                        y_here = float(feet[j, 1] / float(H))
                        mag = float(peaks[k]["mag"])
                        # 쉐이크가 있으니 near-miss의 2배 거리까지도 허용, 또는 하단 매우 근접
                        if (r_here < near_thr * SHAKE_PROMOTE_R_FACTOR) or (y_here >= 0.90):
                            # 점수는 "작을수록" 좋으니 r - 작은 가중치*mag 로 보너스
                            score = r_here - 0.5 * mag
                            extra = {"t_shake": float(peak_ts[k]), "shake_mag": mag}
                            cand = (score, int(tr["id"]), t, r_here, "shake_peak", extra)
                            if (best is None) or (cand[0] < best[0]): best = cand
                        # 한 트랙에서 여러 피크를 다 보지 않아도 됨
                        # 가장 가까운 한 번만 확인하고 넘어가도 충분
                        # break  # 원하면 조기종료
            # fallback (가장 가까웠던 물체)
            fb = (r_min, int(tr["id"]), t_min, r_min, "closest_min", {})
            if (best_fb is None) or (fb[0] < best_fb[0]):
                best_fb = fb

        if best is not None:
            score, tid, t_at, r_min_px, method, extra = best
            out = {"other_id": tid, "t_at": t_at, "score": score, "r_min_px": r_min_px, "method": method}
            out.update(extra)  # t_shake / shake_mag 들어있으면 같이 반환
            return out

        if best_fb is not None:
            score, tid, t_at, r_min_px, method, _ = best_fb
            return {"other_id": tid, "t_at": t_at, "score": score, "r_min_px": r_min_px, "method": method}

        return None

    cam_shake = detect_camera_shake(video_path, fps)

    ego_col = select_primary_other(out_tracks, W, H, fps, cam_shake=cam_shake)

    # ---------- ego_collision이 있으면 셰이크 시각 주입 ----------
    if ego_col is not None and cam_shake and cam_shake.get("series"):
        t_at = float(ego_col.get("t_at", 0.0))
        peaks = cam_shake.get("peaks", [])
        t_shake = None
        mag_shake = None

        # 우선 근처(±SHAKE_NEAR_WIN_SEC) 피크
        near = [p for p in peaks if abs(p["t"] - t_at) <= SHAKE_NEAR_WIN_SEC]
        if near:
            j = int(np.argmin([abs(p["t"] - t_at) for p in near]))
            t_shake, mag_shake = near[j]["t"], near[j]["mag"]
        else:
            # 근처가 없으면 전역 피크
            peak = cam_shake.get("peak")
            if peak:
                t_shake, mag_shake = float(peak["t"]), float(peak["mag"])

        if t_shake is not None:
            ego_col["t_shake"] = float(t_shake)
            ego_col["shake_mag"] = float(mag_shake)
            ego_col["shake_near_collision"] = bool(abs(t_shake - t_at) <= SHAKE_NEAR_WIN_SEC)

    # ---------- 여기서부터 center_t 기준으로 collision/entry_order 계산 ----------
    ego_other_id = None
    primary_pair = None
    collision = None
    entry_order = None

    primary_pair = None
    collision = None
    entry_order = None
    ego_other_id = None

    if ego_col is not None:
        ego_other_id = int(ego_col["other_id"])
        primary_pair = [ego_other_id, ego_other_id]

        video_len_sec = float(fidx / fps) if fps and fps > 0 else None

        # 쉐이크 있으면 무조건 그 시점을 충돌 중심으로 사용
        center_t = float(ego_col.get("t_shake", ego_col.get("t_at", 0.0)))

        half = float(PAIR_HALF_WINDOW_SEC)            # 예: 0.15s
        t1 = max(0.0, center_t - half)
        t2_raw = center_t + POST_COLLISION_PAD_SEC    # 예: +1.0s
        t2 = min(t2_raw, video_len_sec) if video_len_sec is not None else t2_raw

        dmin = float(ego_col.get("r_min_px", 0.0))
        collision = {"t1": t1, "t2": t2, "dmin": dmin}

        # entry_order도 center_t 기준으로 비교
        tbid = {t["id"]: t for t in out_tracks}
        tr_other = tbid.get(ego_other_id)
        if tr_other is not None:
            t_other_enter = first_enter_ts(tr_other, W, H, fps)
            entry_order = (
                {"earlier_id": ego_other_id, "later_id": -1}
                if t_other_enter <= center_t else
                {"earlier_id": -1, "later_id": ego_other_id}
            )
        else:
            entry_order = None

        # 디버깅 로그
        src = "shake" if "t_shake" in ego_col else ego_col.get("method", "t_at")
        print(f"[EGO] collision center={center_t:.2f}s by {src}, window=({t1:.2f} ~ {t2:.2f}), dmin={dmin:.1f}")
    else:
        # ego_col이 없을 때의 fallback (필요시 유지)
        if out_tracks:
            s = sorted(
                out_tracks,
                key=lambda t: (
                    -np.percentile(np.array([f[1] for f in t.get("feet", [])] or [0.0]) / float(H), 75),
                    -np.max(t.get("areas", [0.0]))
                )
            )
            ego_other_id = int(s[0]["id"])
            primary_pair = [ego_other_id, ego_other_id]
            collision = None
            entry_order = None

            # 충돌 구간을 셰이크 중심으로 덮어쓰고 싶으면 아래 주석 해제
            # hh = float(PAIR_HALF_WINDOW_SEC)
            # collision = {"t1": max(0.0, t_shake - hh), "t2": t_shake + hh, "dmin": ego_col.get("r_min_px", 0.0)}

    # 저장(JSON에 ego_other_id도 넣어 명시)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "video": os.path.basename(video_path),
        "fps": float(fps),
        "W": int(W),
        "H": int(H),
        "tracks": out_tracks,
        "ego_other_id": ego_other_id,
        "ego_collision": ego_col,
        "primary_pair": primary_pair,
        "collision": collision,
        "entry_order": entry_order,
        "camera_shake": cam_shake  # {"series":..., "peaks":[...], "peak":..., "thr":...}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(to_native(payload), f, indent=2, ensure_ascii=False)

def run_dir(video_dir, out_dir):
    for name in os.listdir(video_dir):
        if os.path.splitext(name)[1].lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            continue
        video_path = os.path.join(video_dir, name)
        out_json = os.path.join(out_dir, os.path.splitext(name)[0] + ".tracks.json")
        print("→", name)
        run_one(video_path, out_json)

# ----------------- CLI -----------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Track vehicles + detect ego collision & camera shake, export JSON")
    ap.add_argument("--video_dir", help="Directory of videos")
    ap.add_argument("--out_dir", help="Directory to save .tracks.json")
    ap.add_argument("--video_path", help="(Optional) Single video path")
    ap.add_argument("--out_json", help="(Optional) Output JSON path for the single video")
    args = ap.parse_args()

    if args.video_path:
        if not args.out_json:
            base = os.path.splitext(os.path.basename(args.video_path))[0] + ".tracks.json"
            out_dir = args.out_dir or os.path.dirname(args.video_path)
            os.makedirs(out_dir, exist_ok=True)
            args.out_json = os.path.join(out_dir, base)
        print("→ single:", os.path.basename(args.video_path))
        run_one(args.video_path, args.out_json)
        print("saved:", args.out_json)
    else:
        assert args.video_dir and args.out_dir, "--video_dir와 --out_dir를 지정하세요 (또는 --video_path)"
        run_dir(args.video_dir, args.out_dir)
