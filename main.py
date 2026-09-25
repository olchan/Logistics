#!/usr/bin/env python3
"""
main.py — CJ 미래기술챌린지 box sizing 추론 진입점
실행: python main.py --input {video_folder_path}
"""
import os
import sys
import json
import glob
import shutil
import argparse
import traceback
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RANSACRegressor

# ==============================================================================
# 경로 설정 (상대경로만 사용 — 하드코딩 금지 규정 준수)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR   = os.path.join(SCRIPT_DIR, 'checkpoints')

BOX_ONNX_PATH   = os.path.join(CKPT_DIR, 'yolo26l_v5_augmentation.onnx')
RAIL_ONNX_PATH  = os.path.join(CKPT_DIR, 'rail_seg_yolov8s_LabelStudio_best.onnx')
DEPTH_ONNX_PATH = os.path.join(CKPT_DIR, 'DA3METRIC-LARGE.onnx')
REG_ONNX_PATH   = os.path.join(CKPT_DIR, 'box_regressor_final.onnx')

# ★ 단일 경로 대신 5-fold 앙상블 경로 리스트
N_ENSEMBLE = 5
REG_ONNX_PATHS = [os.path.join(CKPT_DIR, f'box_regressor_fold{i}.onnx') for i in range(N_ENSEMBLE)]

NORM_STATS_PATH = os.path.join(CKPT_DIR, 'norm_stats.json')

WORK_DIR = os.path.join(SCRIPT_DIR, '_work')
os.makedirs(WORK_DIR, exist_ok=True)

with open(NORM_STATS_PATH) as f:
    NS = json.load(f)

DEPTH_MEAN = NS['DEPTH_MEAN']; DEPTH_STD = NS['DEPTH_STD']
SC_MEAN    = np.array(NS['SC_MEAN'], np.float32)
SC_STD     = np.array(NS['SC_STD'], np.float32)
R_MEAN     = np.array(NS['R_MEAN'], np.float32)
R_STD      = np.array(NS['R_STD'], np.float32)
R_CLIP_LO  = NS['R_CLIP_LO']; R_CLIP_HI = NS['R_CLIP_HI']
SC_FALLBACK_K = NS['SC_FALLBACK_K']
FOCAL_MEAN = NS['FOCAL_MEAN']
F_COEF     = NS['F_COEF']; F_INT = NS['F_INT']
SENSOR_W   = NS['SENSOR_W']
RAIL_WIDTH_CM = NS['RAIL_WIDTH_CM']
CROP_SIZE  = NS['CROP_SIZE']

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# ==============================================================================
# [1] 상자 검출 하이퍼파라미터 + 세션
# ==============================================================================
MOTION_THRESHOLD    = 44
MOTION_RATIO_THRESH = 0.0294
ROI_PAD             = 20

BASE_MAX_W_RATIO     = 0.70
BASE_MAX_H_RATIO     = 0.70
RELAXED_MAX_W_RATIO  = 0.90
RELAXED_MAX_H_RATIO  = 0.90
BASE_MIN_CONF     = 0.6
RELAXED_MIN_CONF  = 0.85
DUPLICATE_OVERLAP_THRESH = 0.90

LABEL_INTERVAL_SEC  = 0.1
CONF_THRESH         = 0.25
IOU_THRESH          = 0.45

BT_HIGH_THRESH  = 0.50
BT_LOW_THRESH   = 0.25
BT_IOU_THRESH_1 = 0.30
BT_IOU_THRESH_2 = 0.20
BT_MAX_MISS     = 45

MERGE_IOU_THRESH    = 0.20
MERGE_MAX_FRAME_GAP = 30
POST_NMS_THRESH = 0.60

LANE_TOLERANCE_RATIO = 0.6
DEPTH_DIST_RATIO     = 1.5
FORWARD_ONLY_TOLERANCE_RATIO = 0.3

PARTIAL_AREA_RATIO         = 0.5
EDGE_TOLERANCE_RATIO       = 0.35
CROSS_AXIS_TOLERANCE_RATIO = 0.7
MIN_TRACK_FRAMES_FOR_MERGE = 2
MIN_TRACK_FRAMES_FOR_COUNT = 2

GLOBAL_MOTION_THRESH = MOTION_THRESHOLD
GLOBAL_RATIO_THRESH  = MOTION_RATIO_THRESH

print("🔍 상자 검출 세션 로드 중...")
box_session = ort.InferenceSession(BOX_ONNX_PATH, providers=providers)
box_input_name = box_session.get_inputs()[0].name
print(f"✅ box_detector 로드 완료  provider={box_session.get_providers()}")


def preprocess(img_bgr, imgsz=640):
    h, w  = img_bgr.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas   = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top  = (imgsz - nh) // 2
    pad_left = (imgsz - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]
    return tensor, scale, pad_top, pad_left


def postprocess(outputs, scale, pad_top, pad_left, conf_thresh=0.25, crop_h=640, crop_w=640):
    pred = outputs[0][0]
    kept = []
    for row in pred:
        x1, y1, x2, y2, conf, cls_id = row
        if conf < conf_thresh:
            continue
        x1 = (x1 - pad_left) / scale; y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale; y2 = (y2 - pad_top) / scale
        bx1, by1, bx2, by2 = map(int, [x1, y1, x2, y2])
        w, h = (bx2 - bx1), (by2 - by1)
        if w <= crop_w * BASE_MAX_W_RATIO and h <= crop_h * BASE_MAX_H_RATIO:
            if conf >= BASE_MIN_CONF:
                kept.append((bx1, by1, bx2, by2, float(conf), False))
        elif (w <= crop_w * RELAXED_MAX_W_RATIO and h <= crop_h * RELAXED_MAX_H_RATIO
              and conf >= RELAXED_MIN_CONF):
            kept.append((bx1, by1, bx2, by2, float(conf), True))
    return kept


def post_nms(dets, iou_thresh=POST_NMS_THRESH):
    if len(dets) == 0:
        return dets
    dets_sorted = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    while dets_sorted:
        best = dets_sorted.pop(0)
        keep.append(best)
        remaining = []
        for d in dets_sorted:
            ax1, ay1, ax2, ay2 = best[:4]
            bx1, by1, bx2, by2 = d[:4]
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            if inter == 0:
                remaining.append(d); continue
            ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            if inter / ua < iou_thresh:
                remaining.append(d)
        dets_sorted = remaining
    return keep


def ort_detect(crop_bgr, post_nms_thresh=POST_NMS_THRESH):
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    crop_h, crop_w = crop_bgr.shape[:2]
    tensor, scale, pad_top, pad_left = preprocess(crop_bgr)
    outputs = box_session.run(None, {box_input_name: tensor})
    dets = postprocess(outputs, scale, pad_top, pad_left, CONF_THRESH, crop_h, crop_w)
    return post_nms(dets, post_nms_thresh)


# ==============================================================================
# [2] ROI / 트래킹
# ==============================================================================
def extract_frames(video_path, interval_sec=LABEL_INTERVAL_SEC):
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * interval_sec)), 1)
    out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % step == 0:
            out.append((frame, idx / fps))
        idx += 1
    cap.release()
    return out, fps


def get_roi_for_test_video(video_path, motion_threshold=GLOBAL_MOTION_THRESH,
                            motion_ratio_thresh=GLOBAL_RATIO_THRESH):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_sample_frames = 20
    frame_step = max(1, min(int(fps * 0.5), max(total_f // min_sample_frames, 1)))
    extra = set(range(min(5, total_f))) | set(range(max(0, total_f - 5), total_f))
    sample_indices = np.array(sorted(set(np.arange(0, total_f, frame_step, dtype=int).tolist()) | extra))
    sampled_frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(sampled_frames) < 3:
        return None
    stack = np.stack(sampled_frames, axis=0).astype(np.float32)
    bg_median = np.median(stack, axis=0)
    motion_count = np.zeros((orig_h, orig_w), dtype=np.float32)
    for frame_gray in sampled_frames:
        diff = np.abs(frame_gray.astype(np.float32) - bg_median)
        motion_count += (diff > motion_threshold).astype(np.float32)
    motion_ratio = motion_count / len(sampled_frames)
    duration_sec = total_f / max(fps, 1)
    adaptive_thresh = motion_ratio_thresh
    if duration_sec < 5: adaptive_thresh = motion_ratio_thresh * 0.4
    elif duration_sec < 10: adaptive_thresh = motion_ratio_thresh * 0.7
    motion_mask = (motion_ratio >= adaptive_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, orig_w, orig_h)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    bx, by, bw, bh = cv2.boundingRect(contours[0])
    pad = 20
    return (max(0, bx-pad), max(0, by-pad), min(orig_w, bx+bw+pad), min(orig_h, by+bh+pad))


def estimate_conveyor_speed(all_tracks):
    vx_list, vy_list = [], []
    for tid, hist in all_tracks.items():
        if len(hist) < 3: continue
        frames = [h[0] for h in hist]
        bboxes = [h[1] for h in hist]
        for k in range(1, len(frames)):
            dt = frames[k] - frames[k-1]
            if dt <= 0 or dt > 5: continue
            cx_prev = (bboxes[k-1][0] + bboxes[k-1][2]) / 2
            cy_prev = (bboxes[k-1][1] + bboxes[k-1][3]) / 2
            cx_cur  = (bboxes[k][0]   + bboxes[k][2])   / 2
            cy_cur  = (bboxes[k][1]   + bboxes[k][3])   / 2
            vx_list.append((cx_cur - cx_prev) / dt)
            vy_list.append((cy_cur - cy_prev) / dt)
    if not vx_list:
        return 0.0, 0.0
    return float(np.median(vx_list)), float(np.median(vy_list))


def compute_dynamic_params(vx, vy, frame_interval_sec=LABEL_INTERVAL_SEC, bbox_avg_px=150):
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 1.0:
        return 30, 35
    frames_per_box = bbox_avg_px / speed
    max_miss  = int(np.clip(frames_per_box * 1.5, 5, BT_MAX_MISS))
    merge_gap = int(np.clip(frames_per_box * 2.0, 5, MERGE_MAX_FRAME_GAP))
    return int(max_miss), int(merge_gap)


def build_spatial_order(all_tracks, conveyor_vx=0.0, conveyor_vy=0.0):
    track_profiles = {}
    for tid, hist in all_tracks.items():
        if not hist: continue
        frames = [h[0] for h in hist]
        bboxes = [h[1] for h in hist]
        cx_vals = [(b[0]+b[2])/2 for b in bboxes]
        cy_vals = [(b[1]+b[3])/2 for b in bboxes]
        w_vals = [b[2]-b[0] for b in bboxes]
        h_vals = [b[3]-b[1] for b in bboxes]
        cx_rep = float(np.median(cx_vals))
        first_idx = int(np.argmin(frames))
        cy_entry = cy_vals[first_idx]
        track_profiles[tid] = {
            'cx': cx_rep, 'cy_entry': cy_entry,
            'first_frame': min(frames), 'last_frame': max(frames),
            'avg_w': float(np.median(w_vals)), 'avg_h': float(np.median(h_vals)),
        }
    return track_profiles


def is_same_lane(prof_a, prof_b, lane_tolerance_ratio=LANE_TOLERANCE_RATIO):
    avg_w = (prof_a['avg_w'] + prof_b['avg_w']) / 2
    cx_diff = abs(prof_a['cx'] - prof_b['cx'])
    return cx_diff < avg_w * lane_tolerance_ratio


def depth_order_consistent(earlier_prof, later_prof, conveyor_vy=0.0):
    if abs(conveyor_vy) < 0.5:
        return True
    cy_diff = later_prof['cy_entry'] - earlier_prof['cy_entry']
    if conveyor_vy > 0:
        tolerance = (earlier_prof['avg_h'] + later_prof['avg_h']) / 2
        return cy_diff > -tolerance
    else:
        tolerance = (earlier_prof['avg_h'] + later_prof['avg_h']) / 2
        return cy_diff < tolerance


class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, conf, init_vx=0.0, init_vy=0.0):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],
        ], dtype=float)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],
        ], dtype=float)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.kf.x[4, 0] = init_vx
        self.kf.x[5, 0] = init_vy
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1; self.miss = 0; self.conf = conf; self.history = []
        self.last_matched_bbox = tuple(map(int, bbox))

    def _bbox_to_z(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)
        ratio = (x2 - x1) / float(y2 - y1 + 1e-6)
        return np.array([[cx],[cy],[area],[ratio]], dtype=float)

    def _z_to_bbox(self):
        cx, cy, area, ratio = (self.kf.x[0,0], self.kf.x[1,0], self.kf.x[2,0], self.kf.x[3,0])
        area = max(area, 1.0); ratio = max(ratio, 0.1)
        w = np.sqrt(area * ratio); h = area / w
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])

    def predict(self):
        if self.kf.x[2,0] + self.kf.x[6,0] <= 0:
            self.kf.x[6,0] = 0.0
        self.kf.predict(); self.miss += 1
        return self._z_to_bbox()

    def update(self, bbox, conf, frame_idx):
        self.kf.update(self._bbox_to_z(bbox))
        self.hits += 1; self.miss = 0; self.conf = conf
        self.history.append((frame_idx, tuple(map(int, bbox)), conf))
        self.last_matched_bbox = tuple(map(int, bbox))

    def get_bbox(self):
        return self._z_to_bbox()


def _iou_matrix(trackers, detections):
    iou_mat = np.zeros((len(trackers), len(detections)), dtype=float)
    for i, t in enumerate(trackers):
        tb = t.get_bbox()
        for j, d in enumerate(detections):
            db = d[:4]
            xx1 = max(tb[0], db[0]); yy1 = max(tb[1], db[1])
            xx2 = min(tb[2], db[2]); yy2 = min(tb[3], db[3])
            inter = max(0, xx2-xx1) * max(0, yy2-yy1)
            if inter == 0: continue
            ua = ((tb[2]-tb[0])*(tb[3]-tb[1]) + (db[2]-db[0])*(db[3]-db[1]) - inter)
            iou_mat[i,j] = inter / ua
    return iou_mat


def _candidate_overlap_ratio(base_box, candidate_box):
    ax1, ay1, ax2, ay2 = base_box[:4]
    bx1, by1, bx2, by2 = candidate_box[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    candidate_area = (bx2-bx1) * (by2-by1)
    if candidate_area <= 0: return 0.0
    return inter / candidate_area


def _hungarian_match(iou_mat, thresh):
    if iou_mat.size == 0:
        return [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matched, unmatched_t, unmatched_d = [], [], []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r,c] >= thresh:
            matched.append((r,c)); matched_rows.add(r); matched_cols.add(c)
    for r in range(iou_mat.shape[0]):
        if r not in matched_rows: unmatched_t.append(r)
    for c in range(iou_mat.shape[1]):
        if c not in matched_cols: unmatched_d.append(c)
    return matched, unmatched_t, unmatched_d


def _lane_distance_matrix(trackers, detections, lane_tolerance_ratio=LANE_TOLERANCE_RATIO):
    lane_mat = np.full((len(trackers), len(detections)), np.inf)
    for i, t in enumerate(trackers):
        tb = t.get_bbox(); tcx = (tb[0] + tb[2]) / 2; t_w = tb[2] - tb[0]
        for j, d in enumerate(detections):
            dcx = (d[0] + d[2]) / 2; d_w = d[2] - d[0]
            avg_w = (t_w + d_w) / 2; cx_diff = abs(tcx - dcx)
            if cx_diff < avg_w * lane_tolerance_ratio:
                tcy = (tb[1] + tb[3]) / 2; dcy = (d[1] + d[3]) / 2
                lane_mat[i, j] = abs(tcy - dcy) / max(avg_w, 1)
    return lane_mat


def _apply_forward_only_mask(trackers, detections, vx, vy, tolerance_ratio=FORWARD_ONLY_TOLERANCE_RATIO):
    n_t, n_d = len(trackers), len(detections)
    mask = np.ones((n_t, n_d), dtype=bool)
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 0.5: return mask
    ux, uy = vx / speed, vy / speed
    for i, t in enumerate(trackers):
        lx1, ly1, lx2, ly2 = t.last_matched_bbox
        lcx, lcy = (lx1+lx2)/2, (ly1+ly2)/2
        lw, lh = (lx2-lx1), (ly2-ly1)
        for j, d in enumerate(detections):
            dx1, dy1, dx2, dy2 = d[:4]
            dcx, dcy = (dx1+dx2)/2, (dy1+dy2)/2
            dw, dh = (dx2-dx1), (dy2-dy1)
            disp_x, disp_y = dcx - lcx, dcy - lcy
            projection = disp_x * ux + disp_y * uy
            avg_size = (lw + lh + dw + dh) / 4
            tolerance = avg_size * tolerance_ratio
            if projection < -tolerance:
                mask[i, j] = False
    return mask


class ByteTracker:
    def __init__(self, high_thresh=BT_HIGH_THRESH, low_thresh=BT_LOW_THRESH,
                 iou_thresh_1=BT_IOU_THRESH_1, iou_thresh_2=BT_IOU_THRESH_2,
                 max_miss=BT_MAX_MISS, conveyor_vx=0.0, conveyor_vy=0.0,
                 pos_dist_thresh=DEPTH_DIST_RATIO, lane_tol_ratio=LANE_TOLERANCE_RATIO,
                 forward_only_tol=FORWARD_ONLY_TOLERANCE_RATIO,
                 duplicate_overlap_thresh=DUPLICATE_OVERLAP_THRESH):
        self.high_thresh = high_thresh; self.low_thresh = low_thresh
        self.iou_thresh_1 = iou_thresh_1; self.iou_thresh_2 = iou_thresh_2
        self.max_miss = max_miss; self.conveyor_vx = conveyor_vx; self.conveyor_vy = conveyor_vy
        self.pos_dist_thresh = pos_dist_thresh; self.lane_tol_ratio = lane_tol_ratio
        self.forward_only_tol = forward_only_tol
        self.duplicate_overlap_thresh = duplicate_overlap_thresh
        self.trackers = []; self.dead_trackers = []
        KalmanBoxTracker.count = 0

    def update(self, dets, frame_idx):
        base_dets = [d for d in dets if not (len(d) > 5 and d[5])]
        relaxed_dets = [d for d in dets if (len(d) > 5 and d[5])]
        filtered_relaxed = []
        for d in relaxed_dets:
            max_overlap = 0.0
            for b in base_dets:
                ov = _candidate_overlap_ratio(b[:4], d[:4])
                if ov > max_overlap: max_overlap = ov
            if max_overlap < self.duplicate_overlap_thresh:
                filtered_relaxed.append(d)
        dets = base_dets + filtered_relaxed
        high_dets = [d for d in dets if d[4] >= self.high_thresh]
        low_dets  = [d for d in dets if self.low_thresh <= d[4] < self.high_thresh]
        for t in self.trackers: t.predict()

        iou1 = _iou_matrix(self.trackers, high_dets)
        if len(self.trackers) > 0 and len(high_dets) > 0:
            fwd_mask1 = _apply_forward_only_mask(self.trackers, high_dets, self.conveyor_vx, self.conveyor_vy, self.forward_only_tol)
            iou1 = np.where(fwd_mask1, iou1, 0.0)
        matched1, unmatched_t1, unmatched_d1 = _hungarian_match(iou1, self.iou_thresh_1)
        for ti, di in matched1:
            self.trackers[ti].update(high_dets[di][:4], high_dets[di][4], frame_idx)

        remain2_t = [self.trackers[i] for i in unmatched_t1]
        remain2_idxs = list(unmatched_t1)
        iou2 = _iou_matrix(remain2_t, low_dets)
        if len(remain2_t) > 0 and len(low_dets) > 0:
            fwd_mask2 = _apply_forward_only_mask(remain2_t, low_dets, self.conveyor_vx, self.conveyor_vy, self.forward_only_tol)
            iou2 = np.where(fwd_mask2, iou2, 0.0)
        matched2, unmatched_t2_local, _ = _hungarian_match(iou2, self.iou_thresh_2)
        matched_t2_set = set(remain2_idxs[li] for li, _ in matched2)
        for li, di in matched2:
            remain2_t[li].update(low_dets[di][:4], low_dets[di][4], frame_idx)

        still_unmatched_t = [self.trackers[i] for i in unmatched_t1 if i not in matched_t2_set]
        still_unmatched_d = [high_dets[i] for i in unmatched_d1]

        if still_unmatched_t and still_unmatched_d:
            lane_mat = _lane_distance_matrix(still_unmatched_t, still_unmatched_d, self.lane_tol_ratio)
            fwd_mask3 = _apply_forward_only_mask(still_unmatched_t, still_unmatched_d, self.conveyor_vx, self.conveyor_vy, self.forward_only_tol)
            lane_mat = np.where(fwd_mask3, lane_mat, np.inf)
            matched3_local = []; used_t, used_d = set(), set()
            flat_indices = np.argsort(lane_mat.ravel())
            for idx in flat_indices:
                ti_l = idx // len(still_unmatched_d); di_l = idx % len(still_unmatched_d)
                if lane_mat[ti_l, di_l] == np.inf: break
                if lane_mat[ti_l, di_l] > self.pos_dist_thresh: break
                if ti_l in used_t or di_l in used_d: continue
                matched3_local.append((ti_l, di_l)); used_t.add(ti_l); used_d.add(di_l)
            for ti_l, di_l in matched3_local:
                d = still_unmatched_d[di_l]
                still_unmatched_t[ti_l].update(d[:4], d[4], frame_idx)
            matched3_dets = {id(still_unmatched_d[di_l]) for _, di_l in matched3_local}
            unmatched_d1_final = [di for di in unmatched_d1 if id(high_dets[di]) not in matched3_dets]
        else:
            unmatched_d1_final = list(unmatched_d1)

        for di in unmatched_d1_final:
            d = high_dets[di]
            new_t = KalmanBoxTracker(d[:4], d[4], init_vx=self.conveyor_vx, init_vy=self.conveyor_vy)
            new_t.history.append((frame_idx, tuple(map(int, d[:4])), d[4]))
            self.trackers.append(new_t)

        alive, dead = [], []
        for t in self.trackers:
            (alive if t.miss <= self.max_miss else dead).append(t)
        self.trackers = alive; self.dead_trackers.extend(dead)

        results = []
        for t in self.trackers:
            if t.miss > 0: continue
            bbox = t.get_bbox(); x1, y1, x2, y2 = map(int, bbox)
            results.append((t.id, x1, y1, x2, y2, t.conf))
        return results

    def get_all_tracks(self):
        all_t = self.trackers + self.dead_trackers
        return {t.id: t.history for t in all_t if t.history}


def merge_tracks(all_tracks, iou_thresh=MERGE_IOU_THRESH, max_frame_gap=MERGE_MAX_FRAME_GAP,
                  conveyor_vx=0.0, conveyor_vy=0.0, partial_area_ratio=PARTIAL_AREA_RATIO,
                  edge_tolerance_ratio=EDGE_TOLERANCE_RATIO,
                  cross_axis_tolerance_ratio=CROSS_AXIS_TOLERANCE_RATIO,
                  min_track_frames_for_merge=MIN_TRACK_FRAMES_FOR_MERGE):
    if len(all_tracks) <= 1:
        return all_tracks

    track_profiles = build_spatial_order(all_tracks, conveyor_vx, conveyor_vy)

    def bbox_area(b): return (b[2]-b[0]) * (b[3]-b[1])
    def bbox_iou(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1=max(ax1,bx1); iy1=max(ay1,by1); ix2=min(ax2,bx2); iy2=min(ay2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter == 0: return 0.0
        ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/ua
    def center_dist(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        return np.sqrt(((ax1+ax2)/2-(bx1+bx2)/2)**2 + ((ay1+ay2)/2-(by1+by2)/2)**2)
    def bbox_diag(b):
        x1,y1,x2,y2 = b
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    track_info = {}
    for tid, hist in all_tracks.items():
        if not hist: continue
        frames_ = [h[0] for h in hist]; bboxes = [h[1] for h in hist]
        areas = [bbox_area(b) for b in bboxes]
        track_info[tid] = {
            'first_frame': min(frames_), 'last_frame': max(frames_),
            'first_bbox': bboxes[0], 'last_bbox': bboxes[-1],
            'max_area': max(areas), 'frames': set(frames_), 'n_frames': len(hist),
        }

    parent = {tid: tid for tid in track_info}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y): parent[find(x)] = find(y)

    speed = np.sqrt(conveyor_vx**2 + conveyor_vy**2)
    is_horizontal = speed < 0.5 or abs(conveyor_vx) >= abs(conveyor_vy)
    tids = list(track_info.keys())

    for i in range(len(tids)):
        for j in range(i+1, len(tids)):
            ti, tj = tids[i], tids[j]
            info_i, info_j = track_info[ti], track_info[tj]
            if info_i['n_frames'] < min_track_frames_for_merge or info_j['n_frames'] < min_track_frames_for_merge:
                continue
            range_overlap = (info_i['first_frame'] < info_j['last_frame'] and info_j['first_frame'] < info_i['last_frame'])
            if range_overlap: continue
            if info_i['last_frame'] <= info_j['first_frame']:
                earlier, later = info_i, info_j; eid, lid = ti, tj
            elif info_j['last_frame'] <= info_i['first_frame']:
                earlier, later = info_j, info_i; eid, lid = tj, ti
            else:
                continue
            dt = later['first_frame'] - earlier['last_frame']
            if dt > max_frame_gap: continue

            is_partial = later['max_area'] < earlier['max_area'] * partial_area_ratio
            if is_partial:
                ex1,ey1,ex2,ey2 = earlier['last_bbox']
                ex1 += conveyor_vx*dt; ex2 += conveyor_vx*dt
                ey1 += conveyor_vy*dt; ey2 += conveyor_vy*dt
                lx1,ly1,lx2,ly2 = later['first_bbox']
                if is_horizontal:
                    e_w = ex2 - ex1
                    edge_tolerance_px = e_w * edge_tolerance_ratio
                    edge_gap = min(abs(ex1-lx1), abs(ex1-lx2), abs(ex2-lx1), abs(ex2-lx2))
                    cross_overlap = min(ey2,ly2) - max(ey1,ly1)
                    cross_tol = -max(ey2-ey1, ly2-ly1) * cross_axis_tolerance_ratio
                    ok = edge_gap <= edge_tolerance_px and cross_overlap > cross_tol
                else:
                    e_h = ey2 - ey1
                    edge_tolerance_px = e_h * edge_tolerance_ratio
                    edge_gap = min(abs(ey1-ly1), abs(ey1-ly2), abs(ey2-ly1), abs(ey2-ly2))
                    cross_overlap = min(ex2,lx2) - max(ex1,lx1)
                    cross_tol = -max(ex2-ex1, lx2-lx1) * cross_axis_tolerance_ratio
                    ok = edge_gap <= edge_tolerance_px and cross_overlap > cross_tol
                if ok: union(eid, lid)
                continue

            prof_e = track_profiles.get(eid); prof_l = track_profiles.get(lid)
            if prof_e and prof_l:
                if not is_same_lane(prof_e, prof_l): continue
                if not depth_order_consistent(prof_e, prof_l, conveyor_vy): continue

            extrap = (earlier['last_bbox'][0]+conveyor_vx*dt, earlier['last_bbox'][1]+conveyor_vy*dt,
                     earlier['last_bbox'][2]+conveyor_vx*dt, earlier['last_bbox'][3]+conveyor_vy*dt)
            first_bbox = later['first_bbox']
            iou = bbox_iou(extrap, first_bbox); dist = center_dist(extrap, first_bbox)
            diag = bbox_diag(earlier['last_bbox'])
            if iou >= iou_thresh or dist <= diag * 0.4:
                union(eid, lid)

    groups = defaultdict(list)
    for tid in tids: groups[find(tid)].append(tid)

    merged = {}
    for root, members in groups.items():
        combined = []
        for tid in members: combined.extend(all_tracks[tid])
        combined.sort(key=lambda x: x[0])
        rep_id = max(members, key=lambda m: track_info[m]['max_area'])
        merged[rep_id] = combined
    return merged


def run_two_pass_tracking(video_path, precomputed_roi=None, verbose=False,
                          min_track_frames_for_count=MIN_TRACK_FRAMES_FOR_COUNT):
    lv = os.path.join(WORK_DIR, os.path.basename(video_path))
    if not os.path.exists(lv):
        shutil.copy(video_path, lv)
    roi = precomputed_roi if precomputed_roi is not None else get_roi_for_test_video(lv)
    frames, fps = extract_frames(lv, LABEL_INTERVAL_SEC)

    KalmanBoxTracker.count = 0
    tracker1 = ByteTracker()
    all_dets = []
    for fi, (frame, t) in enumerate(frames):
        if roi is not None:
            rx1, ry1, rx2, ry2 = map(int, roi)
            crop = frame[ry1:ry2, rx1:rx2].copy()
        else:
            crop = frame.copy()
        if crop.size == 0:
            all_dets.append([]); continue
        dets = ort_detect(crop)
        all_dets.append(dets)
        tracker1.update(dets, fi)

    raw1 = tracker1.get_all_tracks()
    vx, vy = estimate_conveyor_speed(raw1)
    all_bboxes = [h[1] for hist in raw1.values() for h in hist]
    avg_w = float(np.mean([(b[2]-b[0]) for b in all_bboxes])) if all_bboxes else 150.0
    max_miss, merge_gap = compute_dynamic_params(vx, vy, LABEL_INTERVAL_SEC, avg_w)

    KalmanBoxTracker.count = 0
    tracker2 = ByteTracker(max_miss=max_miss, conveyor_vx=vx, conveyor_vy=vy)
    for fi, dets in enumerate(all_dets):
        tracker2.update(dets, fi)

    raw2 = tracker2.get_all_tracks()
    merged = merge_tracks(raw2, max_frame_gap=merge_gap, conveyor_vx=vx, conveyor_vy=vy)
    merged = {tid: hist for tid, hist in merged.items() if len(hist) >= min_track_frames_for_count}

    return merged, vx, vy, max_miss, merge_gap, all_dets, frames, roi


# ==============================================================================
# [3] 레일 검출 + f̂ 추정
# ==============================================================================
rail_session = ort.InferenceSession(RAIL_ONNX_PATH, providers=providers)
print(f"✅ rail_detector 로드 완료  provider={rail_session.get_providers()}")


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape[0]/h, new_shape[1]/w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    pad_top = (new_shape[0] - nh) // 2; pad_left = (new_shape[1] - nw) // 2
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = resized
    return canvas, scale, pad_left, pad_top


def preprocess_rail(img_bgr, imgsz=640):
    canvas, scale, pad_left, pad_top = letterbox(img_bgr, (imgsz, imgsz))
    tensor = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[np.newaxis]
    return tensor, scale, pad_left, pad_top


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms_rail(boxes, scores, iou_thresh=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]; keep.append(i)
        if len(idxs) == 1: break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0]); yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2]); yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_r = (boxes[rest,2]-boxes[rest,0]) * (boxes[rest,3]-boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        idxs = rest[iou < iou_thresh]
    return keep


def process_mask(protos, mask_coeffs, boxes_xyxy_640, orig_h, orig_w, scale, pad_left, pad_top, imgsz=640):
    c, mh, mw = protos.shape
    protos_flat = protos.reshape(c, -1)
    masks = mask_coeffs @ protos_flat
    masks = 1 / (1 + np.exp(-masks))
    masks = masks.reshape(-1, mh, mw)
    final_masks = []
    for i in range(masks.shape[0]):
        m = cv2.resize(masks[i], (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        nh = int(round(orig_h * scale)); nw = int(round(orig_w * scale))
        m = m[pad_top:pad_top+nh, pad_left:pad_left+nw]
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = boxes_xyxy_640[i]
        x1 = (x1 - pad_left) / scale; y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale; y2 = (y2 - pad_top) / scale
        x1, y1 = max(0,int(x1)), max(0,int(y1))
        x2, y2 = min(orig_w,int(x2)), min(orig_h,int(y2))
        mask_bin = np.zeros((orig_h, orig_w), dtype=np.uint8)
        mask_bin[y1:y2, x1:x2] = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8)
        final_masks.append(mask_bin)
    return final_masks


def run_yolov8_seg_onnx(session, img_bgr, conf_thresh=0.25, iou_thresh=0.45, imgsz=640):
    orig_h, orig_w = img_bgr.shape[:2]
    tensor, scale, pad_left, pad_top = preprocess_rail(img_bgr, imgsz)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    pred = outputs[0][0]; protos = outputs[1][0]; pred = pred.T
    nc = 1
    boxes_xywh = pred[:, :4]; class_scores = pred[:, 4:4+nc]; mask_coeffs = pred[:, 4+nc:]
    conf = class_scores[:, 0]
    mask_valid = conf > conf_thresh
    if not mask_valid.any():
        return []
    boxes_xywh = boxes_xywh[mask_valid]; conf = conf[mask_valid]; mask_coeffs = mask_coeffs[mask_valid]
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    keep = nms_rail(boxes_xyxy, conf, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]; conf = conf[keep]; mask_coeffs = mask_coeffs[keep]
    masks = process_mask(protos, mask_coeffs, boxes_xyxy, orig_h, orig_w, scale, pad_left, pad_top, imgsz)
    results = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        x1 = (x1 - pad_left) / scale; y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale; y2 = (y2 - pad_top) / scale
        results.append({'box': (int(x1), int(y1), int(x2), int(y2)), 'conf': float(conf[i]), 'mask': masks[i]})
    return results


def refine_rail_from_mask(mask, W, H, center_tol=0.35, ransac_residual=15, ratio_min=0.15, ratio_max=0.85):
    mask_bin = (mask > 0).astype(np.uint8)
    ys_all, xs_all = np.where(mask_bin > 0)
    if len(xs_all) == 0:
        return None, "empty"
    cx = xs_all.mean()
    if abs(cx - W/2) / W > center_tol:
        return None, "center"
    left_pts, right_pts = [], []
    y_min, y_max = ys_all.min(), ys_all.max()
    for y in range(y_min, y_max+1):
        xs = np.where(mask_bin[y] > 0)[0]
        if len(xs) < 3: continue
        left_pts.append([xs.min(), y]); right_pts.append([xs.max(), y])
    if len(left_pts) < 10 or len(right_pts) < 10:
        return None, "insufficient_points"

    def fit_line(pts):
        pts = np.array(pts); X = pts[:,1].reshape(-1,1); y = pts[:,0]
        ransac = RANSACRegressor(residual_threshold=ransac_residual, min_samples=10)
        ransac.fit(X, y)
        return ransac

    left_model = fit_line(left_pts); right_model = fit_line(right_pts)
    near_y = H - 1; far_y = y_min
    x_near_left = float(left_model.predict([[near_y]])[0])
    x_near_right = float(right_model.predict([[near_y]])[0])
    x_far_left = float(left_model.predict([[far_y]])[0])
    x_far_right = float(right_model.predict([[far_y]])[0])
    near_w = abs(x_near_right - x_near_left); far_w = abs(x_far_right - x_far_left)
    if far_w >= near_w:
        return None, "perspective_violation"
    ratio = far_w / (near_w + 1e-6)
    if not (ratio_min < ratio < ratio_max):
        return None, "ratio_out_of_range"
    return {
        'near_left': (x_near_left, near_y), 'near_right': (x_near_right, near_y),
        'far_left': (x_far_left, far_y), 'far_right': (x_far_right, far_y),
        'near_width_px': near_w, 'far_width_px': far_w,
    }, "OK"


def run_rail_inference_refined_onnx(session, img_bgr, conf_thresh=0.25):
    H, W = img_bgr.shape[:2]
    dets = run_yolov8_seg_onnx(session, img_bgr, conf_thresh=conf_thresh)
    if not dets:
        return None, "no_detection"
    best = max(dets, key=lambda d: d['conf'])
    return refine_rail_from_mask(best['mask'], W, H)


RAIL_N_SAMPLE = 30
RAIL_MIN_OK = 3

def estimate_rail_quad(frames, session=None, conf_thresh=0.25):
    if session is None:
        session = rail_session
    if not frames:
        return None
    H, W = frames[0][0].shape[:2]
    n = len(frames)
    idxs = np.unique(np.linspace(0, n-1, min(RAIL_N_SAMPLE, n)).astype(int))
    cols = {k: [] for k in ['nl', 'nr', 'fl', 'fr', 'fy']}
    for i in idxs:
        quad, reason = run_rail_inference_refined_onnx(session, frames[i][0], conf_thresh)
        if quad is None:
            continue
        cols['nl'].append(quad['near_left'][0]); cols['nr'].append(quad['near_right'][0])
        cols['fl'].append(quad['far_left'][0]); cols['fr'].append(quad['far_right'][0])
        cols['fy'].append(quad['far_left'][1])
    if len(cols['nl']) < RAIL_MIN_OK:
        return None
    q = {k: float(np.median(v)) for k, v in cols.items()}
    q['near_y'] = float(H - 1); q['W'] = W; q['H'] = H
    q['near_w'] = q['nr'] - q['nl']; q['far_w'] = q['fr'] - q['fl']
    q['far_y'] = q['fy']; q['n_ok'] = len(cols['nl'])
    return q


def rail_px_width_at_y(quad, y):
    t = (quad['near_y'] - y) / max(quad['near_y'] - quad['far_y'], 1e-6)
    t = float(np.clip(t, 0.0, 1.15))
    w = quad['near_w'] + (quad['far_w'] - quad['near_w']) * t
    return max(w, 1.0)


def rail_scale_at_y(quad, y):
    return RAIL_WIDTH_CM / rail_px_width_at_y(quad, y)


def vanish_y_of(q):
    ny, fy = q['near_y'], q['fy']
    denom = (fy - ny)
    aL = (q['fl'] - q['nl']) / denom; aR = (q['fr'] - q['nr']) / denom
    y_lines = ny + (q['nr'] - q['nl']) / (aL - aR) if abs(aL - aR) > 1e-9 else None
    y_width = ny - q['near_w'] * (ny - fy) / (q['near_w'] - q['far_w'] + 1e-9)
    if y_lines is None:
        return float(y_width)
    return float((y_lines + y_width) / 2)


def quad_feats(q):
    y_v = vanish_y_of(q); cy = q['H'] / 2
    return [(cy - y_v) / q['H'], q['far_w'] / q['near_w'], q['near_w'] / q['W'], (q['near_y'] - q['fy']) / q['H']]


def fhat_from_quad(quad):
    if quad is None:
        return FOCAL_MEAN
    return float(np.dot(F_COEF, quad_feats(quad)) + F_INT)


def focal_px_of(focal_mm, frame_w):
    return focal_mm / SENSOR_W * frame_w


# ==============================================================================
# [4] Depth (DA3)
# ==============================================================================
class DA3Depth:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name
        sh = self.sess.get_inputs()[0].shape
        self.fh = sh[2] if isinstance(sh[2], int) else None
        self.fw = sh[3] if isinstance(sh[3], int) else None

    def predict_raw(self, bgr):
        oh, ow = bgr.shape[:2]
        if self.fh and self.fw:
            th, tw = self.fh, self.fw
        else:
            P = 14; th = max((oh//P)*P, P); tw = max((ow//P)*P, P)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (tw, th)).astype(np.float32) / 255.0
        t = r.transpose(2, 0, 1)[None]
        raw = self.sess.run([self.oname], {self.iname: t})[0][0, 0]
        return cv2.resize(raw, (ow, oh))


da3_model = DA3Depth(DEPTH_ONNX_PATH)
print(f"✅ depth_model 로드 완료  provider={da3_model.sess.get_providers()}")


def scalars_rail(tr, quad):
    s = rail_scale_at_y(quad, tr['y_bottom'])
    return np.array([tr['bbox_w'] * s, tr['bbox_h'] * s], np.float32)


def scalars_da3(tr):
    g = tr['raw_med'] / 300.0
    return np.array([tr['bbox_w'] * g, tr['bbox_h'] * g], np.float32)



# ==============================================================================
# [5] Box Regressor (ONNX 앙상블, 5-fold)
# ==============================================================================
reg_sessions = []
for p in REG_ONNX_PATHS:
    if not os.path.exists(p):
        raise FileNotFoundError(f"필수 파일 없음: {p}")
    reg_sessions.append(ort.InferenceSession(p, providers=providers))

print(f"✅ box_regressor 앙상블 {len(reg_sessions)}개 로드 완료  "
      f"provider={reg_sessions[0].get_providers()}")


def predict_track_eval_onnx(tr, quad, f_hat_v, W):
    f_px = focal_px_of(f_hat_v, W)
    dp = tr['raw'] * (f_px / 300.0)
    r = tr['rgb'].astype(np.float32) / 255.0
    dp = (dp - DEPTH_MEAN) / DEPTH_STD
    img = np.concatenate([r.transpose(2, 0, 1), dp[None]], 0)[None].astype(np.float32)

    base = scalars_rail(tr, quad) if quad is not None else scalars_da3(tr) * SC_FALLBACK_K
    sc_i = np.array([base[0], base[1], f_hat_v], np.float32)
    sc_n = ((sc_i - SC_MEAN) / SC_STD)[None].astype(np.float32)

    # ★ 5개 세션 각각 추론 후 평균 (K-fold 앙상블)
    ratio_preds = [sess.run(['ratio_pred'], {'img': img, 'sc': sc_n})[0][0]
                   for sess in reg_sessions]
    r_pr = np.mean(ratio_preds, axis=0) * R_STD + R_MEAN
    r_pr = np.clip(r_pr, R_CLIP_LO, R_CLIP_HI)

    p = r_pr * np.array([sc_i[0], sc_i[1], sc_i[1]])
    return [max(1., float(p[0])), max(1., float(p[1])), max(1., float(p[2]))]
# ==============================================================================
# [6] 영상 1개 추론
# ==============================================================================
def infer_one_video(video_path):
    lv = os.path.join(WORK_DIR, os.path.basename(video_path))
    if not os.path.exists(lv):
        shutil.copy(video_path, lv)
    roi_pre = get_roi_for_test_video(lv)

    merged, vx, vy, mm, mg, all_dets, frames, roi = run_two_pass_tracking(video_path, precomputed_roi=roi_pre)
    if not merged or not frames:
        return []

    quad = estimate_rail_quad(frames)
    f_hat = fhat_from_quad(quad)
    H, W = frames[0][0].shape[:2]
    rx1, ry1, rx2, ry2 = map(int, roi) if roi is not None else (0, 0, W, H)

    reps = []
    for tid, hist in merged.items():
        best = max(hist, key=lambda h: (h[1][2]-h[1][0]) * (h[1][3]-h[1][1]))
        fi, (x1, y1, x2, y2), conf = best
        ox1, oy1 = max(0, int(x1+rx1)), max(0, int(y1+ry1))
        ox2, oy2 = min(W, int(x2+rx1)), min(H, int(y2+ry1))
        if ox2 <= ox1+3 or oy2 <= oy1+3:
            continue
        reps.append((fi, (ox1, oy1, ox2, oy2)))

    raw_cache, preds = {}, []
    for fi, (ox1, oy1, ox2, oy2) in reps:
        frame = frames[fi][0]
        if fi not in raw_cache:
            raw_cache[fi] = da3_model.predict_raw(frame)
        raw = raw_cache[fi]
        rc, dc = frame[oy1:oy2, ox1:ox2], raw[oy1:oy2, ox1:ox2]
        if rc.size == 0:
            continue
        rc_rgb = cv2.cvtColor(rc, cv2.COLOR_BGR2RGB)
        tr = {'rgb': cv2.resize(rc_rgb, (CROP_SIZE, CROP_SIZE)).astype(np.uint8),
              'raw': cv2.resize(dc, (CROP_SIZE, CROP_SIZE)).astype(np.float32),
              'raw_med': float(np.median(dc)),
              'bbox_w': float(ox2-ox1), 'bbox_h': float(oy2-oy1),
              'y_bottom': float(oy2)}
        preds.append(predict_track_eval_onnx(tr, quad, f_hat, W))
    return preds


# ==============================================================================
# [7] 엔트리 포인트
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='영상 폴더 경로')
    args = parser.parse_args()

    input_dir = args.input
    video_files = sorted(glob.glob(os.path.join(input_dir, '*.mp4')))
    print(f"입력 폴더: {input_dir}  |  영상 {len(video_files)}개")

    videos_out = []
    for i, vp in enumerate(video_files):
        vid = os.path.splitext(os.path.basename(vp))[0]
        try:
            preds = infer_one_video(vp)
        except Exception:
            print(f"  ⚠️ {vid} 실패 — 빈 결과")
            traceback.print_exc()
            preds = []
        videos_out.append({
            'video_id': vid,
            'objects': [{'size_cm': {'w': round(p[0], 3), 'd': round(p[1], 3), 'h': round(p[2], 3)}}
                        for p in preds]
        })
        print(f"[{i+1}/{len(video_files)}] {vid}: {len(preds)}개")

    result_path = os.path.join(SCRIPT_DIR, 'result.json')
    with open(result_path, 'w') as f:
        json.dump({'videos': videos_out}, f, indent=2)
    print(f"✅ 저장 완료: {result_path}")


if __name__ == '__main__':
    main()