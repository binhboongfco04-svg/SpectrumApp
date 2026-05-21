import cv2
from typing import Union, List, Optional, Tuple
import numpy as np
import json
import os
# ── Thay thế scipy.signal bằng NumPy thuần (không cần SciPy khi build Android) ──

def _savgol_coeffs_numpy(window: int, polyorder: int) -> np.ndarray:
    """Tính kernel Savitzky-Golay bằng NumPy (least-squares Vandermonde)."""
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vstack([x ** i for i in range(polyorder + 1)]).T  # (window, poly+1)
    ATA_inv = np.linalg.pinv(A.T @ A)
    return (ATA_inv @ A.T)[0]  # hàng 0 = hệ số cho điểm giữa


def savgol_filter(arr: np.ndarray, window_length: int, polyorder: int, **kwargs) -> np.ndarray:
    """
    Thay thế scipy.signal.savgol_filter — chỉ dùng NumPy.
    Hỗ trợ cùng signature cơ bản: savgol_filter(arr, window_length, polyorder).
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if window_length < 3 or window_length % 2 == 0 or n < window_length:
        return arr.copy()
    polyorder = min(polyorder, window_length - 1)
    kernel = _savgol_coeffs_numpy(window_length, polyorder)
    half = window_length // 2
    padded = np.pad(arr, half, mode='reflect')
    return np.convolve(padded, kernel[::-1], mode='valid')


def find_peaks(arr: np.ndarray, distance=None, prominence=None, height=None, **kwargs):
    """
    Thay thế scipy.signal.find_peaks — chỉ dùng NumPy.
    Hỗ trợ: distance, prominence, height. Trả về (peaks, props) giống scipy.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)

    # Local maxima: arr[i] > arr[i-1] và arr[i] > arr[i+1]
    peaks = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))[0] + 1

    props = {}

    # Lọc theo height
    if height is not None:
        min_h = height[0] if hasattr(height, '__len__') else height
        peaks = peaks[arr[peaks] >= min_h]

    # Tính prominence cho tất cả peak hiện tại
    prom_vals = np.zeros(len(peaks), dtype=np.float64)
    for k, pk in enumerate(peaks):
        left_min = float(np.min(arr[:pk])) if pk > 0 else arr[pk]
        right_min = float(np.min(arr[pk + 1:])) if pk < n - 1 else arr[pk]
        prom_vals[k] = arr[pk] - max(left_min, right_min)

    # Lọc theo prominence
    if prominence is not None:
        min_prom = prominence[0] if hasattr(prominence, '__len__') else prominence
        keep = prom_vals >= min_prom
        peaks = peaks[keep]
        prom_vals = prom_vals[keep]

    # Lọc theo distance (giữ peak có prominence cao hơn khi gần nhau)
    if distance is not None and len(peaks) > 1:
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(len(peaks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(peaks)):
                if not keep[j]:
                    continue
                if peaks[j] - peaks[i] < distance:
                    if prom_vals[i] >= prom_vals[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                else:
                    break
        peaks = peaks[keep]
        prom_vals = prom_vals[keep]

    props['prominences'] = prom_vals
    return peaks, props

# ─────────────────────────────────────────────
# 1. IMAGE LOADING & LUMINANCE
# ─────────────────────────────────────────────

def load_image(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Không đọc được ảnh: {path}")
        return None
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def luminance(img: np.ndarray) -> np.ndarray:
    """BGR → grayscale luminance (float64), hệ số chuẩn ITU-R BT.601"""
    return np.dot(img[..., ::-1].astype(np.float64)[..., :3], [0.299, 0.587, 0.114])


# ─────────────────────────────────────────────
# 2. ROTATE (per-channel angle)
# ─────────────────────────────────────────────

def _rotate_lum(lum: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-4:
        return lum.copy()
    h, w = lum.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        lum, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


# ─────────────────────────────────────────────
# 3. CORE PROFILE EXTRACTION
# ─────────────────────────────────────────────

def _savgol_smooth_1d(arr: np.ndarray, window: int = 3, poly: int = 2) -> np.ndarray:
    # FIX #2: Dùng scipy.signal.savgol_filter thay vì tự implement.
    # Bản tự implement cũ có bug: normalize kernel theo tổng (kernel /= kernel.sum())
    # chỉ đúng với poly=0 (moving average). Với poly>=1, Savitzky-Golay kernel
    # không cần normalize theo tổng — nó tự bảo toàn đa thức bậc poly.
    # Việc normalize sai làm lệch biên độ và hình dạng phổ sau smooth.
    arr = np.asarray(arr, dtype=np.float64)
    if window < 3 or window % 2 == 0 or len(arr) < window:
        return arr.copy()
    # Đảm bảo poly < window (yêu cầu của savgol_filter)
    poly = min(poly, window - 1)
    return savgol_filter(arr, window_length=window, polyorder=poly)


def _extract_profile(lum_rotated: np.ndarray,
                     y_center: float,
                     col_start: int, col_end: int,
                     dr: int,
                     mode: str = 'mean',
                     ws: int = 3) -> np.ndarray:
    h, w = lum_rotated.shape
    col_start = int(np.clip(col_start, 0, w - 1))
    col_end   = int(np.clip(col_end,   0, w - 1))
    if col_start > col_end:
        raise ValueError(f"col_start={col_start} > col_end={col_end}")

    dd = col_end - col_start + 1
    rows = np.zeros((dr * 2 + 1, dd), dtype=np.float64)
    for i in range(dr * 2 + 1):
        row_idx = int(np.clip(y_center - dr + i, 0, h - 1))
        rows[i, :] = lum_rotated[row_idx, col_start:col_end + 1]

    profile = np.mean(rows, axis=0) if mode == 'mean' else np.amax(rows, axis=0)
    return _savgol_smooth_1d(profile, window=ws, poly=2)


def shift_subsample(arr: np.ndarray, shift_px: float) -> np.ndarray:
    """
    Dịch mảng 1D với độ phân giải dưới-pixel bằng nội suy tuyến tính.
    shift_px > 0: kéo phổ sang phải.
    """
    arr = np.asarray(arr, dtype=np.float64)
    x = np.arange(len(arr), dtype=np.float64)
    x_src = x - float(shift_px)
    return np.interp(x, x_src, arr, left=arr[0], right=arr[-1])


def _normalize_roi(roi: Optional[Tuple[int, int]], n: int) -> Tuple[int, int]:
    if roi is None:
        return 0, n - 1
    s, e = int(roi[0]), int(roi[1])
    s = int(np.clip(s, 0, n - 1))
    e = int(np.clip(e, 0, n - 1))
    if s > e:
        s, e = e, s
    return s, e

def _subpixel_peak_parabola(arr: np.ndarray, idx: int) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if idx <= 0 or idx >= n - 1:
        return float(idx)

    y1 = float(arr[idx - 1])
    y2 = float(arr[idx])
    y3 = float(arr[idx + 1])

    denom = (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-12:
        return float(idx)

    dx = 0.5 * (y1 - y3) / denom
    return float(idx) + dx


def _make_roi_from_reference_peak(arr: np.ndarray,
                                  search_roi: Optional[Tuple[int, int]] = None,
                                  smooth_window: int = 51,
                                  half_width: int = 100) -> Tuple[int, int, float]:
    """
    Tìm đỉnh trên phổ chuẩn, rồi tạo ROI = peak ± half_width.
    Sử dụng window lớn hơn để smooth trước khi tìm peak → tránh peak giả do nhiễu đèn.
    half_width được mở rộng để ROI bao phủ sườn đỉnh đầy đủ hơn.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 7:
        c = max(0, n // 2)
        return max(0, c - half_width), min(n - 1, c + half_width), float(c)

    # Dùng window lớn hơn để loại bỏ nhiễu nhỏ từ đèn không ổn định
    w = int(smooth_window)
    w = min(max(w, 5), n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        w = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)

    # Smooth 2 lần: lần 1 loại nhiễu cao tần, lần 2 tìm cấu trúc tổng thể
    sm = _savgol_smooth_1d(arr, window=w, poly=3)
    w2 = min(max(w * 2 + 1, 7), n if n % 2 == 1 else n - 1)
    if w2 % 2 == 0:
        w2 -= 1
    sm_coarse = _savgol_smooth_1d(arr, window=w2, poly=2)

    roi_s, roi_e = _normalize_roi(search_roi, n)
    seg = sm_coarse[roi_s:roi_e + 1]
    if len(seg) < 3:
        c = (roi_s + roi_e) // 2
        return max(0, c - half_width), min(n - 1, c + half_width), float(c)

    peak_idx = roi_s + int(np.argmax(seg))
    # Dùng sm (ít smooth hơn) để subpixel refine chính xác hơn
    peak_pos = _subpixel_peak_parabola(sm, peak_idx)

    # Mở rộng half_width để ROI bao phủ đủ sườn đỉnh (giúp cross-corr tốt hơn)
    effective_half = max(half_width, 150)
    c = int(round(peak_pos))
    s = max(0, c - int(effective_half))
    e = min(n - 1, c + int(effective_half))
    return s, e, float(peak_pos)


def _gaussian_subpixel_refine(scores: np.ndarray, i_best: int) -> float:
    """
    Subpixel refine bằng Gaussian fit 5-điểm (chính xác hơn parabola 3-điểm
    khi peak correlation không đối xứng hoàn hảo).
    Fallback về parabola 3-điểm nếu điều kiện Gaussian không thỏa mãn.
    """
    n = len(scores)
    # Gaussian 5-điểm (nếu đủ điểm hai bên)
    if 2 <= i_best <= n - 3:
        ys = scores[i_best - 2: i_best + 3].copy()
        # Clamp để log không bị âm/NaN
        ys = np.clip(ys, 1e-12, None)
        log_ys = np.log(ys)
        xs = np.array([-2., -1., 0., 1., 2.], dtype=np.float64)
        # Fit bậc 2 lên log → peak Gaussian = -b/(2a)
        coeffs = np.polyfit(xs, log_ys, 2)
        a_coef, b_coef = coeffs[0], coeffs[1]
        if a_coef < -1e-12:          # đỉnh hướng xuống → hợp lệ
            dx = -b_coef / (2.0 * a_coef)
            if abs(dx) <= 2.0:       # giới hạn refine trong ±2px
                return float(i_best) + dx
    # Fallback: parabola 3-điểm
    if 0 < i_best < n - 1:
        y1, y2, y3 = scores[i_best - 1], scores[i_best], scores[i_best + 1]
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            return float(i_best) + 0.5 * (y1 - y3) / denom
    return float(i_best)


def find_wavelength_shift(src: np.ndarray,
                          ref: np.ndarray,
                          smooth_window: int = 51,
                          search_roi: Optional[Tuple[int, int]] = None,
                          max_shift_px: int = 30) -> float:
    """
    Tìm shift pixel để căn chỉnh profile src về profile ref.
    Dùng normalized cross-correlation trong ROI sau smooth,
    subpixel refine bằng Gaussian fit 5-điểm trên correlation scores.

    max_shift_px: giới hạn tìm kiếm tối đa (px). Mặc định 30px.
    Giá trị này nên lớn hơn shift thực tế lớn nhất của thiết bị.
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if src.shape != ref.shape:
        raise ValueError("find_wavelength_shift(): src/ref phải cùng chiều dài.")

    n = len(src)
    if n < 7:
        return 0.0

    w = int(smooth_window)
    w = min(max(w, 5), n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        w = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)

    roi_s, roi_e = _normalize_roi(search_roi, n)
    src_sm = _savgol_smooth_1d(src, window=w, poly=3)[roi_s:roi_e + 1]
    ref_sm = _savgol_smooth_1d(ref, window=w, poly=3)[roi_s:roi_e + 1]
    m = len(src_sm)
    if m < 7:
        return 0.0

    src0 = src_sm - np.mean(src_sm)
    ref0 = ref_sm - np.mean(ref_sm)
    src_norm = np.linalg.norm(src0)
    ref_norm = np.linalg.norm(ref0)
    if src_norm < 1e-12 or ref_norm < 1e-12:
        return 0.0

    x = np.arange(m, dtype=np.float64)
    # Giới hạn max_lag = max_shift_px để tránh false peak ở xa.
    max_lag = int(np.clip(max_shift_px, 1, m // 3))
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int32)
    scores = np.zeros(len(lags), dtype=np.float64)

    for i, lag in enumerate(lags):
        shifted = np.interp(x, x - float(lag), src0, left=src0[0], right=src0[-1])
        den = np.linalg.norm(shifted) * ref_norm
        scores[i] = float(np.dot(shifted, ref0) / den) if den > 1e-12 else -1.0

    i_best = int(np.argmax(scores))

    lag_raw = float(lags[i_best])
    print(f"      [DEBUG] best_lag={lag_raw:+.1f}px  score={scores[i_best]:.4f}  max_lag={max_lag}")

    if scores[i_best] < 0.5:
        print(f"    [Shift WARNING] Score thấp ({scores[i_best]:.3f}) → shift = 0.0 px")
        return 0.0

    # ── Subpixel refine bằng Gaussian 5-điểm trên correlation scores ─────────
    # Chính xác hơn parabola 3-điểm khi peak correlation không đối xứng
    # (thường xảy ra với kênh xa trục chuẩn như ch1, ch6).
    i_best_subpx = _gaussian_subpixel_refine(scores, i_best)
    lag_final = float(lags[0]) + i_best_subpx   # chuyển về lag (có dấu)

    print(f"      [DEBUG] lag_final={lag_final:+.3f}px")

    return lag_final


# ─────────────────────────────────────────────
# 3b. REFINE Y_CENTER
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 4. DARK FRAME
# ─────────────────────────────────────────────

def _load_dark_frame(dark_paths: Optional[List[str]]) -> Optional[np.ndarray]:
    if not dark_paths:
        return None
    frames = []
    for p in dark_paths:
        img = load_image(p)
        if img is not None:
            frames.append(luminance(img))
    return np.mean(np.array(frames), axis=0) if frames else None


def _subtract_dark(lum: np.ndarray, dark_lum: Optional[np.ndarray]) -> np.ndarray:
    if dark_lum is None:
        return lum
    return np.clip(lum - dark_lum, 0.0, None)


# ─────────────────────────────────────────────
# 5. MASKED ABSORBANCE
# ─────────────────────────────────────────────

def _masked_absorbance(profile_ref: np.ndarray,
                        profile_sam: np.ndarray,
                        ref_threshold_pct: float = 0.15) -> np.ndarray:  # FIX noise: tăng 0.1→0.15
    ref = np.asarray(profile_ref, dtype=np.float64)
    sam = np.asarray(profile_sam, dtype=np.float64)

    ref_peak = ref.max()
    valid = (ref > ref_threshold_pct * ref_peak) if ref_peak > 0 \
            else np.zeros(len(ref), dtype=bool)

    eps = max(ref.max() * 1e-4, 1e-9)  
    A = np.full(len(ref), np.nan, dtype=np.float64)
    A[valid] = -np.log10(
        np.clip(sam[valid], eps, None) / np.clip(ref[valid], eps, None)
    )
    A = np.where(np.isfinite(A), np.clip(A, -0.01, 3.0), np.nan)  # FIX noise: -0.05→-0.01
    return A


# ─────────────────────────────────────────────
# 6. CALIBRATION PARAMS
# ─────────────────────────────────────────────

_CALIB_JSON = os.path.join(os.path.dirname(__file__), "calibration.json")

_DEFAULT_PARAMS: dict = {
    "y_centers":           [],
    "angles":              [],
    "col_start":           0,
    "col_end":             9999,
    "dr":                  15,
    "ws":                  3,
    "mode":                "mean",
    "wl_min":              400,
    "wl_max":              750,
    # Trục bước sóng chuẩn (từ kênh 3+4)
    "calib_a":             None,   # nm/pixel, dùng chung MỌI kênh sau shift
    "calib_b":             None,   # intercept nm
    "calib_offset":        0,
    # Pixel vị trí 2 laser trên trục chuẩn (ch3+ch4) — do người dùng xác định
    "pixel_532":           1912.3,   # pixel ứng với 532 nm
    "pixel_653":           2286.0,   # pixel ứng với 653 nm
    "ref_threshold_pct":   0.1,
    "shift_smooth_window": 51,
    "max_shift_px":        60,   # Giới hạn tìm kiếm shift tối đa (px) — đủ cho kênh lệch nhiều (ch1/ch6)
    # Giới hạn tối đa (nm) cho per-measurement anchor correction.
    "anchor_max_correction_nm": 10,
    # Bán kính ROI (nm) khi tìm peak hấp thụ: dùng cho cả peak logging và anchor.
    "peak_search_half_nm": 15.0,
    "calib_scale_a": [0.98283993, 1.03496017, 1.08841071, 1.07586766, 1.01564048, 1.08433889],
    "calib_scale_b": [0.15117493, 0.16339897, 0.12077977, 0.12414855, 0.12635715, 0.04736341],
    # "calib_scale_a_fluo": [125.86, 125.86, 125.86, 125.86, 125.86, 125.86],  # hệ số a từ fit Avantes
    # "calib_scale_b_fluo": [-3289.5, -3289.5, -3289.5, -3289.5, -3289.5, -3289.5],  # hệ số b từ fit Avantes
    "calib_scale_a_fluo": [179.86157, 179.86157, 179.86157, 179.86157, 179.86157, 179.86157],  # hệ số a từ fit Avantes
    "calib_scale_b_fluo": [-1616.48458,-1616.48458,-1616.48458,-1616.48458,-1616.48458,-1616.48458],  # hệ số b từ fit Avantes
    "wl_offset_nm": 4,
}


def load_params() -> dict:
    if os.path.exists(_CALIB_JSON):
        try:
            with open(_CALIB_JSON, encoding="utf-8") as f:
                p = json.load(f)
            for k, v in _DEFAULT_PARAMS.items():
                p.setdefault(k, v)
            return p
        except Exception as e:
            print(f"⚠️ Không đọc được calibration.json: {e}. Dùng DEFAULT_PARAMS.")
    return dict(_DEFAULT_PARAMS)


def _apply_calib_scale(results: List[dict], params: dict) -> List[dict]:
    """
    Áp dụng hiệu chuẩn tuyến tính với máy chuẩn cho TẤT CẢ kênh:
        A_cal = a_i * A_raw + b_i
    Bỏ qua nếu calib_scale_a/b chưa được cấu hình ([] hoặc thiếu).
    """
    a_list = params.get("calib_scale_a", [])
    b_list = params.get("calib_scale_b", [])
    if not a_list or not b_list:
        return results

    print("  [CalibScale] Áp dụng A_cal = a*A_raw + b:")
    for ch_idx, r in enumerate(results):
        if ch_idx >= len(a_list) or ch_idx >= len(b_list):
            print(f"    ch{ch_idx+1}: không có hệ số → bỏ qua")
            continue
        a = float(a_list[ch_idx])
        b = float(b_list[ch_idx])
        A = np.asarray(r["absorbance"], dtype=np.float64)
        r["absorbance"] = np.where(np.isfinite(A), a * A + b, np.nan)
        # Cập nhật peak_value
        pw = r.get("peak_wavelength")
        wl = np.asarray(r["wavelengths"], dtype=np.float64)
        if pw is not None and len(wl) > 0:
            r["peak_value"] = float(np.interp(pw, wl, r["absorbance"]))
        print(f"    ch{ch_idx+1}: a={a:+.4f}, b={b:+.4f}")
    return results


def _normalize_intercept_between_channels(
        results: List[dict],
        ref_ch_indices: tuple = (2, 3),
        baseline_offset_nm: float = 100,
        fallback_wl: Optional[float] = None,
) -> List[dict]:
    """
    Chuẩn hóa baseline (intercept) giữa các kênh sau khi đã tính A.

    Nguyên lý:
        - Mỗi kênh dùng I0 riêng từ cuvette blank của nó → nếu các cuvette
          blank không hoàn toàn giống nhau (độ sạch, vị trí, góc đặt), A sẽ
          có intercept lệch khác nhau dù đo cùng mẫu.
        - Hàm này lấy giá trị A tại một bước sóng "baseline" (ngoài vùng
          hấp thụ của mẫu) của kênh reference (ch3+ch4) làm mốc 0, rồi trừ
          đi offset tương ứng của từng kênh.

    Tham số:
        ref_ch_indices   : chỉ số (0-based) của kênh reference. Mặc định (2,3)
                           tức ch3+ch4 — thường là kênh ổn định nhất.
        baseline_offset_nm: khoảng cách tính từ peak_wavelength về phía ngắn
                           hơn để lấy điểm baseline. Mặc định 50 nm.
        fallback_wl      : nếu không tìm được peak_wavelength, dùng bước sóng
                           này làm baseline. None = dùng wl_min của kênh.
    """
    # 1. Xác định bước sóng baseline
    ref_peaks = [
        results[i]["peak_wavelength"]
        for i in ref_ch_indices
        if i < len(results) and results[i].get("peak_wavelength") is not None
    ]
    if ref_peaks:
        ref_peak_wl  = float(np.mean(ref_peaks))
        baseline_wl  = ref_peak_wl - baseline_offset_nm
    elif fallback_wl is not None:
        baseline_wl = float(fallback_wl)
    else:
        # Dùng bước sóng nhỏ nhất chung của các kênh reference
        wl_mins = [results[i]["wavelengths"][0]
                   for i in ref_ch_indices
                   if i < len(results) and len(results[i]["wavelengths"]) > 0]
        if not wl_mins:
            return results            # không đủ dữ liệu → bỏ qua
        baseline_wl = float(np.max(wl_mins))   # lấy max để nằm trong tất cả kênh

    # 2. Tính baseline trung bình của kênh reference tại baseline_wl
    ref_vals = []
    for i in ref_ch_indices:
        if i >= len(results):
            continue
        wl = np.asarray(results[i]["wavelengths"], dtype=np.float64)
        A  = np.asarray(results[i]["absorbance"],  dtype=np.float64)
        if len(wl) == 0 or baseline_wl < wl[0] or baseline_wl > wl[-1]:
            continue
        val = float(np.interp(baseline_wl, wl, A))
        if np.isfinite(val):
            ref_vals.append(val)

    if not ref_vals:
        print("  [NormBaseline] ⚠️ Không lấy được baseline từ kênh reference → bỏ qua")
        return results
    ref_baseline = float(np.mean(ref_vals))

    # 3. Trừ offset cho từng kênh
    print(f"  [NormBaseline] baseline_wl={baseline_wl:.1f} nm, "
          f"ref_baseline={ref_baseline:.5f}")
    for r in results:
        wl = np.asarray(r["wavelengths"], dtype=np.float64)
        A  = np.asarray(r["absorbance"],  dtype=np.float64)
        ch = r["channel"]
        if len(wl) == 0 or baseline_wl < wl[0] or baseline_wl > wl[-1]:
            print(f"    ch{ch}: baseline_wl nằm ngoài dải → bỏ qua")
            continue
        ch_baseline = float(np.interp(baseline_wl, wl, A))
        offset = ch_baseline - ref_baseline
        if abs(offset) > 0.03:
            r["absorbance"] = np.where(np.isfinite(A), A - offset, np.nan)
        # Cập nhật peak_value sau khi dịch
        pw = r.get("peak_wavelength")
        if pw is not None and len(wl) > 0:
            r["peak_value"] = float(np.interp(pw, wl, r["absorbance"]))
        print(f"    ch{ch}: ch_baseline={ch_baseline:.5f}, "
              f"offset={offset:+.5f}")
    return results


def _find_peak_in_roi(wl: np.ndarray, A: np.ndarray,
                      center_nm: float, half_nm: float,
                      min_peak_value: float = 0.01) -> Optional[float]:
    """
    Tìm đỉnh hấp thụ trong ROI hẹp [center_nm ± half_nm].
    Dùng subpixel refine (parabola). Trả về None nếu không tìm được peak hợp lệ.
    """
    wl = np.asarray(wl, dtype=np.float64)
    A  = np.asarray(A,  dtype=np.float64)
    mask = np.isfinite(A) & (wl >= center_nm - half_nm) & (wl <= center_nm + half_nm)
    if not np.any(mask):
        return None
    local_idx = np.where(mask)[0]
    best_local = int(np.nanargmax(A[mask]))
    pk_global  = local_idx[best_local]
    if A[pk_global] < min_peak_value:
        return None
    # Subpixel refine
    pk_sub = _subpixel_peak_parabola(A, pk_global)
    return float(np.interp(pk_sub, np.arange(len(wl)), wl))


def _apply_anchor_correction(results: List[dict],
                              ref_ch_indices: Tuple[int, int] = (2, 3),
                              max_correction_nm: float = 1.5,
                              min_peak_value: float = 0.01,
                              anchor_search_half_nm: float = 20.0,
                              roi_half_nm: float = 15.0) -> List[dict]:
    """
    Tầng 2 — tinh chỉnh lệch nhỏ còn dư sau tầng 1 (shift pixel từ phổ đèn).

    Nguyên lý:
      - Tầng 1 (cross-corr trên phổ đèn) sửa lệch quang học cố định per-channel.
      - Tầng 2 này sửa phần dư nhỏ biến động theo từng lần đo (đèn trôi nhiệt độ...).

    Thuật toán:
      1. Tính anchor_wl = mean peak ch3+ch4, tìm trong ROI hẹp ±anchor_search_half_nm
         (không dùng global max — tránh peak nhiễu hoặc cấu trúc phụ).
      2. Với kênh i: tìm peak trong ROI ±roi_half_nm quanh anchor_wl.
         residual_i = peak_i - anchor_wl.
      3. Nếu |residual_i| < max_correction_nm: dịch wl_i -= residual_i.
         Nếu không: bỏ qua (không correction).

    Tham số:
      ref_ch_indices        : kênh làm anchor (0-based).
      max_correction_nm     : giới hạn tối đa tinh chỉnh. Giữ nhỏ (1–2nm).
      min_peak_value        : ngưỡng A tối thiểu để tin peak là thật.
      anchor_search_half_nm : bán kính ROI khi tìm peak anchor (ch3+ch4).
      roi_half_nm           : bán kính ROI khi tìm peak từng kênh.
    """
    if not results:
        return results

    # ── 1. Tìm anchor_wl từ ch3+ch4 trong ROI hẹp ───────────────────────────
    # Ước lượng sơ bộ vị trí anchor từ trung tâm dải wl (dùng làm center cho ROI)
    anchor_candidates = []
    for idx in ref_ch_indices:
        if idx >= len(results):
            continue
        r   = results[idx]
        wl  = np.asarray(r.get("wavelengths", []), dtype=np.float64)
        A   = np.asarray(r.get("absorbance",  []), dtype=np.float64)
        if len(wl) < 3:
            continue
        # Lần 1: global argmax để lấy vùng ước lượng
        finite = np.isfinite(A)
        if not np.any(finite):
            continue
        rough_center = float(wl[np.nanargmax(np.where(finite, A, -np.inf))])
        # Lần 2: tìm chính xác trong ROI quanh rough_center
        pw = _find_peak_in_roi(wl, A, rough_center, anchor_search_half_nm, min_peak_value)
        if pw is not None:
            anchor_candidates.append(pw)

    if not anchor_candidates:
        print("  [AnchorCorr] Không tìm được anchor peak (ch3/ch4) → bỏ qua correction.")
        for r in results:
            r["anchor_correction_nm"] = 0.0
        return results

    anchor_wl = float(np.mean(anchor_candidates))
    print(f"  [AnchorCorr] Anchor (ch3+ch4, ROI±{anchor_search_half_nm}nm) = {anchor_wl:.3f} nm")

    # ── 2. Per-channel correction (áp dụng TẤT CẢ kênh, kể cả ch3+ch4) ───────
    # ch3+ch4 đã dùng làm anchor_wl nhưng cũng cần correction về đúng anchor_wl
    # (residual của chúng thường nhỏ vì anchor = mean của chúng, nhưng không = 0
    #  khi 2 kênh lệch nhau một ít).
    for ch_idx, r in enumerate(results):
        wl_arr = np.asarray(r.get("wavelengths", []), dtype=np.float64)
        A_arr  = np.asarray(r.get("absorbance",  []), dtype=np.float64)

        # Tìm peak trong ROI hẹp quanh anchor_wl (không global max)
        pw = _find_peak_in_roi(wl_arr, A_arr, anchor_wl, roi_half_nm, min_peak_value)
        if pw is None:
            r["anchor_correction_nm"] = 0.0
            print(f"    ch{ch_idx+1}: không tìm được peak trong ROI "
                  f"[{anchor_wl-roi_half_nm:.0f}–{anchor_wl+roi_half_nm:.0f}nm] → bỏ qua")
            continue

        residual = pw - anchor_wl
        if abs(residual) > max_correction_nm:
            r["anchor_correction_nm"] = 0.0
            print(f"    ch{ch_idx+1}: residual={residual:+.3f}nm > giới hạn "
                  f"{max_correction_nm}nm → bỏ qua")
            continue

        # Dịch trục wavelength
        wl_arr = wl_arr - residual

        # Tính lại peak_wavelength sau khi dịch (trong ROI)
        new_peak_wl = _find_peak_in_roi(wl_arr, A_arr, anchor_wl, roi_half_nm, min_peak_value)
        if new_peak_wl is None:
            new_peak_wl = float(anchor_wl)

        r["wavelengths"]          = wl_arr
        r["peak_wavelength"]      = new_peak_wl
        r["anchor_correction_nm"] = float(-residual)

        print(f"    ch{ch_idx+1}: peak_in_ROI={pw:.3f}nm  residual={residual:+.3f}nm "
              f"→ correction={-residual:+.3f}nm  peak_new={new_peak_wl:.3f}nm")

    return results


def pixels_to_wavelengths_unified(n_pixels: int,
                                   pixel_offset: int = 0,
                                   params: Optional[dict] = None) -> np.ndarray:
    """
    Chuyển pixel → nm dùng MỘT công thức duy nhất cho tất cả kênh.

    Công thức tuyến tính neo vào 2 điểm laser:
        nm = calib_a * pixel + calib_b

    Trong đó calib_a/b được xác định từ pixel_532 và pixel_653 trên trục chuẩn.
    pixel_offset: vị trí bắt đầu (= col_start) để pixel tuyệt đối đúng.
    """
    if params is None:
        params = load_params()
    a = params.get("calib_a")
    b = params.get("calib_b")
    if a is None or b is None:
        raise ValueError(
            "Chưa có calib_a/b. Hãy chạy run_calibration() để xác định "
            "pixel_532 và pixel_653 trên trục chuẩn kênh 3+4."
        )
    pixels = np.arange(pixel_offset, pixel_offset + n_pixels, dtype=np.float64)
    return float(a) * pixels + float(b)


def _filter_spectrum_by_range(values: np.ndarray, wavelengths: np.ndarray,
                               wl_min: float, wl_max: float):
    values      = np.asarray(values,      dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    mask = (np.isfinite(wavelengths)
            & (wavelengths >= float(wl_min))
            & (wavelengths <= float(wl_max)))
    return wavelengths[mask], values[mask]


# ─────────────────────────────────────────────
# 6b. BƯỚC 1 — TÌM SHIFT PIXEL TỪNG KÊNH
# (domain: pixel, trục chuẩn = trung bình ch3+ch4)
# ─────────────────────────────────────────────

def _estimate_channel_shifts_from_ref_profiles(
        ref_lums: List[np.ndarray],
        y_centers: List[float],
        angles: List[float],
        col_start: int,
        col_end_eff: int,
        dr: int,
        mode: str,
        ws: int,
        smooth_window: int,
        search_roi: Optional[Tuple[int, int]],
        max_shift_px: int = 30) -> np.ndarray:
    if not ref_lums:
        raise ValueError("Không có ảnh reference để ước lượng shift.")

    # Dùng trung bình tất cả ảnh ref để tính shift ổn định hơn
    # (tránh trường hợp ảnh đầu tiên nhiễu do đèn chưa ổn định)
    num_ch = len(y_centers)
    dd = col_end_eff - col_start + 1
    prof_ref_ch = np.zeros((num_ch, dd), dtype=np.float64)

    for ch_idx in range(num_ch):
        y_c   = float(y_centers[ch_idx])
        angle = float(angles[ch_idx])
        stack = []
        for lum_ref in ref_lums:
            I_ref_rot = _rotate_lum(lum_ref, angle)
            stack.append(_extract_profile(I_ref_rot, y_c, col_start, col_end_eff,
                                          dr, mode, ws))
        prof_ref_ch[ch_idx] = np.mean(np.array(stack), axis=0)

    # Trục chuẩn pixel = trung bình ch3 + ch4 (index 2 và 3)
    if num_ch >= 4:
        ref_signal = 0.5 * (prof_ref_ch[2] + prof_ref_ch[3])
        print("  [Shift] Trục chuẩn pixel = trung bình kênh 3 + kênh 4")
    else:
        ref_signal = np.mean(prof_ref_ch, axis=0)
        print(f"  [Shift] Trục chuẩn pixel = trung bình tất cả {num_ch} kênh")

    # Tìm đỉnh trên phổ chuẩn (ref_signal = ch3+ch4) để làm anchor chung
    _, _, ref_peak_pos = _make_roi_from_reference_peak(
        ref_signal,
        search_roi=search_roi,
        smooth_window=smooth_window,
        half_width=100
    )
    print(f"  [Shift] Peak chuẩn (ch3+ch4) = {ref_peak_pos:.2f} px")

    # FIX A: normalize profile về [0,1] trước khi cross-corr
    # → loại bỏ ảnh hưởng của cường độ đèn khác nhau giữa các kênh
    def _normalize_profile(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    ref_signal_norm = _normalize_profile(ref_signal)

    # ── Smooth ref để dùng trong cross-corr ──────────────────────────────────
    _w_sm = min(max(smooth_window, 5), dd - 1)
    if _w_sm % 2 == 0:
        _w_sm -= 1

    shifts = np.zeros(num_ch, dtype=np.float64)
    for ch_idx in range(num_ch):
        ch_prof = prof_ref_ch[ch_idx]
        ch_prof_norm = _normalize_profile(ch_prof)

        # ── Cross-correlation trên ROI HẸP quanh đỉnh ────────────────────────
        # ROI chỉ bao gồm vùng đỉnh (±hw_peak px) + đủ chỗ cho lag tối đa.
        # ROI hẹp giúp cross-corr không bị kéo bởi sườn bất đối xứng và
        # vùng phụ xa đỉnh (như bump 460-490nm).
        # hw_peak=60 đủ bao phủ đỉnh rộng 30nm × 4px/nm = 120px total.
        hw_peak = 60
        roi_s = max(0,      int(ref_peak_pos) - hw_peak - max_shift_px)
        roi_e = min(dd - 1, int(ref_peak_pos) + hw_peak + max_shift_px)

        print(f"    ch{ch_idx+1}: ROI=[{roi_s},{roi_e}] (peak±{hw_peak}+lag±{max_shift_px})")

        lag = find_wavelength_shift(
            ch_prof_norm,
            ref_signal_norm,
            smooth_window=smooth_window,
            search_roi=(roi_s, roi_e),
            max_shift_px=max_shift_px,
        )
        shifts[ch_idx] = lag

    print("  [Shift] shift_px từng kênh (sau shift → trùng trục chuẩn):")
    for i, sh in enumerate(shifts):
        print(f"    ch{i+1}: {sh:+.3f} px")
    return shifts


# ─────────────────────────────────────────────
# 7. AUTO-CALIBRATION
# ─────────────────────────────────────────────

def _gaussian1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode='same')


def _refine_y_center(gray, y_peak, angle_deg, col_start=None, col_end=None,thr_ratio=0.35, margin=10, dr=15, *args, **kwargs):
    if col_start is None or col_end is None:
        col_start, col_end = _detect_col_range(gray, thr_ratio=thr_ratio, margin=margin)
    _, y_mid, _, _ = _detect_channel_line(
        gray, y_peak, col_start, col_end, half_height=28, min_col_intensity=3.0
    )
    return float(y_mid) if y_mid is not None else float(y_peak)

def _detect_col_range(gray,thr_ratio=0.35, margin=10,use_energy=False):
    h, w = gray.shape

    # ===== STEP 1: ROI Y =====
    py_full = np.sum(gray, axis=1).astype(np.float32)
    py_blur = cv2.GaussianBlur(py_full.reshape(-1, 1), (1, 51), 0).flatten()

    y_active = np.where(py_blur > np.max(py_blur) * 0.05)[0]
    if len(y_active) == 0:
        return 0, w - 1

    y_roi_start = max(0, int(y_active[0]) - 20)
    y_roi_end   = min(h - 1, int(y_active[-1]) + 20)

    # ===== STEP 2: ROI X từ ROI Y (QUAN TRỌNG) =====
    region  = gray[y_roi_start:y_roi_end, :]
    profile = np.sum(region, axis=0).astype(np.float32)

    # kernel nhỏ (giống code gốc)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 21), 0).flatten()
    # =========================================
    # FLUORESCENCE ENERGY ROI
    # =========================================

    if use_energy:

        profile = np.clip(profile, 0, None)

        total_energy = np.sum(profile)

        if total_energy <= 0:
            return 0, w - 1

        cumulative = np.cumsum(profile)
        cumulative /= cumulative[-1]

        x_start = int(np.searchsorted(
            cumulative,
            0.0025
        ))

        x_end = int(np.searchsorted(
            cumulative,
            0.9975
        ))

    # =========================================
    # NORMAL THRESHOLD ROI
    # =========================================

    else:

        thr = np.max(profile) * thr_ratio

        active = np.where(profile > thr)[0]

        if len(active) == 0:
            return 0, w - 1

        gaps = np.where(
            np.diff(active) > 80
        )[0]

        clusters = []

        prev = 0

        for g in gaps:

            clusters.append(
                active[prev:g + 1]
            )

            prev = g + 1

        clusters.append(active[prev:])

        clusters = [
            cl for cl in clusters
            if len(cl) > 50
        ]

        if len(clusters) == 0:
            return 0, w - 1

        strengths = [
            profile[cl].sum()
            for cl in clusters
        ]

        main_cl = clusters[
            int(np.argmax(strengths))
        ]

        x_start = int(main_cl[0])
        x_end = int(main_cl[-1])

    x_start = max( 0, int(x_start) - margin ) 
    x_end = min( w - 1, int(x_end) + margin )
    return x_start, x_end

# ============================================================
# PATCH 2: Y PEAK DETECTION
# ============================================================

def _detect_y_centers(gray, n_channels=6, dr=15,thr_ratio=0.35, margin=10,use_energy=False):
    col_start, col_end = _detect_col_range(gray, thr_ratio=thr_ratio, margin=margin, use_energy=use_energy)

    roi_strip = gray[:, col_start:col_end + 1]
    profile_y = np.sum(roi_strip, axis=1).astype(np.float32)
    profile_y = cv2.GaussianBlur(profile_y.reshape(-1, 1), (1, 51), 0).flatten()

    peaks, props = find_peaks(
        profile_y,
        distance=50,
        prominence=np.max(profile_y) * 0.03
    )

    if len(peaks) > n_channels:
        idx = np.argsort(props["prominences"])[-n_channels:]
        peaks = np.sort(peaks[idx])

    return peaks.tolist()


# ============================================================
# PATCH 3: GAUSSIAN CENTROID + FIT
# ============================================================

def _detect_channel_line(gray: np.ndarray,
                         y_peak: int,
                         col_start: int,
                         col_end: int,
                         half_height: int = 28,
                         min_col_intensity: float = 3.0,
                         intensity_ratio: float = 0.15):

    h, w = gray.shape

    x0 = int(np.clip(col_start, 0, w - 1))
    x1 = int(np.clip(col_end,   0, w - 1))
    if x1 < x0:
        x0, x1 = x1, x0

    y0 = max(0, int(y_peak - half_height))
    y1 = min(h, int(y_peak + half_height))

    if y1 - y0 < 3 or x1 - x0 < 3:
        return None, None, None, None

    band = gray[y0:y1, x0:x1 + 1].astype(np.float64)
    ys   = np.arange(band.shape[0], dtype=np.float64)

    col_sums = band.sum(axis=0)
    intensity_threshold = col_sums.max() * intensity_ratio

    cx_list, cy_list = [], []

    for col in range(band.shape[1]):
        col_data = band[:, col]
        total    = col_data.sum()

        if total < intensity_threshold:
            continue

        centroid = np.sum(ys * col_data) / total
        cx_list.append(col + x0)
        cy_list.append(centroid + y0)

    if len(cx_list) < 20:
        return None, None, None, None

    cx = np.array(cx_list, dtype=np.float64)
    cy = np.array(cy_list, dtype=np.float64)

    q1, q3 = np.percentile(cy, 25), np.percentile(cy, 75)
    iqr = q3 - q1
    ok  = (cy >= q1 - 1.5 * iqr) & (cy <= q3 + 1.5 * iqr)
    cx, cy = cx[ok], cy[ok]

    if len(cx) > 50:
        x_min, x_max = cx.min(), cx.max()
        x_range = x_max - x_min
        trim_ok = (cx >= x_min + x_range * 0.05) & (cx <= x_max - x_range * 0.05)
        cx, cy = cx[trim_ok], cy[trim_ok]

    if len(cx) < 10:
        return None, None, None, None

    coeffs = np.polyfit(cx, cy, 1)
    angle_deg = float(np.degrees(np.arctan(coeffs[0])))

    x_mid = float(np.mean(cx))
    y_mid = float(np.polyval(coeffs, x_mid))

    y_pred = np.polyval(coeffs, cx)
    ss_res = float(np.sum((cy - y_pred) ** 2))
    ss_tot = float(np.sum((cy - cy.mean()) ** 2))

    r2  = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
    rms = float(np.sqrt(np.mean((cy - y_pred) ** 2)))
    fit_info = {
        "cx": cx,
        "cy": cy,
        "coeffs": coeffs,
        "r2": r2,
        "rms": rms
    }

    return angle_deg, y_mid, x_mid, fit_info





def _detect_angle_for_channel(gray, y_peak, col_start=None, col_end=None, dr=15, thr_ratio=0.35, margin=10, *args, **kwargs):
    if col_start is None or col_end is None:
        col_start, col_end = _detect_col_range(gray, thr_ratio=thr_ratio, margin=margin)
    angle_deg, _, _, _ = _detect_channel_line(
        gray, y_peak, col_start, col_end, half_height=28, min_col_intensity=3.0
    )
    return float(angle_deg) if angle_deg is not None else 0.0

def run_calibration(ref_path: Union[str, List[str]],
                    dark_path: Optional[Union[str, List[str]]] = None,
                    n_channels: int = 6,
                    pixel_532: Optional[float] = 1912.3,
                    pixel_653: Optional[float] = 2286.0,
                    ref_threshold_pct: float = 0.15,
                    max_shift_px: int = 60,
                    dr: int = 15) -> dict:

    # ================================
    # 1. INPUT PATHS
    # ================================
    if isinstance(ref_path, str):
        ref_paths = [ref_path]
    else:
        ref_paths = list(ref_path)

    if not ref_paths:
        raise ValueError("Cần ít nhất một ảnh reference.")

    # ================================
    # 2. LOAD DARK (giữ nguyên)
    # ================================
    dark_list = None
    if dark_path:
        if isinstance(dark_path, str):
            dark_list = [dark_path]
        else:
            dark_list = [p for p in dark_path if p]
        if not dark_list:
            dark_list = None

    dark_lum = _load_dark_frame(dark_list)

    # ================================
    # 3. LOAD + STACK
    # ================================
    gray_stack = []     # 👈 dùng cho detect (QUAN TRỌNG)
    B_stack = []
    R_stack = []

    h, w = 0, 0

    for p in ref_paths:
        img = load_image(p)
        if img is None:
            continue

        ih, iw = img.shape[:2]
        if not gray_stack:
            h, w = ih, iw
        elif (ih, iw) != (h, w):
            raise ValueError("Ảnh không cùng kích thước")

        # ===== DETECT dùng grayscale RAW =====
        gray_stack.append(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        )

        # ===== CALIBRATION dùng B/R =====
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)

        B_stack.append(img_blur[..., 0].astype(np.float64))
        R_stack.append(img_blur[..., 2].astype(np.float64))

    if not gray_stack:
        raise FileNotFoundError("Không có ảnh hợp lệ")

    n_ref = len(gray_stack)

    print(f"[Calib] Reference: trung bình {n_ref} ảnh")

    # ================================
    # 4. AVERAGE (QUAN TRỌNG NHẤT)
    # ================================
    gray_avg = np.mean(np.stack(gray_stack, axis=0), axis=0).astype(np.uint8)

    B_ch = np.mean(np.stack(B_stack, axis=0), axis=0)
    R_ch = np.mean(np.stack(R_stack, axis=0), axis=0)

    # ================================
    # 5. DETECT (GIỐNG CODE PASTE)
    # ================================
    col_start, col_end = _detect_col_range(gray_avg)
    y_peaks = _detect_y_centers(gray_avg, n_channels, dr)

    if len(y_peaks) < n_channels:
        print(f"⚠️ detect thiếu kênh: {len(y_peaks)}/{n_channels}")

    angles = []
    y_centers = []

    for yp in y_peaks:
        ang = _detect_angle_for_channel(gray_avg, yp, col_start, col_end, dr)
        yc  = _refine_y_center(gray_avg, yp, ang, col_start, col_end, dr)

        angles.append(round(float(ang), 4))
        y_centers.append(round(float(yc), 4))

    print(f"y_peaks    : {y_peaks}")
    print(f"y_centers  : {y_centers}")
    print(f"angles     : {[f'{a:+.3f}' for a in angles]}")
    print(f"col range  : {col_start} -> {col_end}")

    # ================================
    # 6. WAVELENGTH CALIBRATION (GIỮ NGUYÊN)
    # ================================
    dd = col_end - col_start + 1

    ref_ch_indices = [2, 3] if len(y_centers) >= 4 else list(range(len(y_centers)))

    B_ref_profile = np.zeros(dd)
    R_ref_profile = np.zeros(dd)

    for ch_idx in ref_ch_indices:
        yc    = y_centers[ch_idx]
        angle = angles[ch_idx]

        B_rot = _rotate_lum(B_ch, angle)
        R_rot = _rotate_lum(R_ch, angle)

        B_ref_profile += _extract_profile(B_rot, yc, col_start, col_end, dr, 'mean', 3)
        R_ref_profile += _extract_profile(R_rot, yc, col_start, col_end, dr, 'mean', 3)

    B_ref_profile /= len(ref_ch_indices)
    R_ref_profile /= len(ref_ch_indices)

    if pixel_532 is not None and pixel_653 is not None:
        px_532 = float(pixel_532)
        px_653 = float(pixel_653)
    else:
        px_532 = col_start + int(np.argmax(_gaussian1d(B_ref_profile, 5)))
        px_653 = col_start + int(np.argmax(_gaussian1d(R_ref_profile, 5)))

    if px_653 <= px_532:
        raise ValueError("pixel_653 phải > pixel_532")

    calib_a = (653 - 532.5) / (px_653 - px_532)
    calib_b = 532.5 - calib_a * px_532

    wl_start = calib_a * col_start + calib_b
    wl_end   = calib_a * col_end   + calib_b

    print(f"calib_a = {calib_a:.6f}")
    print(f"calib_b = {calib_b:.4f}")

    # ================================
    # 7. SAVE PARAMS
    # ================================
    params = {
        "image_width": w,
        "image_height": h,
        "y_centers": y_centers,
        "angles": angles,
        "col_start": int(col_start),
        "col_end": int(col_end),
        "dr": dr,
        "ws": 3,
        "mode": "mean",
        "wl_min":              round(max(400.0, wl_start - 10), 1),
        "wl_max":              round(min(750.0, wl_end   + 10), 1),
        "calib_a": round(float(calib_a), 6),
        "calib_b": round(float(calib_b), 4),
        "pixel_532": round(px_532, 2),
        "pixel_653": round(px_653, 2),
        "max_shift_px": max_shift_px,
        "ref_threshold_pct": ref_threshold_pct
    }

    with open(_CALIB_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print("✔ Calibration saved")

    return params


def has_calibration() -> bool:
    return os.path.exists(_CALIB_JSON)


# ─────────────────────────────────────────────
# 8. ABSORPTION SPECTRUM
# ─────────────────────────────────────────────

def compute_absorption_spectrum_6ch(ref_paths: List[str],
                                     sample_paths: List[str],
                                     dark_paths: Optional[List[str]] = None) -> List[dict]:
    p = load_params()
    y_centers   = p["y_centers"]
    angles      = p["angles"]
    col_start   = p["col_start"]
    col_end     = p["col_end"]
    dr          = p["dr"]
    ws          = p["ws"]
    mode        = p["mode"]
    wl_min      = p["wl_min"]
    wl_max      = p["wl_max"]
    ref_thr_pct = float(p.get("ref_threshold_pct", 0.1))
    shift_smooth_window  = int(p.get("shift_smooth_window", 51))
    max_shift_px = int(p.get("max_shift_px", 30))
    num_ch      = len(y_centers)

    if num_ch == 0:
        raise ValueError("Chưa có y_centers. Hãy chạy run_calibration() trước.")
    if len(angles) != num_ch:
        raise ValueError(f"Số angles ({len(angles)}) != num_ch ({num_ch}).")

    dark_lum = _load_dark_frame(dark_paths)

    ref_lums: List[np.ndarray] = []
    for path in ref_paths:
        img = load_image(path)
        if img is None:
            continue
        img = cv2.GaussianBlur(img, (3, 3), 0)
        ref_lums.append(_subtract_dark(luminance(img), dark_lum))
    if not ref_lums:
        raise ValueError("Không đọc được ảnh I0 (reference).")

    sam_lums: List[np.ndarray] = []
    for path in sample_paths:
        img = load_image(path)
        if img is None:
            continue
        img = cv2.GaussianBlur(img, (3, 3), 0)
        sam_lums.append(_subtract_dark(luminance(img), dark_lum))
    if not sam_lums:
        raise ValueError("Không có ảnh mẫu hợp lệ.")

    n_pairs = min(len(ref_lums), len(sam_lums))
    if len(ref_lums) != len(sam_lums):
        print(f"⚠️ Số ảnh ref ({len(ref_lums)}) ≠ sample ({len(sam_lums)}). "
              f"Chỉ dùng {n_pairs} cặp đầu.")

    h, w = ref_lums[0].shape
    col_end_eff = min(col_end, w - 1)
    dd = col_end_eff - col_start + 1

    if col_start < 0 or col_start >= w:
        raise ValueError(f"col_start={col_start} nằm ngoài ảnh (width={w}).")
    if dd <= 0:
        raise ValueError(f"Vùng phổ không hợp lệ: col_start={col_start}, "
                         f"col_end={col_end}, width={w}")
    # ── BƯỚC 1: Tìm shift pixel cho từng kênh (so với trục chuẩn ch3+ch4) ──
    channel_shifts = _estimate_channel_shifts_from_ref_profiles(
        ref_lums=ref_lums,
        y_centers=y_centers,
        angles=angles,
        col_start=col_start,
        col_end_eff=col_end_eff,
        dr=dr,
        mode=mode,
        ws=ws,
        smooth_window=shift_smooth_window,
        search_roi=None,
        max_shift_px=max_shift_px,
    )

    # ── BƯỚC 2: Trục bước sóng dùng chung (1 lần) ───────────────────────────
    wavelengths_full = pixels_to_wavelengths_unified(dd, pixel_offset=col_start, params=p)
    calib_a = float(p["calib_a"])  # nm/pixel

    results = []
    for ch_idx in range(num_ch):
        y_c   = float(y_centers[ch_idx])
        angle = float(angles[ch_idx])

        A_stack: List[np.ndarray] = []
        profile_ref_last = profile_sam_last = None

        # Tầng 1: shift trục wavelength từ phổ đèn (quang học cố định)
        sh_nm = float(channel_shifts[ch_idx]) * calib_a
        wl_ch = wavelengths_full - sh_nm
        wl_ch  = wl_ch + float(p.get("wl_offset_nm", 0.0)) # offset chung cho tất cả kênh nếu cần hiệu chỉnh thêm
        print(f"    ch{ch_idx+1}: shift={channel_shifts[ch_idx]:+.3f}px = {sh_nm:+.3f}nm "
              f"→ wl_ch[0]={wl_ch[0]:.2f}nm")

        for k in range(n_pairs):
            I_ref_rot = _rotate_lum(ref_lums[k], angle)
            I_sam_rot = _rotate_lum(sam_lums[k], angle)

            profile_ref = _extract_profile(I_ref_rot, y_c, col_start, col_end_eff,
                                           dr, mode, ws)
            profile_sam = _extract_profile(I_sam_rot, y_c, col_start, col_end_eff,
                                           dr, mode, ws)

            A_k = _masked_absorbance(profile_ref, profile_sam, ref_thr_pct)
            A_stack.append(A_k)
            profile_ref_last = profile_ref
            profile_sam_last = profile_sam

        with np.errstate(all='ignore'):
            A = np.nanmean(np.array(A_stack), axis=0)
        A = np.where(np.isfinite(A), A, np.nan)
        #smooth đồ thị — FIX noise: tăng window 51→71, poly 2→3
        # poly=3 giữ hình dạng peak tốt hơn khi window lớn
        mask = np.isfinite(A)
        A[mask] = _savgol_smooth_1d(A[mask], window=89, poly=3)
        wl_out, A_out = _filter_spectrum_by_range(A, wl_ch, wl_min, wl_max)

        # Tìm peak trong ROI hẹp quanh trung tâm phổ (tránh peak giả ở rìa)
        peak_wl = peak_val = None
        if len(A_out) > 0 and np.any(np.isfinite(A_out)):
            # Rough center: argmax toàn phổ chỉ dùng để xác định vùng tìm kiếm
            A_fin = np.where(np.isfinite(A_out), A_out, -np.inf)
            rough_pk   = int(np.argmax(A_fin))
            rough_wl   = float(wl_out[rough_pk])
            # Tìm chính xác trong ROI ±15nm quanh rough center
            peak_search_half = float(p.get("peak_search_half_nm", 15.0))
            pw = _find_peak_in_roi(wl_out, A_out, rough_wl, peak_search_half)
            if pw is not None:
                peak_wl  = pw
                peak_val = float(np.interp(pw, wl_out, A_out))

        # Tính SEM của absorbance tại từng pixel (trước khi filter dải wl)
        # SEM = std / sqrt(n), tính trên A_stack (các lần đo riêng lẻ)
        if len(A_stack) > 1:
            A_stack_arr = np.array(A_stack)  # shape: (n_pairs, n_pixels)
            A_std = np.nanstd(A_stack_arr, axis=0, ddof=1)
            A_sem_full = A_std / np.sqrt(np.sum(np.isfinite(A_stack_arr), axis=0).clip(1))
        else:
            A_sem_full = np.zeros(len(A_stack[0]))

        # Filter SEM theo cùng dải wl như A_out
        _, A_sem_out = _filter_spectrum_by_range(A_sem_full, wl_ch, wl_min, wl_max)

        results.append({
            "channel":              ch_idx + 1,
            "y_center":             y_c,
            "angle":                angle,
            "wavelengths":          wl_out,
            "absorbance":           A_out,
            "absorbance_sem":       A_sem_out,
            "profile_ref":          profile_ref_last,
            "profile_sam":          profile_sam_last,
            "wavelength_shift_px":  float(channel_shifts[ch_idx]),
            "peak_offset_nm":       0.0,
            "peak_wavelength":      peak_wl,
            "peak_value":           peak_val,
        })

    if sum(len(r["wavelengths"]) for r in results) == 0:
        raise ValueError(
            "Không có dữ liệu phổ trong dải bước sóng đã chọn. "
            "Hãy chạy lại hiệu chỉnh thiết bị."
        )

    # ── BƯỚC 3: Per-measurement anchor correction ────────────────────────────
    # Bù lệch đỉnh còn lại do đèn "trôi" trong từng lần đo (không phải hằng số
    # thiết bị — phần đó đã được xử lý bởi channel_peak_offsets_nm ở bước trên).
    # Dùng đỉnh ch3+ch4 của chính lần đo này làm anchor tuyệt đối.
    anchor_max_corr = float(p.get("anchor_max_correction_nm", 10))
    results = _apply_anchor_correction(
        results,
        ref_ch_indices=(2, 3),
        max_correction_nm=anchor_max_corr,
    )

    # ── BƯỚC 4: Hiệu chuẩn tuyến tính với máy chuẩn ────────────────────────────
    # A_cal = a_i * A_raw + b_i per-channel. Bỏ qua nếu chưa set_calib_scale().
    results = _apply_calib_scale(results, p)

    # ── BƯỚC 5: Chuẩn hóa baseline giữa các kênh ────────────────────────────────
    # Loại bỏ intercept lệch do I0 không đồng đều giữa 6 cuvette blank.
    # Dùng ch3+ch4 làm reference baseline, trừ offset tuyệt đối cho từng kênh.
    results = _normalize_intercept_between_channels(
        results,
        ref_ch_indices=(2, 3),
        baseline_offset_nm=100.0,   # lấy baseline tại peak_wl - 100nm
    )

    # ── In bảng đỉnh hấp thụ từng kênh (phục vụ fit A_hệ ~ A_chuẩn) ──────────
    ref_peak_wls = [results[i]["peak_wavelength"] for i in [2, 3]
                    if i < len(results) and results[i]["peak_wavelength"] is not None]
    ref_peak_wl  = float(np.mean(ref_peak_wls)) if ref_peak_wls else None

    print("\n" + "─" * 56)
    print("  HẤP THỤ TẠI BƯỚC SÓNG CHUẨN")
    if ref_peak_wl is not None:
        print(f"  Bước sóng chuẩn (ch3+ch4): {ref_peak_wl:.2f} nm")
    print("─" * 56)
    print(f"  {'Kênh':<6} {'A tại chuẩn':>12}  {'SEM':>10}")
    print(f"  {'-'*34}")
    for r in results:
        ch  = r["channel"]
        wl  = r["wavelengths"]
        A   = r["absorbance"]
        sem = r.get("absorbance_sem", np.zeros(len(A)))
        if ref_peak_wl is not None and len(wl) > 0:
            a_at_ref   = float(np.interp(ref_peak_wl, wl, A))
            sem_at_ref = float(np.interp(ref_peak_wl, wl, sem))
            print(f"  ch{ch:<4} {a_at_ref:>12.4f}  {sem_at_ref:>10.4f}")
        else:
            print(f"  ch{ch:<4} {'N/A':>12}  {'N/A':>10}")
    print("─" * 56)
    print("  → Dùng cột này để fit: A_chuẩn = f(A_hệ)")
    print("─" * 56 + "\n")

    return results

def _apply_calib_scale_fluo(results: List[dict], params: dict) -> List[dict]:
    a_list = params.get("calib_scale_a_fluo", [])
    b_list = params.get("calib_scale_b_fluo", [])
    if not a_list or not b_list:
        return results

    print("  [CalibScale Fluo] Áp dụng F_cal = a*F_raw + b (scale theo đỉnh):")
    for ch_idx, r in enumerate(results):
        if ch_idx >= len(a_list) or ch_idx >= len(b_list):
            print(f"    ch{ch_idx+1}: không có hệ số → bỏ qua")
            continue
        a = float(a_list[ch_idx])
        b = float(b_list[ch_idx])
        F = np.asarray(r["intensity"], dtype=np.float64)

        # Tính giá trị đỉnh raw và đỉnh target theo Avantes
        peak_idx = int(np.argmax(F))
        peak_raw = float(F[peak_idx])
        peak_target = a * peak_raw + b  # giá trị đỉnh đúng theo Avantes

        # Scale toàn phổ theo tỉ lệ: đỉnh đúng, phổ không âm
        if peak_raw > 1e-12:
            # Ước tính baseline từ vùng đầu và cuối phổ (không có tín hiệu)
            baseline = float(np.mean(np.concatenate([F[:10], F[-10:]])))
            
            F_zeroed = F - baseline              # đưa noise về ~0
            peak_zeroed = peak_raw - baseline    # đỉnh thực sự so với baseline
            
            if peak_zeroed > 1e-12:
                F_cal = F_zeroed * (peak_target / peak_zeroed)
            else:
                F_cal = F_zeroed.copy()
        else:
            F_cal = F.copy()

        r["intensity"] = F_cal

        # Cập nhật peak_value
        pk_wl = r.get("peak_wavelength")
        wl = np.asarray(r["wavelengths"], dtype=np.float64)
        if pk_wl is not None and len(wl) > 0:
            r["peak_value"] = float(np.interp(pk_wl, wl, r["intensity"]))
        print(f"    ch{ch_idx+1}: peak_raw={peak_raw:.2f} → peak_target={peak_target:.2f}  "
              f"(a={a:+.4f}, b={b:+.4f})")
    return results

def compute_fluorescence_spectrum_6ch(
        image_paths: List[str],
        dark_paths: Optional[List[str]] = None
) -> List[dict]:

    # =========================================
    # LOAD GLOBAL WAVELENGTH CALIBRATION
    # =========================================

    p = load_params()

    calib_a = p.get("calib_a")
    calib_b = p.get("calib_b")
    a_list = p.get("calib_scale_a_fluo", [])
    b_list = p.get("calib_scale_b_fluo", [])

    if calib_a is None or calib_b is None:
        raise ValueError(
            "Chưa có wavelength calibration."
        )

    # =========================================
    # LOAD DARK
    # =========================================

    dark_lum = _load_dark_frame(dark_paths)

    # =========================================
    # LOAD FLUORESCENCE IMAGES
    # =========================================

    sam_lums = []

    for path in image_paths:

        img = load_image(path)

        if img is None:
            continue

        img = cv2.GaussianBlur(
            img,
            (3, 3),
            0
        )

        lum = luminance(img)

        lum = _subtract_dark(
            lum,
            dark_lum
        )

        sam_lums.append(lum)

    if len(sam_lums) == 0:
        raise ValueError(
            "Không load được fluorescence images."
        )

    # =========================================
    # MEAN IMAGE
    # =========================================

    lum_mean = np.mean(
        np.stack(sam_lums, axis=0),
        axis=0
    )

    h, w = lum_mean.shape

    # =========================================
    # DETECT CHANNELS
    # =========================================

    y_centers = _detect_y_centers(
        lum_mean,
        n_channels=6,
        dr=15,
        thr_ratio=0.2,
        margin=40,
        use_energy=True,
    )

    # =========================================
    # REMOVE GHOST CHANNELS
    # =========================================

    if len(y_centers) > 0:

        scores = []

        for yc in y_centers:

            y0 = max(0, int(yc - 25))
            y1 = min(h, int(yc + 25))

            score = np.sum(
                lum_mean[y0:y1, :]
            )

            scores.append(score)

        scores = np.array(scores)

        mx = np.max(scores)

        if mx > 0:
            scores /= mx
        else:
            scores = np.zeros_like(scores)

        valid_idx = np.where(
            scores > 0.20
        )[0]

        y_centers = [
            y_centers[i]
            for i in valid_idx
        ]

    if len(y_centers) == 0:
        raise ValueError(
            "Không detect được fluorescence channels."
        )

    # =========================================
    # DETECT ROI
    # =========================================

    col_start, col_end = _detect_col_range(
        lum_mean,
        margin=40,
        use_energy=True,
    )

    # =========================================
    # WAVELENGTH AXIS
    # =========================================

    pixels = np.arange(
        col_start,
        col_end + 1,
        dtype=np.float64
    )

    wavelengths_full = (
        float(calib_a) * pixels
        + float(calib_b)
    )
    wavelengths_full = wavelengths_full + float(p.get("wl_offset_nm", 0.0)) # offset chung nếu cần hiệu chỉnh thêm

    # =========================================
    # VALID WL DEBUG
    # =========================================

    valid_dbg = (
        (wavelengths_full >= 450) &
        (wavelengths_full <= 650)
    )

    wl_dbg = wavelengths_full[valid_dbg]

    print("=" * 60)
    print("[FLUORESCENCE DEBUG]")

    print("[Fluo] y_centers:", y_centers)

    print(
        f"[Fluo] col range: "
        f"{col_start} → {col_end}"
    )

    print(
        f"[Fluo WL] "
        f"{np.min(wavelengths_full):.1f}"
        f" → "
        f"{np.max(wavelengths_full):.1f} nm"
    )

    if len(wl_dbg) > 0:

        print(
            f"[Fluo VALID WL] "
            f"{np.min(wl_dbg):.1f}"
            f" → "
            f"{np.max(wl_dbg):.1f} nm"
        )

    print("=" * 60)

    # =========================================
    # PROCESS CHANNELS
    # =========================================

    results = []

    angles = []

    for ch_idx in range(len(y_centers)):

        y_c = int(y_centers[ch_idx])

        angle = _detect_angle_for_channel(
            lum_mean,
            y_c,
            col_start,
            col_end,
            dr=15,
        )

        angles.append(angle)

        prof_stack = []

        # =====================================
        # PROCESS ALL FRAMES
        # =====================================

        for lum_sam in sam_lums:

            I_rot = _rotate_lum(
                lum_sam,
                angle
            )

            y_refined = _refine_y_center(
                I_rot,
                y_c,
                angle,
                col_start,
                col_end,
            )

            profile = _extract_profile(
                I_rot,
                y_refined,
                col_start,
                col_end,
                dr=15,
                mode="mean",
                ws=3,
            )

            prof_stack.append(profile)

        # =====================================
        # AVERAGE SPECTRUM
        # =====================================

        prof_arr = np.array(prof_stack)  # shape: (n_frames, n_pixels)

        profile_mean = np.mean(prof_arr, axis=0)

        # =====================================
        # SMOOTH SPECTRUM
        # =====================================

        if len(profile_mean) > 21:

            try:

                profile_mean = savgol_filter(
                    profile_mean,
                    window_length=21,
                    polyorder=2
                )

            except Exception as e:

                print(
                    "[Fluo] smooth error:",
                    e
                )

        # =====================================
        # LIMIT VALID WL RANGE
        # =====================================

        valid = (
            (wavelengths_full >= 450) &
            (wavelengths_full <= 650)
        )

        wl_out = wavelengths_full[valid]
        F_out = profile_mean[valid]

        if len(wl_out) == 0:
            continue

        # =====================================
        # PEAK
        # =====================================

        peak_wl = None
        peak_val = None
        peak_sem = None

        if len(F_out) > 0:

            pk = int(np.argmax(F_out))

            peak_wl = float(wl_out[pk])
            peak_val = float(F_out[pk])

            # SEM tính trên giá trị raw (trước smooth) tại pixel đỉnh
            valid_indices = np.where(valid)[0]
            pk_global = valid_indices[pk]
            if len(prof_stack) > 1:
                peak_vals_raw = prof_arr[:, pk_global]
                peak_sem = float(np.std(peak_vals_raw, ddof=1) / np.sqrt(len(prof_stack)))
            else:
                peak_sem = 0.0

        print(
            f"[Fluo CH {ch_idx+1}] "
            f"Peak: {peak_wl:.2f} nm "
            f"| Intensity: {peak_val:.2f}"
        )

        # =====================================
        # SAVE RESULT
        # =====================================

        results.append({

            "channel": ch_idx + 1,

            "y_center": y_c,

            "angle": angle,

            "wavelengths": wl_out,

            "intensity": F_out,

            "peak_wavelength": peak_wl,

            "peak_value": peak_val,

            "peak_sem": peak_sem,
        })

    # =========================================
    # PRINT ANGLES
    # =========================================

    print("[Fluo] angles:", angles)

    # =========================================
    # FINAL CHECK
    # =========================================

    total_points = sum(
        len(r["wavelengths"])
        for r in results
    )

    if total_points == 0:

        raise ValueError(
            "Không có dữ liệu fluorescence hợp lệ."
        )

    # APPLY AVANTES CALIBRATION SCALE
    results = _apply_calib_scale_fluo(results, p)

    print("\n" + "─" * 56)
    print("  HUỲNH QUANG TẠI ĐỈNH (SAU CALIB)")
    print("─" * 56)
    print(f"  {'Kênh':<6} {'Peak (nm)':>10} {'F tại đỉnh':>12}  {'SEM':>10}")
    print(f"  {'-'*44}")
    for r in results:
        ch     = r["channel"]
        pk_wl  = r.get("peak_wavelength")
        pk_val = r.get("peak_value")
        sem    = r.get("peak_sem")
        if pk_wl is not None:
            sem_str = f"{sem:>10.2f}" if sem is not None else f"{'N/A':>10}"
            print(f"  ch{ch:<4} {pk_wl:>10.2f} {pk_val:>12.2f}  {sem_str}")
        else:
            print(f"  ch{ch:<4} {'N/A':>10} {'N/A':>12}  {'N/A':>10}")
    print("─" * 56 + "\n")

    return results