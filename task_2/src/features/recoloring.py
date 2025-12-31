import cv2
import numpy as np


def match_mean_std(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    mx, sx = float(x.mean()), float(x.std())
    my, sy = float(y.mean()), float(y.std())
    sx = max(sx, 1e-6)
    sy = max(sy, 1e-6)
    return (x - mx) / sx * sy + my


def circ_mean_h(h: np.ndarray) -> float:
    """Круговое среднее Hue (0..179) -> 0..180."""
    h = h.astype(np.float32)
    ang = h / 180.0 * 2.0 * np.pi
    a = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    if a < 0:
        a += 2.0 * np.pi
    return a / (2.0 * np.pi) * 180.0


def recolor_hsv_region(
    src_bgr: np.ndarray,
    src_mask_u8: np.ndarray,
    ref_bgr: np.ndarray,
    ref_mask_u8: np.ndarray,
    target_h_range: tuple[float, float],
    sv_strength: float,
    clamp_s: tuple[int, int],
    clamp_v: tuple[int, int],
) -> np.ndarray:
    """
    Перекраска области по маске:
    - Hue сдвигаем к среднему Hue референса и затем клипуем в target_h_range
    - S,V матчим по mean/std к референсу и смешиваем с исходными через sv_strength
    """
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    sm = (src_mask_u8 > 0)
    rm = (ref_mask_u8 > 0)
    if sm.sum() < 500 or rm.sum() < 500:
        return src_bgr.copy()

    H, S, V = src_hsv[..., 0], src_hsv[..., 1], src_hsv[..., 2]
    Hr, Sr, Vr = ref_hsv[..., 0], ref_hsv[..., 1], ref_hsv[..., 2]

    h_src = H[sm].copy()
    h_un = h_src.copy()
    h_un[h_un >= 160] -= 180.0  # разворот красных хвостов

    mu_src = float(h_un.mean())
    mu_ref = float(circ_mean_h(Hr[rm]))

    h_new = h_un + (mu_ref - mu_src)
    lo, hi = target_h_range
    h_new = np.clip(h_new, lo, hi)
    H[sm] = (h_new % 180.0)

    S_new = match_mean_std(S[sm], Sr[rm])
    V_new = match_mean_std(V[sm], Vr[rm])

    S[sm] = np.clip((1.0 - sv_strength) * S[sm] +
                    sv_strength * S_new, clamp_s[0], clamp_s[1])
    V[sm] = np.clip((1.0 - sv_strength) * V[sm] +
                    sv_strength * V_new, clamp_v[0], clamp_v[1])

    out = cv2.cvtColor(np.clip(src_hsv, 0, 255).astype(
        np.uint8), cv2.COLOR_HSV2BGR)
    return out


def map_green_to_autumn(
    src_bgr: np.ndarray,
    mask_u8: np.ndarray,
    ref_bgr: np.ndarray,
    ref_mask_u8: np.ndarray,
    out_range: tuple[float, float],
    sv_strength: float,
) -> np.ndarray:
    """Маппинг зелёных Hue (28..105) -> осенний диапазон (out_range) + матчинг S,V к осени."""
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    sm = (mask_u8 > 0)
    rm = (ref_mask_u8 > 0)
    if sm.sum() < 500 or rm.sum() < 500:
        return src_bgr.copy()

    H, S, V = src_hsv[..., 0], src_hsv[..., 1], src_hsv[..., 2]
    Hr, Sr, Vr = ref_hsv[..., 0], ref_hsv[..., 1], ref_hsv[..., 2]

    Hm = H[sm]
    in_green = (Hm >= 28) & (Hm <= 105)

    H_new = Hm.copy()
    lo, hi = out_range
    H_new[in_green] = lo + (Hm[in_green] - 28.0) / (105.0 - 28.0) * (hi - lo)

    if in_green.sum() > 100:
        H_new[in_green] += (Hm[in_green] - float(Hm[in_green].mean())) * 0.05

    H[sm] = np.clip(H_new, 0, 179)

    S_new = match_mean_std(S[sm], Sr[rm])
    V_new = match_mean_std(V[sm], Vr[rm])

    S[sm] = np.clip((1.0 - sv_strength) * S[sm] + sv_strength * S_new, 10, 255)
    V[sm] = np.clip((1.0 - sv_strength) * V[sm] + sv_strength * V_new, 5, 250)

    out = cv2.cvtColor(np.clip(src_hsv, 0, 255).astype(
        np.uint8), cv2.COLOR_HSV2BGR)
    return out


def degreen_trunks(img_bgr: np.ndarray, trunk_u8: np.ndarray, warmth: float) -> np.ndarray:
    """Убираем зеленоватый оттенок со стволов + чуть “согреваем” (коричневатость)."""
    m = (trunk_u8 > 0)
    out = img_bgr.astype(np.float32).copy()
    b, g, r = cv2.split(out)

    gb = (r + b) / 2.0
    g[m] = 0.45 * g[m] + 0.55 * gb[m]

    r[m] = np.clip(r[m] * (1.0 + 0.20 * warmth) + g[m] * 0.05 * warmth, 0, 255)
    b[m] = np.clip(b[m] * (1.0 - 0.15 * warmth), 0, 255)

    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def blend(base_bgr: np.ndarray, changed_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Смешивание исходных тонов с измененными"""
    out = changed_bgr.astype(
        np.float32) * alpha[..., None] + base_bgr.astype(np.float32) * (1.0 - alpha[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)
