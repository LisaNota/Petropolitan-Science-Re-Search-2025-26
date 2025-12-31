import cv2
import numpy as np


def feather_alpha_sky(mask_u8: np.ndarray, sky_bool: np.ndarray, radius: int, erode_px: int) -> np.ndarray:
    """
    Плавная альфа из маски, но:
    - перед blur делаем erode, чтобы не “залезать” на небо
    - alpha=0 на небе жёстко
    """
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if erode_px > 0:
        k = erode_px * 2 + 1
        m = cv2.erode(m, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)), iterations=1)

    k = max(3, (radius // 2) * 2 + 1)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (k, k), 0)
    a = np.clip(a, 0.0, 1.0)
    a[sky_bool] = 0.0
    return a


def protect_bright(alpha: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
    """Защита ярких “дыр” неба/облаков (чтобы не окрашивались)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    protect = (V > 210) & (S < 70)
    out = alpha.copy()
    out[protect] = 0.0
    return out
