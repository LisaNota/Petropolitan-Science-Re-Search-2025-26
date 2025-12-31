import cv2
import numpy as np


def sky_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Строит булеву маску неба/облаков для данного изображения.

    Идея:
    - небо и облака **нельзя** перекрашивать при сезонном переносе цвета,
      поэтому они выделяются в отдельную маску.

    Эвристика:
    - анализируем кадр (60% сверху)
    - в HSV:
        * "blue sky": Hue в диапазоне [80..140], V > 60, S > 25
        * "clouds": V > 180, S < 90  (ярко и слабо насыщенно)

    Parameters
    ----------
    img_bgr : np.ndarray
        Входное изображение в BGR (OpenCV), форма (H, W, 3).

    Returns
    -------
    np.ndarray
        Булева маска формы (H, W):
        True  -> пиксель относится к небу/облакам
        False -> небо/облака отсутствуют
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h, w = img_bgr.shape[:2]

    top = np.zeros((h, w), dtype=bool)
    top[: int(0.60 * h), :] = True

    blue = top & (H >= 80) & (H <= 140) & (V > 60) & (S > 25)
    clouds = top & (V > 180) & (S < 90)
    return blue | clouds
