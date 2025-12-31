import cv2
import numpy as np
from .exg_gradmag import gradmag
from .fill_holes import fill_holes, remove_small


def trunk_mask(img_bgr: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """
    Строит маску стволов/веток (древесных частей), чтобы защитить их от перекраски листвы.

    Эвристика, стволы/ветки часто:
        * менее насыщенные (низкий S)
        * умеренно тёмные (V не слишком высокий)
        * имеют выраженную структуру (градиент/границы)

    Parameters
    ----------
    img_bgr : np.ndarray
        Входное BGR изображение, форма (H, W, 3).
    sky : np.ndarray
        Булева маска неба (H, W), True = небо.
        Небо исключается из кандидатов.

    Returns
    -------
    np.ndarray
        Маска стволов uint8 (H, W) 0/255.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gmag = gradmag(gray)

    cand = (~sky) & (S < 80) & (V < 165) & (V > 20) & (gmag > 20)
    m = (cand.astype(np.uint8) * 255)

    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, vk, iterations=1)
    m = cv2.dilate(m, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

    m = fill_holes(m)
    m[sky] = 0
    m = remove_small(m, min_area=1500)
    return m
