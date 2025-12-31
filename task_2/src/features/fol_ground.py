import cv2
import numpy as np
from src.features.exg_gradmag import exg, gradmag
from src.features.fill_holes import fill_holes, remove_small


def foliage_mask_summer(img_bgr: np.ndarray, sky: np.ndarray, trunk_u8: np.ndarray) -> np.ndarray:
    """Летняя листва (верх кадра), исключая землю и стволы."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    g = exg(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gmag = gradmag(gray)

    h, w = img_bgr.shape[:2]
    upper = np.zeros((h, w), dtype=bool)
    upper[: int(0.72 * h), :] = True

    green_h = (H >= 35) & (H <= 95)
    veg = ((g > 30) & (S > 25) & (V > 30) & green_h) | (
        (g > 45) & (S > 20) & (V > 25))
    veg = veg & (gmag > 10) & upper & (~sky) & (~(trunk_u8 > 0))

    m = (veg.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    m = fill_holes(m)
    m = remove_small(m, min_area=4000)
    return m


def foliage_mask_autumn(img_bgr: np.ndarray, sky: np.ndarray, trunk_u8: np.ndarray) -> np.ndarray:
    """Осенняя листва (крона), исключая землю и стволы."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    g = exg(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gmag = gradmag(gray)

    h, w = img_bgr.shape[:2]
    upper = np.zeros((h, w), dtype=bool)
    upper[: int(0.75 * h), :] = True

    autumn = ((H <= 55) | (H >= 160)) & (S >= 55) & (V >= 45) & (g < 40)
    autumn = autumn & upper & (~sky) & (~(trunk_u8 > 0)) & (gmag > 8)

    m = (autumn.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (17, 17)), iterations=2)
    m = fill_holes(m)
    m = remove_small(m, min_area=5000)
    return m


def ground_warm_mask_autumn(img_bgr: np.ndarray, sky: np.ndarray, trunk_u8: np.ndarray) -> np.ndarray:
    """Тёплая земля/ковёр листьев (нижняя часть) на осеннем фото."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = exg(img_bgr)

    h, w = img_bgr.shape[:2]
    bottom = np.zeros((h, w), dtype=bool)
    bottom[int(0.58 * h):, :] = True

    warm = (H <= 45) & (S >= 40) & (V >= 35) & (g < 45)
    m = warm & bottom & (~sky) & (~(trunk_u8 > 0))

    mu8 = (m.astype(np.uint8) * 255)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (21, 21)), iterations=2)
    mu8 = fill_holes(mu8)
    mu8 = remove_small(mu8, min_area=6000)
    return mu8


def ground_grass_mask_summer(img_bgr: np.ndarray, sky: np.ndarray, trunk_u8: np.ndarray) -> np.ndarray:
    """Зелёная трава (нижняя часть) на летнем фото."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = exg(img_bgr)

    h, w = img_bgr.shape[:2]
    bottom = np.zeros((h, w), dtype=bool)
    bottom[int(0.55 * h):, :] = True

    grass = (g > 35) & (S > 20) & (V > 25) & (H >= 35) & (H <= 95)
    m = grass & bottom & (~sky) & (~(trunk_u8 > 0))

    mu8 = (m.astype(np.uint8) * 255)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (25, 25)), iterations=2)
    mu8 = fill_holes(mu8)
    mu8 = remove_small(mu8, min_area=6000)
    return mu8
