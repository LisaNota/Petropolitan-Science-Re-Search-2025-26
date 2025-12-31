import os
import cv2
import numpy as np
from src.features.sky_mask import sky_mask
from src.features.exg_gradmag import exg, gradmag
from src.features.fol_ground import foliage_mask_autumn, ground_warm_mask_autumn, foliage_mask_summer, ground_grass_mask_summer
from src.features.trunk_mask import trunk_mask
from src.features.protect_bright import protect_bright, feather_alpha_sky
from src.features.recoloring import recolor_hsv_region, map_green_to_autumn, blend, degreen_trunks

PHOTO1_PATH = "data/task-2/Photo1.jpg"  # осень
PHOTO2_PATH = "data/task-2/photo-summer.jpg"  # лето

OUT_SUMMER = "Summer.jpg"  # Photo1 -> Summer
OUT_AUTUMN = "Autumn.jpg"  # Photo2 -> Autumn


def process_images(photo1_path: str = PHOTO1_PATH, photo2_path: str = PHOTO2_PATH) -> None:
    """Основная функция для обработки"""
    p1 = cv2.imread(photo1_path, cv2.IMREAD_COLOR)  # осень
    p2 = cv2.imread(photo2_path, cv2.IMREAD_COLOR)  # лето

    sky1 = sky_mask(p1)
    sky2 = sky_mask(p2)

    g1 = exg(p1)
    g2 = exg(p2)

    gray1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gm1 = gradmag(gray1)
    gm2 = gradmag(gray2)

    tr1 = trunk_mask(p1, sky1)
    tr2 = trunk_mask(p2, sky2)

    fol1 = foliage_mask_autumn(p1, sky1, tr1)
    gnd1 = ground_warm_mask_autumn(p1, sky1, tr1)

    fol2 = foliage_mask_summer(p2, sky2, tr2)
    gnd2 = ground_grass_mask_summer(p2, sky2, tr2)

    alpha_summer = np.zeros(p1.shape[:2], dtype=np.float32)
    alpha_summer[sky1] = 0.0
    a_t2 = np.clip(cv2.GaussianBlur(
        (tr2 > 0).astype(np.float32), (19, 19), 0), 0.0, 1.0)
    a_t2[sky2] = 0.0
    a_t2 = protect_bright(a_t2, p2)  # доп. защита ярких облаков

    alpha_autumn = np.clip(a_t2, 0.0, 1.0)

    step = p1.copy()

    # 1) крона -> зелень
    step1 = recolor_hsv_region(
        step, fol1, p2, fol2,
        target_h_range=(38.0, 56.0),
        sv_strength=0.85,
        clamp_s=(15, 220),
        clamp_v=(10, 240),
    )

    # 2) земля -> трава
    step2 = recolor_hsv_region(
        step1, gnd1, p2, gnd2,
        target_h_range=(38.0, 70.0),
        sv_strength=0.90,
        clamp_s=(10, 200),
        clamp_v=(10, 245),
    )

    # 3) альфа
    a_f = protect_bright(feather_alpha_sky(
        fol1, sky1, radius=27, erode_px=4), p1)
    a_g = protect_bright(feather_alpha_sky(
        gnd1, sky1, radius=31, erode_px=6), p1)
    alpha_summer = np.clip(a_f + a_g * 0.95, 0.0, 1.0)

    # 4) финальный blend
    summer_img = blend(p1, step2, alpha_summer)

    stepA = p2.copy()

    # 1) крона: зелёный -> осень
    stepA1 = map_green_to_autumn(
        stepA, fol2, p1, fol1, out_range=(10.0, 32.0), sv_strength=0.85)

    # 2) земля/трава: зелёный -> осень
    stepA2 = map_green_to_autumn(
        stepA1, gnd2, p1, gnd1, out_range=(12.0, 30.0), sv_strength=0.85)

    # 3) стволы
    autumn_img = degreen_trunks(stepA2, tr2, warmth=0.55)

    # 4) альфа
    a_f2 = protect_bright(feather_alpha_sky(
        fol2, sky2, radius=25, erode_px=2), p2)
    a_g2 = protect_bright(feather_alpha_sky(
        gnd2, sky2, radius=29, erode_px=3), p2)
    a_t2 = np.clip(cv2.GaussianBlur(
        (tr2 > 0).astype(np.float32), (19, 19), 0), 0.0, 1.0)
    a_t2[sky2] = 0.0

    alpha_autumn = np.clip(a_f2 + a_g2 * 0.95 + a_t2, 0.0, 1.0)

    cv2.imwrite(OUT_SUMMER, summer_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(OUT_AUTUMN, autumn_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print("  saved ->", os.path.abspath(OUT_SUMMER))
    print("  saved ->", os.path.abspath(OUT_AUTUMN))


if __name__ == "__main__":
    main()
