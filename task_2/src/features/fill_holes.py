import cv2
import numpy as np


def fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    """
    Заполняет полости внутри бинарной маски.

    Зачем:
    - в масках кроны часто остаются отверстия (просветы неба, шум),
      которые приводят к "пятнам" после перекраски.
    - заливка полостей делает область объекта более цельной.

    Метод:
    - инвертируется маска
    - flood fill от (0,0) по фону
    - оставшееся после flood fill — это полости внутри объекта
    - объединяется исходная маска с найденными полостями

    Parameters
    ----------
    mask_u8 : np.ndarray
        Входная маска формы (H, W)
        Активные пиксели считаются там, где mask_u8 > 0.

    Returns
    -------
    np.ndarray
        Маска uint8 формы (H, W), значения 0/255, с заполненными внутренними полостями.
    """
    m = (mask_u8 > 0).astype(np.uint8) * 255
    inv = cv2.bitwise_not(m)
    h, w = m.shape

    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 0)

    holes = cv2.bitwise_not(ff)
    filled = cv2.bitwise_or(m, holes)
    return filled


def remove_small(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """
    Удаляет мелкие связные компоненты из бинарной маски по порогу площади.
    После порогов Hue/ExG/градиента остаются "островки" шума их лучше убрать, 
    чтобы перекраска не давала случайных точек

    Parameters
    ----------
    mask_u8 : np.ndarray
        Входная маска (H, W) uint8, активна там, где mask_u8 > 0.
    min_area : int
        Минимальная площадь компоненты (в пикселях), чтобы сохранить её.

    Returns
    -------
    np.ndarray
        Очищенная маска uint8 (H, W), значения 0/255.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1

    return (out * 255).astype(np.uint8)
