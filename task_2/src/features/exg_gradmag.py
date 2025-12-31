import cv2
import numpy as np


def exg(img_bgr: np.ndarray) -> np.ndarray:
    """
    Вычисляет показатель Excess Green (ExG) = 2*G - R - B.
    ExG хорошо поднимает "зелёные" области (трава/летняя листва),
      поэтому полезен для построения масок растительности.

    Parameters
    ----------
    img_bgr : np.ndarray
        Изображение в BGR, форма (H, W, 3)

    Returns
    -------
    np.ndarray
        ExG-карта формы (H, W), dtype float32.
        Значения не нормированы и зависят от яркости/контраста.
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    return 2.0 * g - r - b


def gradmag(gray_f32: np.ndarray) -> np.ndarray:
    """
    Вычисляет модуль градиента изображения (Sobel) как меру текстурности/границ.
    Помогает отличать кроны/ветви (много мелких границ) от ровных областей (небо, гладкие участки)
    Используется как дополнительный фильтр при построении масок

    Parameters
    ----------
    gray_f32 : np.ndarray
        Одноканальное изображение (H, W)

    Returns
    -------
    np.ndarray
        Модуль градиента формы (H, W)
    """
    gx = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)
