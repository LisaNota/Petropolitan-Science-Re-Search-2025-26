
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def bgr2rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Конвертирует изображение из формата BGR (OpenCV) в RGB (Matplotlib).

    Parameters
    ----------
    img_bgr : np.ndarray
        Входное изображение в цветовом пространстве BGR.
        Ожидается массив формы (H, W, 3).

    Returns
    -------
    np.ndarray
        Изображение в RGB, форма (H, W, 3).

    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show(img_bgr: np.ndarray, title: str = "", figsize: tuple[int, int] = (7, 5)) -> None:
    """
    Отображает BGR-изображение с помощью Matplotlib.

    Parameters
    ----------
    img_bgr : np.ndarray
        Изображение в BGR, форма (H, W, 3).
    title : str, optional
        Заголовок для изображения. По умолчанию пустая строка.
    figsize : tuple[int, int], optional
        Размер фигуры Matplotlib (ширина, высота). По умолчанию (7, 5).
    """
    plt.figure(figsize=figsize)
    plt.imshow(bgr2rgb(img_bgr))
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def show_mask(mask: np.ndarray, title: str = "", figsize: tuple[int, int] = (7, 5)) -> None:
    """
    Отображает маску как чёрно-белое изображение.

    Parameters
    ----------
    mask : np.ndarray
        Маска:
        - bool массив формы (H, W) или
        - uint8 массив формы (H, W).
    title : str, optional
        Заголовок. По умолчанию пустая строка.
    figsize : tuple[int, int], optional
        Размер фигуры (ширина, высота). По умолчанию (7, 5).
    """
    m = mask
    if m.dtype != np.uint8:
        m = (m.astype(np.uint8) * 255)

    plt.figure(figsize=figsize)
    plt.imshow(m, cmap="gray")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def overlay_mask(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Накладывает маску на изображение (overlay) красным цветом для визуализации.

    Parameters
    ----------
    img_bgr : np.ndarray
        Исходное изображение в BGR, форма (H, W, 3).
    mask_u8 : np.ndarray
        Маска формы (H, W), dtype uint8.
        Считается, что пиксели маски активны там, где mask_u8 > 0.
    alpha : float, optional
        Прозрачность наложения в диапазоне [0..1]:
        - 0.0: маска не видна
        - 1.0: полностью красная заливка в области маски
        По умолчанию 0.55.

    Returns
    -------
    np.ndarray
        Изображение BGR с наложенной красной подсветкой маски, dtype uint8.
    """
    out = img_bgr.copy().astype(np.float32)
    m = (mask_u8 > 0)

    red = np.zeros_like(out)
    red[..., 2] = 255.0  # в BGR это красный канал (R)

    out[m] = out[m] * (1.0 - alpha) + red[m] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def show_overlay(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    title: str = "",
    figsize: tuple[int, int] = (7, 5),
) -> None:
    """
    Отображает изображение с наложенной маской (overlay) красным цветом.

    Parameters
    ----------
    img_bgr : np.ndarray
        Исходное изображение BGR, форма (H, W, 3).
    mask_u8 : np.ndarray
        Маска uint8 формы (H, W), активна там, где mask_u8 > 0.
    title : str, optional
        Заголовок. По умолчанию пустая строка.
    figsize : tuple[int, int], optional
        Размер фигуры Matplotlib. По умолчанию (7, 5).
    """
    show(overlay_mask(img_bgr, mask_u8), title=title, figsize=figsize)


def show_alpha(alpha_map: np.ndarray, title: str = "", figsize: tuple[int, int] = (7, 5)) -> None:
    """
    Отображает альфа-карту (float-маску смешивания) в диапазоне [0..1].

    Parameters
    ----------
    alpha_map : np.ndarray
        Альфа-карта формы (H, W), dtype float (обычно float32),
        значения предполагаются в диапазоне [0..1].
        0 = не применять изменения, 1 = применять полностью.
    title : str, optional
        Заголовок. По умолчанию пустая строка.
    figsize : tuple[int, int], optional
        Размер фигуры Matplotlib. По умолчанию (7, 5).
    """
    plt.figure(figsize=figsize)
    plt.imshow(alpha_map, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
