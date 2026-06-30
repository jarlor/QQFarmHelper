# -*- coding: utf-8 -*-
"""
screen_capture.py
截取 QQ经典农场窗口客户区画面。

依赖：
    pip install mss opencv-python numpy

返回图像格式：
    - capture_window() 返回 OpenCV BGR numpy.ndarray。
    - 坐标默认基于 window.client_rect。
    - Windows 的 client_rect 不含标题栏；macOS 的窗口列表拿不到客户区，
      client_rect 与 window_rect 相同，因此建议把游戏窗口内容区域尽量铺满窗口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import time

import cv2
import mss
import numpy as np

from window_manager import GameWindow, Rect, refresh_window


RatioROI = Sequence[float]  # [x1, y1, x2, y2], 0~1 相对坐标


def ratio_roi_to_rect(base: Rect, roi: RatioROI) -> Rect:
    """
    把相对 ROI 转成屏幕绝对 Rect。
    roi = [x1, y1, x2, y2]，基于 base 的宽高。
    """
    if len(roi) != 4:
        raise ValueError("roi 必须是 [x1, y1, x2, y2]")

    x1, y1, x2, y2 = map(float, roi)
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"非法 roi: {roi}")

    left = base.left + int(round(base.width * x1))
    top = base.top + int(round(base.height * y1))
    right = base.left + int(round(base.width * x2))
    bottom = base.top + int(round(base.height * y2))

    return Rect(left, top, max(1, right - left), max(1, bottom - top))


class ScreenCapture:
    """
    复用 mss 实例，循环截图时更快。
    """

    def __init__(self) -> None:
        self._sct = mss.MSS()

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass

    def capture_rect(self, rect: Rect) -> np.ndarray:
        """
        截取屏幕绝对 rect。
        返回 BGR 图像。
        """
        raw = np.array(self._sct.grab(rect.mss_dict))
        # mss 输出 BGRA，OpenCV 常用 BGR
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def capture_window(
        self,
        window: GameWindow,
        client_only: bool = True,
        refresh: bool = True,
    ) -> np.ndarray:
        """
        截取窗口。
        client_only=True：只截客户区，推荐。
        refresh=True：截图前重新读取窗口位置，窗口移动后坐标不会旧。
        """
        if refresh:
            latest = refresh_window(window.hwnd)
            if latest is not None:
                window = latest

        rect = window.client_rect if client_only else window.window_rect
        return self.capture_rect(rect)

    def capture_window_roi(
        self,
        window: GameWindow,
        roi: RatioROI,
        client_only: bool = True,
        refresh: bool = True,
    ) -> np.ndarray:
        """
        截取窗口中的相对 ROI。
        """
        if refresh:
            latest = refresh_window(window.hwnd)
            if latest is not None:
                window = latest

        base = window.client_rect if client_only else window.window_rect
        rect = ratio_roi_to_rect(base, roi)
        return self.capture_rect(rect)

    def save_debug(
        self,
        image_bgr: np.ndarray,
        path: str | Path,
    ) -> Path:
        """
        保存调试截图。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image_bgr)
        return path


def capture_window_once(
    window: GameWindow,
    client_only: bool = True,
    refresh: bool = True,
) -> np.ndarray:
    """
    简单一次性截图。
    """
    cap = ScreenCapture()
    try:
        return cap.capture_window(window, client_only=client_only, refresh=refresh)
    finally:
        cap.close()


def crop_by_ratio(image_bgr: np.ndarray, roi: RatioROI) -> np.ndarray:
    """
    从已有图像中按相对坐标裁剪。
    注意：这个 roi 基于图像本身，不是屏幕。
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, roi)

    px1 = max(0, min(w - 1, int(round(w * x1))))
    py1 = max(0, min(h - 1, int(round(h * y1))))
    px2 = max(px1 + 1, min(w, int(round(w * x2))))
    py2 = max(py1 + 1, min(h, int(round(h * y2))))

    return image_bgr[py1:py2, px1:px2].copy()


if __name__ == "__main__":
    from window_manager import find_game_window, DEFAULT_WINDOW_KEYWORDS

    win = find_game_window(DEFAULT_WINDOW_KEYWORDS)
    if not win:
        print("没有找到游戏窗口")
    else:
        cap = ScreenCapture()
        img = cap.capture_window(win)
        out = Path("debug_window.png")
        cap.save_debug(img, out)
        print(f"已保存: {out.resolve()}")
