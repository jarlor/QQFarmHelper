# -*- coding: utf-8 -*-
"""
clicker.py
把 vision.py 识别到的“客户区图像坐标”转换成屏幕坐标，并执行鼠标点击。

依赖：
    pip install pywin32

坐标约定：
    - Vision 返回的 MatchResult.center 是“窗口客户区截图”里的像素坐标。
    - Clicker 会把它转换成屏幕坐标：
        screen_x = window.client_rect.left + center_x
        screen_y = window.client_rect.top  + center_y

安全机制：
    - dry_run=True 时只打印，不真正点击。
    - random_offset 默认 2 像素，让点击位置稍微自然一点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import random
import time

import win32api
import win32con

from window_manager import GameWindow, Rect, refresh_window, bring_to_front


@dataclass
class ClickResult:
    ok: bool
    screen_pos: Optional[Tuple[int, int]] = None
    reason: str = ""


class Clicker:
    def __init__(
        self,
        dry_run: bool = False,
        random_offset: int = 2,
        click_down_up_delay: float = 0.05,
        bring_front_before_click: bool = True,
    ) -> None:
        self.dry_run = dry_run
        self.random_offset = int(random_offset)
        self.click_down_up_delay = float(click_down_up_delay)
        self.bring_front_before_click = bool(bring_front_before_click)

    def _latest_client_rect(self, window: GameWindow) -> Optional[Rect]:
        latest = refresh_window(window.hwnd)
        if latest is None:
            return None
        return latest.client_rect

    def client_to_screen(self, window: GameWindow, x: int, y: int) -> Optional[Tuple[int, int]]:
        rect = self._latest_client_rect(window)
        if rect is None:
            return None
        return rect.left + int(x), rect.top + int(y)

    def relative_to_screen(self, window: GameWindow, rx: float, ry: float) -> Optional[Tuple[int, int]]:
        rect = self._latest_client_rect(window)
        if rect is None:
            return None

        x = rect.left + int(round(rect.width * float(rx)))
        y = rect.top + int(round(rect.height * float(ry)))
        return x, y

    def _apply_random_offset(self, x: int, y: int) -> Tuple[int, int]:
        if self.random_offset <= 0:
            return x, y

        return (
            x + random.randint(-self.random_offset, self.random_offset),
            y + random.randint(-self.random_offset, self.random_offset),
        )

    def click_screen(self, x: int, y: int, button: str = "left") -> ClickResult:
        x, y = self._apply_random_offset(int(x), int(y))

        if self.dry_run:
            print(f"[DRY-RUN] click_screen ({x}, {y}) button={button}")
            return ClickResult(True, (x, y), "dry_run")

        try:
            win32api.SetCursorPos((x, y))
            time.sleep(0.02)

            if button == "left":
                down = win32con.MOUSEEVENTF_LEFTDOWN
                up = win32con.MOUSEEVENTF_LEFTUP
            elif button == "right":
                down = win32con.MOUSEEVENTF_RIGHTDOWN
                up = win32con.MOUSEEVENTF_RIGHTUP
            else:
                return ClickResult(False, (x, y), f"不支持的鼠标按钮: {button}")

            win32api.mouse_event(down, 0, 0, 0, 0)
            time.sleep(self.click_down_up_delay)
            win32api.mouse_event(up, 0, 0, 0, 0)
            return ClickResult(True, (x, y), "clicked")
        except Exception as e:
            return ClickResult(False, (x, y), repr(e))

    def click_client(self, window: GameWindow, x: int, y: int, button: str = "left") -> ClickResult:
        if self.bring_front_before_click:
            bring_to_front(window.hwnd)

        pos = self.client_to_screen(window, int(x), int(y))
        if pos is None:
            return ClickResult(False, None, "窗口无效或已关闭")

        return self.click_screen(pos[0], pos[1], button=button)

    def click_relative(self, window: GameWindow, rx: float, ry: float, button: str = "left") -> ClickResult:
        if self.bring_front_before_click:
            bring_to_front(window.hwnd)

        pos = self.relative_to_screen(window, float(rx), float(ry))
        if pos is None:
            return ClickResult(False, None, "窗口无效或已关闭")

        return self.click_screen(pos[0], pos[1], button=button)

    def click_match(self, window: GameWindow, match, button: str = "left") -> ClickResult:
        """
        点击 Vision.MatchResult 的中心。
        match.center 必须是客户区截图坐标。
        """
        if not getattr(match, "found", False):
            return ClickResult(False, None, "match.found=False")

        center = getattr(match, "center", None)
        if not center:
            return ClickResult(False, None, "match.center=None")

        return self.click_client(window, int(center[0]), int(center[1]), button=button)
