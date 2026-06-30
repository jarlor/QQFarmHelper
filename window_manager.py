# -*- coding: utf-8 -*-
"""
window_manager.py
自动寻找 QQ经典农场窗口，并返回截图和点击使用的屏幕坐标。

Windows 使用 pywin32；macOS 使用 PyObjC 的 Quartz/AppKit。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple
import ctypes
import platform
import subprocess
import time


SYSTEM = platform.system()

DEFAULT_WINDOW_KEYWORDS = [
    "QQ经典农场",
    "QQ农场",
    "QQ空间",
    "MuMu",
    "雷电模拟器",
    "腾讯手游助手",
]

DEFAULT_MIN_WIDTH = 120
DEFAULT_MIN_HEIGHT = 120
MACOS_FALLBACK_PROCESS_NAMES = [
    "QQEXMiniProgram",
    "QQMini",
    "MuMu",
]
MACOS_FALLBACK_BUNDLE_KEYWORDS = [
    "com.tencent.qqexminiprogram",
    "mumu",
    "leidian",
]


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def tuple(self) -> Tuple[int, int, int, int]:
        """返回 left, top, right, bottom。"""
        return (self.left, self.top, self.right, self.bottom)

    @property
    def mss_dict(self) -> dict:
        """返回 mss 截图可用的 region dict。"""
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    window_rect: Rect
    client_rect: Rect
    platform: str = SYSTEM
    metadata: dict[str, Any] | None = None

    @property
    def is_valid(self) -> bool:
        return refresh_window(self.hwnd) is not None


def _require_windows_modules():
    try:
        import win32con
        import win32gui
    except ImportError as exc:
        raise RuntimeError("Windows 窗口控制需要安装 pywin32：pip install pywin32") from exc

    return win32con, win32gui


def _require_quartz_modules():
    try:
        import AppKit
        import Quartz
    except ImportError as exc:
        raise RuntimeError(
            "macOS 窗口控制需要安装 PyObjC：pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa"
        ) from exc

    return AppKit, Quartz


def set_dpi_awareness() -> None:
    """
    Windows 设置进程 DPI 感知，避免缩放导致截图坐标和点击坐标错位。
    macOS 下不需要处理。
    """
    if SYSTEM != "Windows":
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _get_window_rect_windows(hwnd: int) -> Rect:
    _, win32gui = _require_windows_modules()
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return Rect(left, top, max(0, right - left), max(0, bottom - top))


def _get_client_rect_on_screen_windows(hwnd: int) -> Rect:
    """
    获取 Windows 窗口客户区在屏幕上的位置。
    客户区不包含标题栏、边框。推荐截图和相对坐标都基于客户区。
    """
    _, win32gui = _require_windows_modules()
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
    screen_right, screen_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    return Rect(
        screen_left,
        screen_top,
        max(0, screen_right - screen_left),
        max(0, screen_bottom - screen_top),
    )


def _mac_bounds_to_rect(bounds: dict) -> Rect:
    return Rect(
        int(round(bounds.get("X", 0))),
        int(round(bounds.get("Y", 0))),
        max(0, int(round(bounds.get("Width", 0)))),
        max(0, int(round(bounds.get("Height", 0)))),
    )


def _mac_title(info: dict) -> str:
    owner = str(info.get("kCGWindowOwnerName") or "").strip()
    name = str(info.get("kCGWindowName") or "").strip()

    if owner and name:
        return f"{owner} - {name}"
    return name or owner


def _mac_window_id(info: dict) -> int:
    return int(info.get("kCGWindowNumber", 0) or 0)


def _mac_owner_pid(info: dict) -> int:
    return int(info.get("kCGWindowOwnerPID", 0) or 0)


def _get_mac_window_infos() -> list[dict]:
    _, Quartz = _require_quartz_modules()
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    return list(Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or [])


def _game_window_from_mac_info(info: dict) -> Optional[GameWindow]:
    title = _mac_title(info)
    if not title:
        return None

    bounds = info.get("kCGWindowBounds") or {}
    rect = _mac_bounds_to_rect(bounds)
    if rect.width <= 0 or rect.height <= 0:
        return None

    return GameWindow(
        hwnd=_mac_window_id(info),
        title=title,
        window_rect=rect,
        client_rect=rect,
        platform="Darwin",
        metadata={"owner_pid": _mac_owner_pid(info), "owner_name": info.get("kCGWindowOwnerName")},
    )


def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _macos_fallback_process_names(keywords: Iterable[str]) -> list[str]:
    names = list(MACOS_FALLBACK_PROCESS_NAMES)
    try:
        AppKit, _ = _require_quartz_modules()
        for app in AppKit.NSWorkspace.sharedWorkspace().runningApplications():
            name = str(app.localizedName() or "").strip()
            bundle_id = str(app.bundleIdentifier() or "").strip().lower()
            if not name:
                continue
            if any(marker in bundle_id for marker in MACOS_FALLBACK_BUNDLE_KEYWORDS):
                if name not in names:
                    names.append(name)
    except Exception:
        pass

    return names


def _run_macos_window_applescript(process_names: Iterable[str]) -> str:
    names_literal = ", ".join(f'"{_escape_applescript_string(name)}"' for name in process_names if name)
    if not names_literal:
        return ""

    script = f"""
set processNames to {{{names_literal}}}
set outputLines to {{}}
tell application "System Events"
  repeat with procName in processNames
    try
      if exists process (procName as text) then
        tell process (procName as text)
          set procPid to unix id
          set procDisplayName to name
          repeat with win in windows
            try
              set winTitle to ""
              try
                set winTitle to name of win as text
              end try
              if winTitle is "" or winTitle is "missing value" then
                try
                  set winTitle to value of attribute "AXTitle" of win as text
                end try
              end if
              set winPos to position of win
              set winSize to size of win
              set lineText to (procPid as text) & tab & procDisplayName & tab & winTitle & tab & ((item 1 of winPos) as text) & tab & ((item 2 of winPos) as text) & tab & ((item 1 of winSize) as text) & tab & ((item 2 of winSize) as text)
              set end of outputLines to lineText
            end try
          end repeat
        end tell
      end if
    end try
  end repeat
end tell
set AppleScript's text item delimiters to linefeed
return outputLines as text
"""

    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except subprocess.TimeoutExpired:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _parse_macos_fallback_windows(output: str) -> list[GameWindow]:
    windows: list[GameWindow] = []
    for line in output.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 7:
            continue

        pid_raw, proc_name, title, left, top, width, height = parts
        try:
            pid = int(pid_raw)
            rect = Rect(int(float(left)), int(float(top)), int(float(width)), int(float(height)))
        except ValueError:
            continue

        if rect.width <= 0 or rect.height <= 0:
            continue

        title = title.strip() or proc_name.strip()
        hwnd = -pid if title == proc_name else -abs(hash((pid, title, rect.tuple)) % 2_000_000_000)
        windows.append(
            GameWindow(
                hwnd=hwnd,
                title=title,
                window_rect=rect,
                client_rect=rect,
                platform="Darwin",
                metadata={"owner_pid": pid, "owner_name": proc_name, "source": "system_events"},
            )
        )

    return windows


def _get_macos_fallback_windows(keywords: Iterable[str]) -> list[GameWindow]:
    process_names = _macos_fallback_process_names(keywords)
    return _parse_macos_fallback_windows(_run_macos_window_applescript(process_names))


def is_normal_window(
    hwnd: int,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> bool:
    if SYSTEM == "Windows":
        _, win32gui = _require_windows_modules()
        if not win32gui.IsWindow(hwnd):
            return False
        if not win32gui.IsWindowVisible(hwnd):
            return False
        if win32gui.IsIconic(hwnd):
            return False

        title = win32gui.GetWindowText(hwnd).strip()
        if not title:
            return False

        rect = _get_window_rect_windows(hwnd)
        return rect.width >= min_width and rect.height >= min_height

    if SYSTEM == "Darwin":
        return refresh_window(hwnd) is not None

    raise RuntimeError(f"暂不支持当前系统：{SYSTEM}")


def enum_candidate_windows(
    keywords: Optional[Iterable[str]] = None,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> List[GameWindow]:
    """
    枚举符合关键词的窗口。
    keywords 为空时，返回所有可见普通窗口。
    """
    if SYSTEM == "Windows":
        return _enum_candidate_windows_windows(keywords, min_width, min_height)

    if SYSTEM == "Darwin":
        return _enum_candidate_windows_macos(keywords, min_width, min_height)

    raise RuntimeError(f"暂不支持当前系统：{SYSTEM}")


def _matches_keywords(title: str, keywords: Iterable[str]) -> bool:
    lowered_title = title.lower()
    lowered_keywords = [k.lower() for k in keywords if k]
    return not lowered_keywords or any(k in lowered_title for k in lowered_keywords)


def _enum_candidate_windows_windows(
    keywords: Optional[Iterable[str]],
    min_width: int,
    min_height: int,
) -> List[GameWindow]:
    set_dpi_awareness()
    _, win32gui = _require_windows_modules()
    keyword_list = list(keywords or [])
    results: List[GameWindow] = []

    def callback(hwnd: int, _extra) -> None:
        try:
            if not is_normal_window(hwnd, min_width=min_width, min_height=min_height):
                return

            title = win32gui.GetWindowText(hwnd).strip()
            if not _matches_keywords(title, keyword_list):
                return

            window_rect = _get_window_rect_windows(hwnd)
            client_rect = _get_client_rect_on_screen_windows(hwnd)

            if client_rect.width < min_width or client_rect.height < min_height:
                return

            results.append(
                GameWindow(
                    hwnd=hwnd,
                    title=title,
                    window_rect=window_rect,
                    client_rect=client_rect,
                    platform="Windows",
                )
            )
        except Exception:
            return

    win32gui.EnumWindows(callback, None)
    results.sort(key=lambda w: w.client_rect.width * w.client_rect.height, reverse=True)
    return results


def _enum_candidate_windows_macos(
    keywords: Optional[Iterable[str]],
    min_width: int,
    min_height: int,
) -> List[GameWindow]:
    keyword_list = list(keywords or [])
    results: List[GameWindow] = []

    for info in _get_mac_window_infos():
        try:
            window = _game_window_from_mac_info(info)
            if window is None:
                continue
            if window.client_rect.width < min_width or window.client_rect.height < min_height:
                continue
            if not _matches_keywords(window.title, keyword_list):
                continue
            results.append(window)
        except Exception:
            continue

    seen_keys = {(w.title, w.client_rect.tuple) for w in results}
    for window in _get_macos_fallback_windows(keyword_list):
        if window.client_rect.width < min_width or window.client_rect.height < min_height:
            continue
        if not _matches_keywords(window.title, keyword_list):
            continue
        key = (window.title, window.client_rect.tuple)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        results.append(window)

    results.sort(key=lambda w: w.client_rect.width * w.client_rect.height, reverse=True)
    return results


def find_game_window(
    keywords: Optional[Iterable[str]] = None,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> Optional[GameWindow]:
    """
    返回最可能的 QQ经典农场窗口。
    """
    candidates = enum_candidate_windows(
        keywords=keywords or DEFAULT_WINDOW_KEYWORDS,
        min_width=min_width,
        min_height=min_height,
    )
    return candidates[0] if candidates else None


def refresh_window(hwnd: int) -> Optional[GameWindow]:
    """
    已知窗口 ID 时，重新读取窗口位置。
    窗口被拖动、缩放后需要调用这个。
    """
    try:
        if SYSTEM == "Windows":
            _, win32gui = _require_windows_modules()
            if not is_normal_window(hwnd):
                return None
            return GameWindow(
                hwnd=hwnd,
                title=win32gui.GetWindowText(hwnd).strip(),
                window_rect=_get_window_rect_windows(hwnd),
                client_rect=_get_client_rect_on_screen_windows(hwnd),
                platform="Windows",
            )

        if SYSTEM == "Darwin":
            for info in _get_mac_window_infos():
                if _mac_window_id(info) == int(hwnd):
                    return _game_window_from_mac_info(info)
            if hwnd < 0:
                for window in _get_macos_fallback_windows(DEFAULT_WINDOW_KEYWORDS):
                    if window.hwnd == hwnd:
                        return window
            return None

        raise RuntimeError(f"暂不支持当前系统：{SYSTEM}")
    except Exception:
        return None


def bring_to_front(hwnd: int, restore: bool = True) -> bool:
    """
    尝试把窗口置前。
    macOS 只能激活窗口所属 App，具体窗口置前由系统决定。
    """
    try:
        if SYSTEM == "Windows":
            win32con, win32gui = _require_windows_modules()
            if restore and win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.2)

            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(hwnd)
            return True

        if SYSTEM == "Darwin":
            AppKit, _ = _require_quartz_modules()
            window = refresh_window(hwnd)
            if window is None:
                return False

            pid = int((window.metadata or {}).get("owner_pid") or 0)
            if pid <= 0:
                return False

            app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            if app is None:
                return False

            opts = AppKit.NSApplicationActivateIgnoringOtherApps
            return bool(app.activateWithOptions_(opts))

        raise RuntimeError(f"暂不支持当前系统：{SYSTEM}")
    except Exception:
        return False


def wait_for_game_window(
    keywords: Optional[Iterable[str]] = None,
    timeout: float = 10.0,
    interval: float = 0.5,
    min_width: int = DEFAULT_MIN_WIDTH,
    min_height: int = DEFAULT_MIN_HEIGHT,
) -> Optional[GameWindow]:
    """
    在 timeout 秒内等待游戏窗口出现。
    """
    start = time.time()
    while time.time() - start <= timeout:
        win = find_game_window(
            keywords=keywords or DEFAULT_WINDOW_KEYWORDS,
            min_width=min_width,
            min_height=min_height,
        )
        if win:
            return win
        time.sleep(interval)
    return None


def print_candidate_windows(keywords: Optional[Iterable[str]] = None) -> None:
    """
    调试用：打印候选窗口，方便确认关键词是否正确。
    """
    candidates = enum_candidate_windows(keywords=keywords)
    for i, w in enumerate(candidates, 1):
        print(
            f"[{i}] id={w.hwnd} platform={w.platform} title={w.title!r} "
            f"client={w.client_rect.left},{w.client_rect.top},"
            f"{w.client_rect.width}x{w.client_rect.height}"
        )


if __name__ == "__main__":
    print_candidate_windows(DEFAULT_WINDOW_KEYWORDS)
