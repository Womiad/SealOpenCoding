"""Windows taskbar icon helpers shared by both Seal STT windows.

The main window runs inside the managed venv's ``pythonw.exe``, launched as a
separate process by the installer. Tk's ``iconbitmap()`` reaches the title bar,
but the taskbar button is grouped under whatever application identity Windows
infers for the process -- with none set that is the Python launcher, so the
button keeps Python's icon no matter what the window itself carries.

Two things are therefore needed, and both are easy to get subtly wrong:

* an explicit AppUserModelID, set *before* the first window exists, so Explorer
  stops folding this process into Python's taskbar group;
* real icon handles on the *decorated frame* (not the Tk widget) via
  ``WM_SETICON``, plus the window class, which is what Explorer reads when a
  window answers ``WM_GETICON`` with nothing.

The frame does not exist on the first pass, so ``install_window_icon()``
retries a few times after the event loop has had a chance to decorate it.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_IMAGE_ICON = 1
_LR_LOADFROMFILE = 0x0010
_WM_SETICON = 0x0080
_ICON_SMALL = 0
_ICON_BIG = 1
_GCLP_HICON = -14
_GCLP_HICONSM = -34
_GA_ROOT = 2
_SM_CXICON = 11
_SM_CYICON = 12
_SM_CXSMICON = 49
_SM_CYSMICON = 50

# LoadImageW hands back handles this process owns for as long as it lives.
# Holding references keeps them out of reach of any future cleanup pass and
# costs two handles for the lifetime of the window.
_LOADED_ICONS: list[int] = []


def set_app_identity(app_id: str) -> bool:
    """Give the process its own taskbar group instead of sharing Python's.

    Must run before the first window is created; Explorer reads the identity
    when the taskbar button appears and does not re-read it afterwards.
    """
    if os.name != "nt":
        return False
    try:
        shell32 = ctypes.windll.shell32
        shell32.SetCurrentProcessExplicitAppUserModelID.restype = ctypes.c_long
        shell32.SetCurrentProcessExplicitAppUserModelID.argtypes = [ctypes.c_wchar_p]
        return shell32.SetCurrentProcessExplicitAppUserModelID(app_id) == 0
    except Exception:
        return False


def _prepare(user32) -> None:
    """Declare signatures: a HANDLE truncated to 32 bits silently fails."""
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.LoadImageW.argtypes = [
        ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint,
    ]
    user32.SendMessageW.restype = ctypes.c_void_p
    user32.SendMessageW.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
    ]
    user32.GetAncestor.restype = ctypes.c_void_p
    user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]


def _load_icon(user32, path: Path, cx_metric: int, cy_metric: int) -> int:
    """Load the .ico at the size Windows actually asks for on this display."""
    cx = int(user32.GetSystemMetrics(cx_metric) or 0)
    cy = int(user32.GetSystemMetrics(cy_metric) or 0)
    handle = user32.LoadImageW(
        None, str(path), _IMAGE_ICON, cx, cy, _LR_LOADFROMFILE,
    )
    if not handle:
        return 0
    _LOADED_ICONS.append(int(handle))
    return int(handle)


def _set_class_icon(user32, hwnd: int, which: int, handle: int) -> None:
    """Explorer falls back to the window class when WM_GETICON returns NULL."""
    setter = getattr(user32, "SetClassLongPtrW", None)
    if setter is None:  # 32-bit Windows exports the plain form only.
        setter = getattr(user32, "SetClassLongW", None)
    if setter is None:
        return
    setter.restype = ctypes.c_void_p
    setter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    try:
        setter(ctypes.c_void_p(hwnd), which, ctypes.c_void_p(handle))
    except Exception:
        pass


def _target_windows(window, user32) -> list[int]:
    """Every HWND worth stamping: the Tk widget, its frame, and their roots."""
    try:
        window.update_idletasks()
    except Exception:
        pass

    raw: list[int] = []
    try:
        # The taskbar button belongs to the decorated frame, not the Tk widget.
        frame = window.wm_frame()
        if frame:
            raw.append(int(frame, 16))
    except Exception:
        pass
    try:
        raw.append(int(window.winfo_id()))
    except Exception:
        pass

    handles: list[int] = []
    for hwnd in raw:
        root = 0
        try:
            root = int(user32.GetAncestor(ctypes.c_void_p(hwnd), _GA_ROOT) or 0)
        except Exception:
            root = 0
        for candidate in (hwnd, root):
            if candidate and candidate not in handles:
                handles.append(candidate)
    return handles


def apply_window_icon(window, ico_path: Path | str) -> bool:
    """Attach real icon handles to a Tk window so the taskbar picks them up."""
    if os.name != "nt":
        return False
    path = Path(ico_path)
    if not path.is_file():
        return False
    try:
        user32 = ctypes.windll.user32
        _prepare(user32)

        small = _load_icon(user32, path, _SM_CXSMICON, _SM_CYSMICON)
        big = _load_icon(user32, path, _SM_CXICON, _SM_CYICON)
        if not small and not big:
            return False

        applied = False
        for hwnd in _target_windows(window, user32):
            for which, class_which, handle in (
                (_ICON_SMALL, _GCLP_HICONSM, small or big),
                (_ICON_BIG, _GCLP_HICON, big or small),
            ):
                if not handle:
                    continue
                user32.SendMessageW(
                    ctypes.c_void_p(hwnd), _WM_SETICON,
                    ctypes.c_void_p(which), ctypes.c_void_p(handle),
                )
                _set_class_icon(user32, hwnd, class_which, handle)
                applied = True
        return applied
    except Exception:
        return False


def install_window_icon(window, ico_path: Path | str,
                        retries: tuple[int, ...] = (250, 900, 2500)) -> bool:
    """Apply the icon now, then again once Tk has decorated the window."""
    applied = apply_window_icon(window, ico_path)
    for delay in retries:
        try:
            window.after(delay, lambda: apply_window_icon(window, ico_path))
        except Exception:
            pass
    return applied
