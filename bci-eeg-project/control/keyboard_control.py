"""
control/keyboard_control.py
----------------------------
Translate motor imagery predictions into keyboard/mouse actions using pyautogui.

Motor imagery → keyboard mapping:
  LEFT HAND  → left arrow key
  RIGHT HAND → right arrow key
  FEET       → down arrow key
  REST       → no action

Safety:
  Press ESC to disable control mode at any time.

Usage
-----
    from control.keyboard_control import execute_motor_command
    execute_motor_command("LEFT HAND")
"""

import time
from typing import Optional

# Debounce interval in seconds
_DEBOUNCE_S = 0.5

_last_command_time: float = 0.0
_control_enabled: bool    = True

# Mapping of intent → pyautogui key name
_KEYMAP = {
    "LEFT HAND":  "left",
    "RIGHT HAND": "right",
    "FEET":       "down",
    "REST":       None,
}


def _get_pyautogui():
    """Lazy-import pyautogui to allow headless import of the module."""
    try:
        import pyautogui  # type: ignore
        return pyautogui
    except ImportError as exc:
        raise ImportError(
            "pyautogui is required for keyboard control. "
            "Install with: pip install pyautogui"
        ) from exc


def enable_control() -> None:
    """Re-enable keyboard control mode."""
    global _control_enabled
    _control_enabled = True
    print("[control] Control mode ENABLED.")


def disable_control() -> None:
    """Disable keyboard control mode (no keypresses will be sent)."""
    global _control_enabled
    _control_enabled = False
    print("[control] Control mode DISABLED.")


def execute_motor_command(intent: str) -> Optional[str]:
    """Translate a motor imagery intent into a keyboard action.

    Applies a 500 ms debounce to prevent repeated keypresses.

    Args:
        intent: One of "LEFT HAND", "RIGHT HAND", "FEET", "REST".

    Returns:
        The key pressed (e.g. "left"), or None if no action was taken.
    """
    global _last_command_time, _control_enabled

    if not _control_enabled:
        return None

    now = time.monotonic()
    if now - _last_command_time < _DEBOUNCE_S:
        return None   # Still within debounce window

    key = _KEYMAP.get(intent.upper())
    if key is None:
        return None   # REST or unknown intent → no action

    pyautogui = _get_pyautogui()

    # Safety: check if ESC is currently held down → disable control
    try:
        import keyboard  # type: ignore
        if keyboard.is_pressed("esc"):
            disable_control()
            return None
    except ImportError:
        pass  # keyboard package not installed — skip ESC check

    pyautogui.press(key)
    _last_command_time = now
    print(f"[control] Pressed: {key}  ← {intent}")
    return key


if __name__ == "__main__":
    print("BCI Keyboard Control — demo mode")
    print("Press Ctrl+C to quit.\n")
    intents = ["LEFT HAND", "REST", "RIGHT HAND", "FEET", "REST", "LEFT HAND"]
    for intent in intents:
        result = execute_motor_command(intent)
        print(f"  intent={intent}  →  key={result}")
        time.sleep(0.6)
