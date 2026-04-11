import logging
import sys
import config

# ANSI color codes
_RESET = "\x1b[0m"
_GRAY = "\x1b[90m"
_CYAN = "\x1b[36m"
_BLUE = "\x1b[94m"
_YELLOW = "\x1b[33m"
_RED = "\x1b[31m"
_BRIGHT_RED = "\x1b[91m"

_LEVEL_COLORS = {
    logging.DEBUG:    _CYAN,
    logging.INFO:     _BLUE,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BRIGHT_RED,
}

_LEVEL_NAMES = {
    logging.DEBUG:    "Debug   ",
    logging.INFO:     "Info    ",
    logging.WARNING:  "Warning ",
    logging.ERROR:    "Error   ",
    logging.CRITICAL: "Critical",
}


def _try_enable_ansi_windows() -> None:
    """Enable ANSI escape codes in the Windows console if needed."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


class _WebiGeoFormatter(logging.Formatter):
    """
    Colored log formatter styled after the weBIGeo C++ logging:
      HH:MM:SS | Level    | file.py:line              | message
    The left section (time, level, file) is colored per level.
    Debug messages additionally render the message text in gray.
    """

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_str = _LEVEL_NAMES.get(record.levelno, f"{record.levelname:<8}")
        file_part = f"{record.filename}:{record.lineno}"
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            left = f"{color}{time_str} | {level_str} | {file_part:<25} |{_RESET}"
            if record.levelno == logging.DEBUG:
                return f"{left} {_GRAY}{msg}{_RESET}"
            return f"{left} {msg}"

        return f"{time_str} | {level_str} | {file_part:<25} | {msg}"


def setup_logging() -> None:
    """Configure the root logger with weBIGeo-style colored formatting."""
    _try_enable_ansi_windows()

    level = getattr(logging, config.log_level.upper(), logging.DEBUG)

    use_color = sys.stdout.isatty()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_WebiGeoFormatter(use_color=use_color))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    for name, override in config.log_level_overrides.items():
        logging.getLogger(name).setLevel(getattr(logging, override.upper(), logging.WARNING))
