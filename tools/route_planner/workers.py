"""Runs one function off the UI thread and reports progress, the result or the error back by signal."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal

TaskFunction = Callable[[Callable[[str, int, int], None]], Any]   # takes the progress hook, returns the result


class TaskThread(QThread):
    progress = Signal(str, int, int)         # stage, done, total
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, function: TaskFunction) -> None:
        super().__init__()
        self._function = function

    def run(self) -> None:
        try:
            result = self._function(self.progress.emit)
        except Exception as error:   # noqa: BLE001 - whatever went wrong goes to the user's message box
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.succeeded.emit(result)
