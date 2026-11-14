"""Continuous terminal logger for real-time, readable agent execution logs with explicit placeholder notices and CLI animations."""

import json
import sys
import threading
import time
from typing import Any, Dict, Optional


class ConsoleSpinner:
    """Threaded CLI spinner with real-time animated frames, elapsed time counter, and dynamic message updates."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Processing..."):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def _spin(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            elapsed = int(time.time() - self._start_time)
            frame = self.FRAMES[idx % len(self.FRAMES)]
            line = f"\r\033[36m{frame}\033[0m \033[1m{self.message}\033[0m \033[2m[{elapsed}s]\033[0m "
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            idx += 1
            self._stop_event.wait(0.08)

        # Cleanly erase the spinner line when finished
        try:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        except Exception:
            pass

    def start(self) -> "ConsoleSpinner":
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def update(self, new_message: str) -> None:
        self.message = new_message

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "ConsoleSpinner":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    async def __aenter__(self) -> "ConsoleSpinner":
        return self.start()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class TerminalLogger:
    """Provides formatted, colored, real-time logging for the engineering agent pipeline."""

    # ANSI Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BG_YELLOW = "\033[43m"
    BLACK = "\033[30m"

    @classmethod
    def header(cls, title: str) -> None:
        line = "=" * 70
        print(f"\n{cls.BOLD}{cls.CYAN}{line}{cls.RESET}")
        print(f"{cls.BOLD}{cls.CYAN}  {title}{cls.RESET}")
        print(f"{cls.BOLD}{cls.CYAN}{line}{cls.RESET}\n", flush=True)

    @classmethod
    def stage(cls, step: int, total: int, title: str) -> None:
        badge = f"STAGE {step}/{total}"
        print(f"\n{cls.BOLD}{cls.MAGENTA}┌─ [{badge}] {title.upper()}{cls.RESET}", flush=True)

    @classmethod
    def thought(cls, thought_text: str) -> None:
        cleaned = thought_text.strip()
        if not cleaned:
            return
        print(f"{cls.DIM}{cls.YELLOW}[Agent Thinking]{cls.RESET} {cleaned}", flush=True)

    @classmethod
    def tool_call(cls, tool_name: str, args: Dict[str, Any]) -> None:
        print(f"{cls.BOLD}{cls.BLUE}▶ [Tool Call]{cls.RESET} {cls.BOLD}{tool_name}{cls.RESET}", flush=True)
        for key, val in args.items():
            print(f"    • {cls.DIM}{key}:{cls.RESET} {val}", flush=True)

    @classmethod
    def _is_placeholder(cls, output: Any) -> bool:
        """Determines if the output originated from a placeholder stage."""
        if isinstance(output, dict):
            if output.get("is_placeholder") is True:
                return True
            if "PLACEHOLDER" in str(output.get("execution_mode", "")).upper():
                return True
            if "PLACEHOLDER" in str(output.get("stage", "")).upper():
                return True
        elif isinstance(output, str):
            if "PLACEHOLDER" in output.upper():
                return True
        return False

    @classmethod
    def tool_result(cls, tool_name: str, output: Any) -> None:
        is_ph = cls._is_placeholder(output)

        if is_ph:
            # High-visibility warning banner for placeholder outputs
            print(f"\n{cls.BOLD}{cls.YELLOW}┌{'─' * 68}┐{cls.RESET}", flush=True)
            print(f"{cls.BOLD}{cls.YELLOW}│ ⚠️  [PLACEHOLDER OUTPUT DETECTED]                                  │{cls.RESET}", flush=True)
            print(f"{cls.BOLD}{cls.YELLOW}│ Stage: {tool_name:<58} │{cls.RESET}", flush=True)
            print(f"{cls.BOLD}{cls.YELLOW}│ Note : This stage is a PLACEHOLDER. Output values are simulated.   │{cls.RESET}", flush=True)
            print(f"{cls.BOLD}{cls.YELLOW}└{'─' * 68}┘{cls.RESET}", flush=True)
            badge_color = cls.YELLOW
            tag = "[PLACEHOLDER RESULT]"
        else:
            badge_color = cls.GREEN
            tag = "[Tool Result]"

        print(f"{cls.BOLD}{badge_color}✔ {tag} - {tool_name}{cls.RESET}", flush=True)

        # Parse dict if received as JSON string
        data = output
        if isinstance(output, str) and (output.strip().startswith("{") and output.strip().endswith("}")):
            try:
                data = json.loads(output)
            except Exception:
                data = output

        if isinstance(data, dict):
            for k, v in data.items():
                prefix = f"{cls.YELLOW}[PLACEHOLDER]{cls.RESET} " if is_ph and "placeholder" in k.lower() else ""
                print(f"    ✔ {prefix}{cls.BOLD}{k}{cls.RESET}: {v}", flush=True)
        else:
            text = str(data)
            lines = text.strip().split("\n")
            for line in lines[:8]:
                print(f"    {line}", flush=True)
            if len(lines) > 8:
                print(f"    ... ({len(lines)-8} more lines)", flush=True)
        print("", flush=True)

    @classmethod
    def info(cls, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"{cls.DIM}[{timestamp}]{cls.RESET} {message}", flush=True)

    @classmethod
    def final_answer(cls, answer: str) -> None:
        line = "─" * 70
        print(f"\n{cls.BOLD}{cls.GREEN}{line}{cls.RESET}", flush=True)
        print(f"{cls.BOLD}{cls.GREEN}✔ PIPELINE EXECUTION COMPLETED - FINAL SUMMARY{cls.RESET}", flush=True)
        print(f"{cls.BOLD}{cls.GREEN}{line}{cls.RESET}\n", flush=True)
        print(answer.strip(), flush=True)
        print(f"\n{cls.BOLD}{cls.GREEN}{line}{cls.RESET}\n", flush=True)

    @classmethod
    def warn(cls, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"{cls.BOLD}{cls.YELLOW}⚠ [{timestamp}] {message}{cls.RESET}", flush=True)

    @classmethod
    def spinner(cls, message: str = "Processing...") -> ConsoleSpinner:
        return ConsoleSpinner(message)

    @classmethod
    def error(cls, message: str) -> None:
        print(f"\n{cls.BOLD}{cls.RED}[ERROR] {message}{cls.RESET}\n", file=sys.stderr, flush=True)
