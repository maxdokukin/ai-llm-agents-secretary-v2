import argparse
import os
import signal
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' is not installed. Run: pip install python-dotenv")
    sys.exit(1)

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]


class LLMServerStopper:
    def __init__(self, pid_file: Path, timeout: int = 10, force: bool = False):
        self.pid_file = pid_file
        self.timeout = timeout
        self.force = force

    def run(self) -> None:
        pid = self._read_pid()
        if pid is None:
            print(f"No PID file found at: {self.pid_file}")
            print("LLMServer does not appear to be running.")
            return

        if not self._pid_exists(pid):
            print(f"Stale PID file found for PID {pid}. Removing it.")
            self.pid_file.unlink(missing_ok=True)
            return

        print(f"Stopping LLMServer with PID {pid}...")

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print("Process already exited.")
            self.pid_file.unlink(missing_ok=True)
            return
        except PermissionError:
            print(f"Permission denied when trying to stop PID {pid}.", file=sys.stderr)
            sys.exit(1)

        if self._wait_for_exit(pid, self.timeout):
            self.pid_file.unlink(missing_ok=True)
            print("LLMServer stopped successfully.")
            return

        if not self.force:
            print(
                f"LLMServer did not stop within {self.timeout} seconds. "
                "Re-run with --force to send SIGKILL.",
                file=sys.stderr,
            )
            sys.exit(1)

        print("Process did not exit after SIGTERM. Sending SIGKILL...")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Permission denied when trying to kill PID {pid}.", file=sys.stderr)
            sys.exit(1)

        if self._wait_for_exit(pid, 5):
            self.pid_file.unlink(missing_ok=True)
            print("LLMServer killed successfully.")
            return

        print("Failed to stop LLMServer.", file=sys.stderr)
        sys.exit(1)

    def _read_pid(self) -> int | None:
        if not self.pid_file.is_file():
            return None

        try:
            return int(self.pid_file.read_text().strip())
        except Exception:
            print(f"Invalid PID file: {self.pid_file}. Removing it.")
            self.pid_file.unlink(missing_ok=True)
            return None

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _wait_for_exit(pid: int, timeout: int) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not LLMServerStopper._pid_exists(pid):
                return True
            time.sleep(0.25)
        return not LLMServerStopper._pid_exists(pid)


if __name__ == "__main__":
    default_cache_dir = PROJECT_ROOT / "llm" / "cache"
    default_pid_file = Path(os.environ.get("PID_FILE", str(default_cache_dir / ".llmserver.pid")))

    parser = argparse.ArgumentParser(
        description="Stop the background llama-server process started by start_llm_server.py."
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=default_pid_file,
        help="Path to the PID file created by the server starter.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("STOP_TIMEOUT", "10")),
        help="Seconds to wait after SIGTERM before giving up or using --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send SIGKILL if the process does not stop after the timeout.",
    )

    args = parser.parse_args()

    stopper = LLMServerStopper(
        pid_file=args.pid_file,
        timeout=args.timeout,
        force=args.force,
    )
    stopper.run()