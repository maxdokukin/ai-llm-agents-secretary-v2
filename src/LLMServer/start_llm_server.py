import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' is not installed. Run: pip install python-dotenv")
    sys.exit(1)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.LLMServer.LLMServer import LLMServer

load_dotenv()


class LLMServerStarter:
    def __init__(
        self,
        model: str,
        parameters: str,
        port: int,
        cache_dir: Path,
        python_bin: str,
        server_script: Path,
        log_file: Path,
        pid_file: Path,
        models_dir: Path,
    ):
        self.model = model
        self.parameters = parameters
        self.port = port
        self.cache_dir = cache_dir
        self.python_bin = python_bin
        self.server_script = server_script
        self.log_file = log_file
        self.pid_file = pid_file
        self.models_dir = models_dir

    def setup_environment(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "huggingface").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "xdg").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("HF_HOME", str(self.cache_dir / "huggingface"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cache_dir / "huggingface" / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir / "huggingface" / "transformers"))
        os.environ.setdefault("XDG_CACHE_HOME", str(self.cache_dir / "xdg"))
        os.environ["MODELS_DIR"] = str(self.models_dir)

    def check_existing_process(self) -> None:
        if not self.pid_file.is_file():
            return

        try:
            old_pid = int(self.pid_file.read_text().strip())
        except Exception:
            self.pid_file.unlink(missing_ok=True)
            return

        if self._pid_exists(old_pid):
            server_ip = self.get_server_ip()
            print(f"LLMServer already running with PID {old_pid}")
            print(f"Port       : {self.port}")
            print(f"Server IP  : {server_ip}")
            print(f"URL        : http://{server_ip}:{self.port}")
            print(f"Cache      : {self.cache_dir}")
            print(f"Models dir : {self.models_dir}")
            sys.exit(0)

        self.pid_file.unlink(missing_ok=True)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def get_server_ip() -> str:
        """
        Return the machine's primary local network IP.
        Falls back to 127.0.0.1 if detection fails.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No traffic is actually sent; this is used to select the outbound interface.
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except Exception:
            return "127.0.0.1"
        finally:
            sock.close()

    def validate_paths(self) -> None:
        python_path = Path(self.python_bin)
        if not python_path.is_file():
            print(f"Python not found: {self.python_bin}", file=sys.stderr)
            sys.exit(1)

        if not self.server_script.is_file():
            print(f"Server script not found: {self.server_script}", file=sys.stderr)
            sys.exit(1)

    def print_header(self) -> None:
        server_ip = self.get_server_ip()
        print("Preparing LLMServer...")
        print(f"Python     : {self.python_bin}")
        print(f"Script     : {self.server_script}")
        print(f"Model      : {self.model}")
        print(f"Parameters : {self.parameters or 'None'}")
        print(f"Port       : {self.port}")
        print(f"Server IP  : {server_ip}")
        print(f"URL        : http://{server_ip}:{self.port}")
        print(f"Log        : {self.log_file}")
        print(f"Cache      : {self.cache_dir}")
        print(f"Models dir : {self.models_dir}")
        print()

    def resolve_model_foreground(self) -> str:
        resolved_model = str(LLMServer.resolve_model(self.model))
        print(f"RESOLVED_MODEL={resolved_model}", flush=True)
        return resolved_model

    def start_background_server(self, resolved_model: str) -> int:
        command = LLMServer.build_command(
            model_path=resolved_model,
            parameters=self.parameters,
            port=self.port,
        )

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text("")

        print()
        print("Starting llama-server in background...")

        try:
            log_handle = open(self.log_file, "a", buffering=1)
        except OSError as exc:
            print(f"Failed to open log file '{self.log_file}': {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            log_handle.close()
            print("Error: 'llama-server' executable not found in PATH.", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            log_handle.close()
            print(f"Failed to start llama-server: {exc}", file=sys.stderr)
            sys.exit(1)

        self.pid_file.write_text(f"{process.pid}\n")

        time.sleep(2)

        if process.poll() is None:
            server_ip = self.get_server_ip()
            print(f"LLMServer started with PID {process.pid}")
            print(f"Server IP  : {server_ip}")
            print(f"URL        : http://{server_ip}:{self.port}")
            print(f"Local URL  : http://127.0.0.1:{self.port}")
            return process.pid

        print(f"LLMServer failed to start. Check log: {self.log_file}", file=sys.stderr)
        self.pid_file.unlink(missing_ok=True)
        sys.exit(1)

    def run(self) -> None:
        self.setup_environment()
        self.check_existing_process()
        self.validate_paths()
        self.print_header()
        resolved_model = self.resolve_model_foreground()
        self.start_background_server(resolved_model)


if __name__ == "__main__":
    project_root = PROJECT_ROOT
    default_cache_dir = project_root / "llm" / "cache"

    parser = argparse.ArgumentParser(
        description="Resolve/download an allowed ggml-org Gemma 4 model in the foreground, then start llama-server in the background."
    )
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "e2b"))
    parser.add_argument("--parameters", type=str, default=os.environ.get("PARAMETERS", ""))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("CACHE_DIR", str(default_cache_dir))),
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=os.environ.get("PYTHON_BIN", sys.executable),
    )
    parser.add_argument(
        "--server-script",
        type=Path,
        default=Path(
            os.environ.get(
                "SERVER_SCRIPT",
                str(project_root / "src" / "LLMServer" / "LLMServer.py"),
            )
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(os.environ.get("LOG_FILE", str(default_cache_dir / ".llmserver.log"))),
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=Path(os.environ.get("PID_FILE", str(default_cache_dir / ".llmserver.pid"))),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("MODELS_DIR", str(project_root / "llm" / "models"))),
    )

    args = parser.parse_args()

    starter = LLMServerStarter(
        model=args.model,
        parameters=args.parameters,
        port=args.port,
        cache_dir=args.cache_dir,
        python_bin=args.python_bin,
        server_script=args.server_script,
        log_file=args.log_file,
        pid_file=args.pid_file,
        models_dir=args.models_dir,
    )
    starter.run()