import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from huggingface_hub import list_repo_files, snapshot_download
except ImportError:
    print("Error: 'huggingface_hub' is not installed. Run: pip install huggingface_hub")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' is not installed. Run: pip install python-dotenv")
    sys.exit(1)

load_dotenv()


class LLMServer:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(PROJECT_ROOT / "llm" / "models")))

    ALLOWED_REPOS = {
        "ggml-org/gemma-4-E2B-it-GGUF",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "ggml-org/gemma-4-31B-it-GGUF",
    }

    MODEL_ALIASES = {
        "e2b": "ggml-org/gemma-4-E2B-it-GGUF",
        "e4b": "ggml-org/gemma-4-E4B-it-GGUF",
        "26b": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "31b": "ggml-org/gemma-4-31B-it-GGUF",
        "5b": "ggml-org/gemma-4-E2B-it-GGUF",
        "8b": "ggml-org/gemma-4-E4B-it-GGUF",
        "25b": "ggml-org/gemma-4-26B-A4B-it-GGUF",
        "31b_params": "ggml-org/gemma-4-31B-it-GGUF",
        "1.7b": "ggml-org/gemma-4-E2B-it-GGUF",
    }

    REPO_DEFAULT_FILES = {
        "ggml-org/gemma-4-E2B-it-GGUF": "gemma-4-e2b-it-Q8_0.gguf",
        "ggml-org/gemma-4-E4B-it-GGUF": "gemma-4-e4b-it-Q4_K_M.gguf",
        "ggml-org/gemma-4-26B-A4B-it-GGUF": "gemma-4-26b-a4b-it-Q4_K_M.gguf",
        "ggml-org/gemma-4-31B-it-GGUF": "gemma-4-31b-it-Q4_K_M.gguf",
    }

    PREFERRED_MARKERS = [
        "Q4_K_M",
        "Q8_0",
        "F16",
    ]

    def __init__(self, model_arg: str, parameters: str, port: int):
        self.model_path = str(self.resolve_model(model_arg))
        self.parameters = parameters
        self.port = port
        self.process = None

    @classmethod
    def _parse_hf_spec(cls, model_arg: str) -> Tuple[str, Optional[str]]:
        parts = model_arg.strip().split("/")
        if len(parts) < 2:
            raise ValueError("Invalid Hugging Face spec")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:]) if len(parts) > 2 else None
        return repo_id, filename

    @classmethod
    def _validate_repo(cls, repo_id: str) -> None:
        if repo_id not in cls.ALLOWED_REPOS:
            allowed = "\n".join(f"  - {repo}" for repo in sorted(cls.ALLOWED_REPOS))
            print(
                "Error: unsupported model repo.\n"
                "Allowed repos are:\n"
                f"{allowed}",
                flush=True,
            )
            sys.exit(1)

    @classmethod
    def _pick_default_gguf(cls, repo_id: str) -> str:
        cls._validate_repo(repo_id)

        repo_files = list_repo_files(repo_id=repo_id, repo_type="model")
        gguf_files = [
            f for f in repo_files
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower()
        ]

        if not gguf_files:
            raise RuntimeError(f"No non-mmproj .gguf files found in repo '{repo_id}'")

        configured_default = cls.REPO_DEFAULT_FILES.get(repo_id)
        if configured_default in gguf_files:
            return configured_default

        def score(name: str):
            upper = name.upper()
            for idx, marker in enumerate(cls.PREFERRED_MARKERS):
                if marker in upper:
                    return (idx, name)
            return (len(cls.PREFERRED_MARKERS), name)

        gguf_files.sort(key=score)
        return gguf_files[0]

    @classmethod
    def _download_hf_model(cls, repo_id: str, filename: Optional[str]) -> Path:
        cls._validate_repo(repo_id)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = cls._pick_default_gguf(repo_id)

        local_repo_dir = cls.MODELS_DIR / repo_id.replace("/", "__")
        local_repo_dir.mkdir(parents=True, exist_ok=True)

        local_target = local_repo_dir / Path(filename).name
        if local_target.is_file():
            print(f"Model found locally at: {local_target}", flush=True)
            return local_target.resolve()

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "Warning: No HF_TOKEN found. Unauthenticated downloads may be slower or rate-limited.",
                flush=True,
            )

        print("Model not found locally.", flush=True)
        print(f"Repo        : {repo_id}", flush=True)
        print(f"File        : {filename}", flush=True)
        print(f"Destination : {local_repo_dir}", flush=True)
        print("Downloading model with progress...\n", flush=True)

        try:
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[filename],
                local_dir=str(local_repo_dir),
                token=hf_token,
                resume_download=True,
            )
        except Exception as exc:
            print(f"\nFailed to download model from Hugging Face: {exc}", flush=True)
            sys.exit(1)

        downloaded_file = Path(snapshot_path) / filename
        if downloaded_file.is_file():
            print(f"\nDownload complete: {downloaded_file}", flush=True)
            return downloaded_file.resolve()

        candidates = list(local_repo_dir.rglob(Path(filename).name))
        if len(candidates) == 1 and candidates[0].is_file():
            print(f"\nDownload complete: {candidates[0]}", flush=True)
            return candidates[0].resolve()

        print(
            f"\nError: download finished but could not locate '{filename}' under {local_repo_dir}",
            flush=True,
        )
        sys.exit(1)

    @classmethod
    def resolve_model(cls, model_arg: str) -> Path:
        model_arg = model_arg.strip()
        if not model_arg:
            print("Error: --model cannot be empty.", flush=True)
            sys.exit(1)

        model_arg = cls.MODEL_ALIASES.get(model_arg, model_arg)

        local_path = Path(model_arg).expanduser()
        if local_path.is_file():
            return local_path.resolve()

        fallback = cls.MODELS_DIR / model_arg
        if fallback.is_file():
            return fallback.resolve()

        try:
            repo_id, filename = cls._parse_hf_spec(model_arg)
        except ValueError:
            print(f"Error: Model '{model_arg}' not found locally or format is invalid.", flush=True)
            sys.exit(1)

        cls._validate_repo(repo_id)
        return cls._download_hf_model(repo_id=repo_id, filename=filename)

    @staticmethod
    def build_command(model_path: str, parameters: str, port: int) -> list[str]:
        command = [
            "llama-server",
            "--model",
            model_path,
            "--port",
            str(port),
        ]

        if parameters:
            try:
                command.extend(shlex.split(parameters))
            except ValueError as exc:
                print(f"Error parsing parameters: {exc}", flush=True)
                sys.exit(1)

        return command

    def start(self):
        command = self.build_command(
            model_path=self.model_path,
            parameters=self.parameters,
            port=self.port,
        )

        print(f"Starting LLM Server...\nCommand: {' '.join(command)}\n", flush=True)

        try:
            self.process = subprocess.Popen(command)
            self.process.wait()
        except FileNotFoundError:
            print("Error: 'llama-server' executable not found in PATH.", flush=True)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nShutting down the LLM server safely...", flush=True)
            self.stop()

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            print("Server successfully stopped.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a llama.cpp server, auto-downloading a GGUF from the allowed ggml-org repos if needed."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Accepted values:\n"
            "  - local .gguf path\n"
            "  - ggml-org/gemma-4-E2B-it-GGUF\n"
            "  - ggml-org/gemma-4-E4B-it-GGUF\n"
            "  - ggml-org/gemma-4-26B-A4B-it-GGUF\n"
            "  - ggml-org/gemma-4-31B-it-GGUF\n"
            "  - alias: e2b, e4b, 26b, 31b, 5b, 8b, 25b\n"
        ),
    )
    parser.add_argument("--parameters", type=str, default="", help="Additional llama-server parameters")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Resolve/download the model, print its final local path, then exit.",
    )

    args = parser.parse_args()

    if args.resolve_only:
        resolved = LLMServer.resolve_model(args.model)
        print(f"RESOLVED_MODEL={resolved}", flush=True)
        sys.exit(0)

    server = LLMServer(model_arg=args.model, parameters=args.parameters, port=args.port)
    server.start()