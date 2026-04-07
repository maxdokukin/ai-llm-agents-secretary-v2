from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

try:
    import aiohttp
except ImportError as exc:
    raise RuntimeError(
        "This implementation requires 'aiohttp'. Install it with: pip install aiohttp"
    ) from exc

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError as exc:
    raise RuntimeError(
        "This implementation requires 'huggingface_hub'. Install it with: pip install huggingface_hub"
    ) from exc


TokenCallback = Optional[Callable[[str], Awaitable[None] | None]]


class LLMServer:
    REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

    MODEL_PRESETS: Dict[str, Dict[str, object]] = {
        "1.7b": {
            "hf_ref": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "repo_id": "Qwen/Qwen3-1.7B-GGUF",
            "quant": "Q4_K_M",
            "presence_penalty": 0.8,
        },
        "8b": {
            "hf_ref": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
            "repo_id": "Qwen/Qwen3-8B-GGUF",
            "quant": "Q4_K_M",
            "presence_penalty": 1.5,
        },
        "14b": {
            "hf_ref": "Qwen/Qwen3-14B-GGUF:Q4_K_M",
            "repo_id": "Qwen/Qwen3-14B-GGUF",
            "quant": "Q4_K_M",
            "presence_penalty": 1.5,
        },
        "32b": {
            "hf_ref": "Qwen/Qwen3-32B-GGUF:Q4_K_M",
            "repo_id": "Qwen/Qwen3-32B-GGUF",
            "quant": "Q4_K_M",
            "presence_penalty": 1.5,
        },
    }

    ALIASES = {
        "1.7": "1.7b",
        "qwen3-1.7b": "1.7b",
        "qwen3-8b": "8b",
        "qwen3-14b": "14b",
        "qwen3-32b": "32b",
        "qwen/qwen3-1.7b-gguf:q4_k_m": "1.7b",
        "qwen/qwen3-8b-gguf:q4_k_m": "8b",
        "qwen/qwen3-14b-gguf:q4_k_m": "14b",
        "qwen/qwen3-32b-gguf:q4_k_m": "32b",
    }

    def __init__(
        self,
        proj_root: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        ngl: int = 99,
        ctx_size: int = 8192,
        predict: int = 2048,
        startup_timeout: int = 300,
        request_timeout: int = 3600,
        verbose: bool = True,
    ) -> None:
        if proj_root is None:
            proj_root = Path(__file__).resolve().parent

        self.proj_root = Path(proj_root).resolve()
        self.host = host
        self.port = port
        self.ngl = ngl
        self.ctx_size = ctx_size
        self.predict = predict
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.verbose = verbose

        self.llm_dir = self.proj_root / "llm"
        self.llama_cpp_dir = self.llm_dir / "llama.cpp"
        self.build_dir = self.llama_cpp_dir / "build"
        self.models_dir = self.llm_dir / "models"

        self.process: asyncio.subprocess.Process | None = None
        self.current_model_key: str | None = None
        self.current_model_path: Path | None = None
        self._session: aiohttp.ClientSession | None = None
        self._start_lock = asyncio.Lock()
        self._ask_lock = asyncio.Lock()

        self.dotenv_path = self._find_dotenv()
        self.launch_env = self._build_launch_env()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self, model_name: str, rebuild: bool = False) -> None:
        async with self._start_lock:
            model_key = self._normalize_model_name(model_name)

            if self.is_running() and self.current_model_key == model_key:
                return

            if self.is_running():
                await self.stop()

            self.launch_env = self._build_launch_env()

            await self._ensure_repo()
            await self._ensure_build(rebuild=rebuild)

            model_path = await asyncio.to_thread(self._download_model_file, model_key)
            binary_path = self._resolve_binary("llama-server")
            cmd = self._build_server_command(binary_path, model_key, model_path)

            if self.verbose:
                print("Starting llama.cpp server...")
                print("Project root :", self.proj_root)
                print("llama.cpp dir:", self.llama_cpp_dir)
                print("build dir    :", self.build_dir)
                print("model ref    :", self.MODEL_PRESETS[model_key]["hf_ref"])
                print("model file   :", model_path)
                print("binary       :", binary_path)
                print("server       :", self.base_url)
                print("dotenv       :", self.dotenv_path if self.dotenv_path else "(not found)")
                print("hf token     :", "loaded" if self._get_hf_token() else "missing")
                print("+", " ".join(cmd))

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.llama_cpp_dir),
                stdout=None,
                stderr=None,
                env=self.launch_env,
            )
            self.current_model_key = model_key
            self.current_model_path = model_path
            await self._wait_until_ready()

    async def serve_forever(self, model_name: str, rebuild: bool = False) -> None:
        await self.start(model_name=model_name, rebuild=rebuild)

        if self.verbose:
            print(f"LLM server is ready at {self.base_url}")
            print("Press Ctrl+C to stop.")

        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await self.stop()

    async def ask(
        self,
        prompt: str,
        on_token: TokenCallback = None,
        system_prompt: str | None = None,
    ) -> str:
        if not self.is_running():
            raise RuntimeError("LLM server is not running. Call await start(model_name) first.")

        async with self._ask_lock:
            session = await self._get_session()

            if self.current_model_key is None or self.current_model_path is None:
                raise RuntimeError("No current model is set.")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": str(self.current_model_path),
                "messages": messages,
                "temperature": 0.8,
                "top_k": 20,
                "top_p": 0.95,
                "min_p": 0.0,
                "presence_penalty": self._presence_penalty_for_current_model(),
                "n_predict": self.predict,
                "stream": True,
            }

            url = f"{self.base_url}/v1/chat/completions"
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            parts: list[str] = []

            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status >= 400:
                    err_text = await resp.text()
                    raise RuntimeError(f"Request failed with HTTP {resp.status}: {err_text}")

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    token = self._extract_stream_text(data)
                    if not token:
                        continue

                    parts.append(token)
                    if on_token is not None:
                        result = on_token(token)
                        if asyncio.iscoroutine(result):
                            await result

            return "".join(parts)

    async def stop(self) -> None:
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()

            self.process = None
            self.current_model_key = None
            self.current_model_path = None

        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _normalize_model_name(self, model_name: str) -> str:
        key = model_name.strip().lower()
        if key in self.MODEL_PRESETS:
            return key
        if key in self.ALIASES:
            return self.ALIASES[key]

        for preset_key, preset in self.MODEL_PRESETS.items():
            if key == str(preset["hf_ref"]).lower():
                return preset_key

        allowed = ", ".join(self.MODEL_PRESETS.keys())
        raise ValueError(
            f"Unsupported model '{model_name}'. Use one of: {allowed}, or one of the exact HF refs."
        )

    async def _run(self, *cmd: str, cwd: Path | None = None) -> None:
        if self.verbose:
            print("+", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=None,
            stderr=None,
            env=self.launch_env,
        )
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")

    async def _ensure_repo(self) -> None:
        if self.llama_cpp_dir.exists():
            return

        self.llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
        await self._run("git", "clone", self.REPO_URL, str(self.llama_cpp_dir))

    def _build_exists(self) -> bool:
        bin_dir = self.build_dir / "bin"
        if platform.system() == "Windows":
            candidates = [
                bin_dir / "Release" / "llama-server.exe",
                bin_dir / "llama-server.exe",
            ]
        else:
            candidates = [
                bin_dir / "llama-server",
            ]
        return any(path.exists() for path in candidates)

    def _detect_generator(self) -> str:
        if shutil.which("ninja"):
            return "Ninja"
        if platform.system() == "Windows":
            return "Visual Studio 17 2022"
        return "Unix Makefiles"

    def _cmake_configure_cmd(self) -> list[str]:
        cmd = [
            "cmake",
            "-S",
            str(self.llama_cpp_dir),
            "-B",
            str(self.build_dir),
            "-G",
            self._detect_generator(),
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if platform.system() == "Darwin":
            cmd.extend(
                [
                    "-DCMAKE_OSX_ARCHITECTURES=arm64",
                    "-DGGML_METAL=ON",
                ]
            )

        return cmd

    async def _ensure_build(self, rebuild: bool = False) -> None:
        if rebuild and self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        if self._build_exists():
            return

        self.build_dir.mkdir(parents=True, exist_ok=True)

        await self._run(*self._cmake_configure_cmd())
        await self._run(
            "cmake",
            "--build",
            str(self.build_dir),
            "--config",
            "Release",
            "-j",
        )

    def _resolve_binary(self, binary_name: str) -> Path:
        is_windows = platform.system() == "Windows"
        suffix = ".exe" if is_windows else ""

        candidates: list[Path] = []
        if is_windows:
            candidates.extend(
                [
                    self.build_dir / "bin" / "Release" / f"{binary_name}{suffix}",
                    self.build_dir / "bin" / f"{binary_name}{suffix}",
                ]
            )
        else:
            candidates.append(self.build_dir / "bin" / f"{binary_name}{suffix}")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find built binary '{binary_name}' under {self.build_dir}"
        )

    def _download_model_file(self, model_key: str) -> Path:
        preset = self.MODEL_PRESETS[model_key]
        repo_id = str(preset["repo_id"])
        quant = str(preset["quant"])
        token = self._get_hf_token()

        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")

        filename = self._select_gguf_filename(files, quant)
        if filename is None:
            ggufs = sorted([f for f in files if f.lower().endswith(".gguf")])
            raise RuntimeError(
                f"Could not find a GGUF file for repo '{repo_id}' with quant '{quant}'. "
                f"Available GGUF files: {ggufs}"
            )

        repo_dir = self.models_dir / repo_id.replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            local_dir=str(repo_dir),
            local_dir_use_symlinks=False,
        )

        return Path(local_path).resolve()

    def _select_gguf_filename(self, files: list[str], quant: str) -> str | None:
        quant_key = quant.lower().replace("-", "_")
        ggufs = [f for f in files if f.lower().endswith(".gguf")]

        if not ggufs:
            return None

        def norm(s: str) -> str:
            return s.lower().replace("-", "_")

        exact = [f for f in ggufs if norm(Path(f).name).endswith(f"{quant_key}.gguf")]
        if exact:
            exact.sort(key=lambda x: ("/" in x, len(x), x.lower()))
            return exact[0]

        contains = [f for f in ggufs if quant_key in norm(Path(f).name)]
        if contains:
            contains.sort(key=lambda x: ("/" in x, len(x), x.lower()))
            return contains[0]

        if len(ggufs) == 1:
            return ggufs[0]

        return None

    def _build_server_command(self, binary_path: Path, model_key: str, model_path: Path) -> list[str]:
        preset = self.MODEL_PRESETS[model_key]
        return [
            str(binary_path),
            "-m",
            str(model_path),
            "--jinja",
            "-ngl",
            str(self.ngl),
            "-fa",
            "auto",
            "--temp",
            "0.8",
            "--top-k",
            "20",
            "--top-p",
            "0.95",
            "--min-p",
            "0",
            "--presence-penalty",
            str(preset["presence_penalty"]),
            "-c",
            str(self.ctx_size),
            "-n",
            str(self.predict),
            "--no-context-shift",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.startup_timeout
        last_error: str | None = None
        session = await self._get_session()

        while asyncio.get_running_loop().time() < deadline:
            if self.process is not None and self.process.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {self.process.returncode}"
                )

            for path in ("/health", "/v1/models", "/models"):
                try:
                    async with session.get(
                        f"{self.base_url}{path}",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status < 500:
                            return
                except Exception as exc:
                    last_error = str(exc)

            await asyncio.sleep(1)

        raise TimeoutError(
            f"Timed out waiting for llama-server to become ready. Last error: {last_error}"
        )

    def _presence_penalty_for_current_model(self) -> float:
        if self.current_model_key is None:
            return 1.5
        return float(self.MODEL_PRESETS[self.current_model_key]["presence_penalty"])

    @staticmethod
    def _extract_stream_text(data: dict) -> str:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            return ""

        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content

        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        text = choice.get("text")
        if isinstance(text, str):
            return text

        return ""

    def _find_dotenv(self) -> Path | None:
        candidates: list[Path] = []

        for start in [self.proj_root, Path.cwd(), Path(__file__).resolve().parent]:
            current = start.resolve()
            candidates.append(current / ".env")
            candidates.extend(parent / ".env" for parent in current.parents)

        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if path.exists() and path.is_file():
                return path

        return None

    def _parse_dotenv(self, path: Path) -> dict[str, str]:
        values: dict[str, str] = {}

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            values[key] = value

        return values

    def _build_launch_env(self) -> dict[str, str]:
        env = dict(os.environ)

        dotenv_path = self._find_dotenv()
        if dotenv_path is not None:
            dotenv_values = self._parse_dotenv(dotenv_path)
            for key, value in dotenv_values.items():
                env[key] = value

        token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

        return env

    def _get_hf_token(self) -> str | None:
        return (
            self.launch_env.get("HF_TOKEN")
            or self.launch_env.get("HUGGING_FACE_HUB_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

    async def aclose(self) -> None:
        await self.stop()

    async def __aenter__(self) -> "LLMServer":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run llama.cpp-backed local LLM server with model auto-download."
    )
    parser.add_argument(
        "--model",
        default="8b",
        help="Model preset or exact HF ref. Examples: 1.7b, 8b, 14b, 32b",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind llama-server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind llama-server to.",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="Number of GPU layers to offload.",
    )
    parser.add_argument(
        "--ctx-size",
        type=int,
        default=8192,
        help="Context size.",
    )
    parser.add_argument(
        "--predict",
        type=int,
        default=2048,
        help="Max generated tokens.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=300,
        help="Seconds to wait for server startup.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=3600,
        help="Seconds to wait for request completion.",
    )
    parser.add_argument(
        "--proj-root",
        default=None,
        help="Optional project root. Defaults to this file's directory.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of llama.cpp.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported model presets and exit.",
    )
    return parser


def _print_supported_models() -> None:
    print("Supported model presets:")
    for key, preset in LLMServer.MODEL_PRESETS.items():
        print(f"  {key:<5} -> {preset['hf_ref']}")


async def _async_main(args: argparse.Namespace) -> int:
    if args.list_models:
        _print_supported_models()
        return 0

    server = LLMServer(
        proj_root=args.proj_root,
        host=args.host,
        port=args.port,
        ngl=args.ngl,
        ctx_size=args.ctx_size,
        predict=args.predict,
        startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout,
        verbose=not args.quiet,
    )

    await server.serve_forever(model_name=args.model, rebuild=args.rebuild)
    return 0


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())