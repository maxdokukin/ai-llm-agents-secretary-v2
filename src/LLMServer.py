# LLMServer.py

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

try:
    import aiohttp
except ImportError as exc:
    raise RuntimeError("This implementation requires 'aiohttp'. Install it with: pip install aiohttp") from exc


TokenCallback = Optional[Callable[[str], Awaitable[None] | None]]


class LLMServer:
    REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

    MODEL_PRESETS: Dict[str, Dict[str, object]] = {
        "1.7b": {
            "hf": "Qwen/Qwen3-1.7B-GGUF:Q4_K_M",
            "presence_penalty": 0.8,
        },
        "8b": {
            "hf": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
            "presence_penalty": 1.5,
        },
        "14b": {
            "hf": "Qwen/Qwen3-14B-GGUF:Q4_K_M",
            "presence_penalty": 1.5,
        },
        "32b": {
            "hf": "Qwen/Qwen3-32B-GGUF:Q4_K_M",
            "presence_penalty": 1.5,
        },
    }

    ALIASES = {
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

        self.process: asyncio.subprocess.Process | None = None
        self.current_model_key: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._start_lock = asyncio.Lock()
        self._ask_lock = asyncio.Lock()

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

            await self._ensure_repo()
            await self._ensure_build(rebuild=rebuild)

            binary_path = self._resolve_binary("llama-server")
            cmd = self._build_server_command(binary_path, model_key)

            if self.verbose:
                print("Starting llama.cpp server...")
                print("Project root :", self.proj_root)
                print("llama.cpp dir:", self.llama_cpp_dir)
                print("build dir    :", self.build_dir)
                print("binary       :", binary_path)
                print("model        :", self.MODEL_PRESETS[model_key]["hf"])
                print("server       :", self.base_url)
                print("+", " ".join(cmd))

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.llama_cpp_dir),
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            self.current_model_key = model_key
            await self._wait_until_ready()

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

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
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
                text = await resp.text() if resp.status >= 400 else None
                if resp.status >= 400:
                    raise RuntimeError(f"Request failed with HTTP {resp.status}: {text}")

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
            if key == str(preset["hf"]).lower():
                return preset_key

        allowed = ", ".join(self.MODEL_PRESETS.keys())
        raise ValueError(f"Unsupported model '{model_name}'. Use one of: {allowed}, or a matching HF reference.")

    async def _run(self, *cmd: str, cwd: Path | None = None) -> None:
        if self.verbose:
            print("+", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=sys.stdout,
            stderr=sys.stderr,
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
        if platform.system().lower() == "windows":
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
        if platform.system().lower() == "windows":
            return "Ninja" if shutil.which("ninja") else "Visual Studio 17 2022"
        return "Ninja" if shutil.which("ninja") else "Unix Makefiles"

    async def _ensure_build(self, rebuild: bool = False) -> None:
        if rebuild and self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        if self._build_exists():
            return

        self.build_dir.mkdir(parents=True, exist_ok=True)

        await self._run(
            "cmake",
            "-S",
            str(self.llama_cpp_dir),
            "-B",
            str(self.build_dir),
            "-G",
            self._detect_generator(),
            "-DCMAKE_BUILD_TYPE=Release",
        )
        await self._run(
            "cmake",
            "--build",
            str(self.build_dir),
            "--config",
            "Release",
            "-j",
        )

    def _resolve_binary(self, binary_name: str) -> Path:
        is_windows = platform.system().lower() == "windows"
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

        raise FileNotFoundError(f"Could not find built binary '{binary_name}' under {self.build_dir}")

    def _build_server_command(self, binary_path: Path, model_key: str) -> list[str]:
        preset = self.MODEL_PRESETS[model_key]
        return [
            str(binary_path),
            "-hf",
            str(preset["hf"]),
            "--jinja",
            "--color",
            "auto",
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
                raise RuntimeError(f"llama-server exited early with code {self.process.returncode}")

            for path in ("/health", "/v1/models"):
                try:
                    async with session.get(f"{self.base_url}{path}", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status < 500:
                            return
                except Exception as exc:
                    last_error = str(exc)

            await asyncio.sleep(1)

        raise TimeoutError(f"Timed out waiting for llama-server to become ready. Last error: {last_error}")

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

        content = choice.get("text")
        if isinstance(content, str):
            return content

        return ""

    async def aclose(self) -> None:
        await self.stop()

    async def __aenter__(self) -> "LLMServer":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()