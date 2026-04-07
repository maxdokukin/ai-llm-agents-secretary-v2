import asyncio
import json

import aiohttp


class LLM:
    """
    Client for an already-running LLM server.
    If no server is running, terminate immediately.
    """

    def __init__(
        self,
        model: str = "8b",
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def start(self) -> None:
        if not await self._is_server_ready():
            raise SystemExit(f"No running LLM server found at {self.base_url}")
        print(f"Using running server at {self.base_url}")

    async def _is_server_ready(self) -> bool:
        session = await self._get_session()

        for path in ("/health", "/v1/models", "/models"):
            try:
                async with session.get(
                    f"{self.base_url}{path}",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status < 500:
                        return True
            except Exception:
                pass

        return False

    async def ask(self, prompt: str, system_prompt: str | None = None) -> str:
        session = await self._get_session()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.8,
            "top_k": 20,
            "top_p": 0.95,
            "min_p": 0.0,
            "n_predict": 512,
            "stream": True,
        }

        url = f"{self.base_url}/v1/chat/completions"
        parts: list[str] = []

        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
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

                print(token, end="", flush=True)
                parts.append(token)

        return "".join(parts)

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

    async def close(self) -> None:
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None


async def main() -> None:
    llm = LLM(
        model="8b",
        host="127.0.0.1",
        port=8080,
    )

    try:
        await llm.start()

        print("\n=== RESPONSE ===\n")
        text = await llm.ask(
            "Write a one paragraph summary of tool calling for local LLMs."
        )

        print("\n\n=== FINAL TEXT ===\n")
        print(text)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())