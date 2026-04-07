import asyncio
import json
import aiohttp


class LLM:
    def __init__(self, host="127.0.0.1", port=8080):
        self.base_url = f"http://{host}:{port}"
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession()
        for path in ("/health", "/v1/models", "/models"):
            try:
                async with self.session.get(
                    f"{self.base_url}{path}",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as r:
                    if r.status < 500:
                        return
            except Exception:
                pass
        await self.session.close()
        raise SystemExit(f"No running LLM server found at {self.base_url}")

    def _flatten(self, value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or item.get("value") or ""
                    if text:
                        parts.append(text)
            return "".join(parts)
        if isinstance(value, dict):
            return value.get("text") or value.get("content") or value.get("value") or ""
        return str(value)

    def _extract_stream_parts(self, payload):
        """
        Return (thinking_text, answer_text, finish_reason).

        Streaming-only extraction from delta fields to avoid duplicating
        the final answer when some servers also include message/content.
        """
        try:
            choice = payload["choices"][0]
        except Exception:
            return "", "", None

        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        thinking = (
            self._flatten(delta.get("reasoning_content"))
            or self._flatten(delta.get("reasoning"))
            or self._flatten(delta.get("thinking"))
            or self._flatten(delta.get("thought"))
        )

        answer = self._flatten(delta.get("content"))

        return thinking, answer, finish_reason

    async def ask(self, prompt):
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=600),
        ) as r:
            if r.status >= 400:
                raise RuntimeError(await r.text())

            thinking_parts = []
            answer_parts = []

            thinking_started = False
            answer_started = False

            async for raw in r.content:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    payload = json.loads(data)
                except Exception:
                    continue

                thinking, answer, finish_reason = self._extract_stream_parts(payload)

                if thinking:
                    if not thinking_started:
                        if answer_started:
                            print()
                        print("=== THINKING ===")
                        thinking_started = True
                    print(thinking, end="", flush=True)
                    thinking_parts.append(thinking)

                if answer:
                    if not answer_started:
                        if thinking_started:
                            print("\n\n=== ANSWER ===")
                        else:
                            print("=== ANSWER ===")
                        answer_started = True
                    print(answer, end="", flush=True)
                    answer_parts.append(answer)

                if finish_reason is not None:
                    break

            if thinking_started or answer_started:
                print()

            return {
                "thinking": "".join(thinking_parts),
                "answer": "".join(answer_parts),
            }

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    llm = LLM()
    try:
        await llm.start()
        result = await llm.ask("Hello LLM")
        #
        # print("\n--- structured result ---")
        # print("thinking:", repr(result["thinking"]))
        # print("answer:", repr(result["answer"]))
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())