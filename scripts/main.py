# main.py

import asyncio
from src.LLMServer import LLMServer


class LLM:
    """
    Drop-in async wrapper:
        await llm.start("32b")
        text = await llm.ask("prompt")
    """

    def __init__(self, proj_root=None) -> None:
        self.server = LLMServer(proj_root=proj_root)

    async def start(self, model_name: str) -> None:
        await self.server.start(model_name)

    async def ask(self, prompt: str) -> str:
        async def on_token(token: str) -> None:
            print(token, end="", flush=True)

        return await self.server.ask(prompt, on_token=on_token)

    async def stop(self) -> None:
        await self.server.stop()


async def main() -> None:
    llm = LLM()

    try:
        await llm.start("1.7b")
        print("\n=== RESPONSE ===\n")
        text = await llm.ask("Write a one paragraph summary of tool calling for local LLMs.")
        print("\n\n=== FINAL TEXT ===\n")
        print(text)
    finally:
        await llm.stop()


if __name__ == "__main__":
    asyncio.run(main())