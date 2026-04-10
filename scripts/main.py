import asyncio
import json
import uuid

import aiohttp

from src.toolbox import TOOLS, execute_tool


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

    def _extract_choice(self, payload):
        try:
            return payload["choices"][0]
        except Exception:
            return {}

    def _extract_message_content(self, message, choice=None):
        if not isinstance(message, dict):
            message = {}

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return self._flatten(content)

        if isinstance(choice, dict):
            text = choice.get("text")
            if isinstance(text, str):
                return text

        return ""

    def _normalize_tool_calls(self, tool_calls):
        normalized = []

        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue

            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                fn = {}

            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue

            arguments = fn.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = "{}"

            normalized.append(
                {
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

        return normalized

    def _extract_stream_text(self, payload):
        try:
            choice = payload["choices"][0]
        except Exception:
            return ""

        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return self._flatten(content)

        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return self._flatten(content)

        text = choice.get("text")
        if isinstance(text, str):
            return text

        return ""

    async def _chat_once(self, messages, *, stream, tools=None, tool_choice="auto"):
        payload = {
            "messages": messages,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as r:
            if r.status >= 400:
                raise RuntimeError(await r.text())

            if not stream:
                data = await r.json()
                choice = self._extract_choice(data)
                message = choice.get("message") or {}
                return {
                    "content": self._extract_message_content(message, choice),
                    "tool_calls": self._normalize_tool_calls(message.get("tool_calls") or []),
                    "finish_reason": choice.get("finish_reason"),
                    "raw": data,
                }

            parts = []

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

                text = self._extract_stream_text(payload)
                if text:
                    print(text, end="", flush=True)
                    parts.append(text)

            print()
            return {
                "content": "".join(parts),
                "tool_calls": [],
                "finish_reason": "stop",
            }

    def _build_tool_messages_variant(self, first_content, tool_calls, tool_results, variant):
        """
        Build message history after the first tool-calling assistant turn.

        variant 1:
            assistant content omitted
            tool messages: role/tool_call_id/content
        variant 2:
            assistant content empty string
            tool messages: role/tool_call_id/content
        variant 3:
            assistant content omitted
            tool messages: role/tool_call_id/name/content
        """
        messages = []

        assistant_message = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }

        if variant == 2:
            assistant_message["content"] = ""
        elif first_content:
            assistant_message["content"] = first_content

        messages.append(assistant_message)

        for tc, result in zip(tool_calls, tool_results):
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            }

            if variant == 3:
                tool_msg["name"] = tc["function"]["name"]

            messages.append(tool_msg)

        return messages

    def _local_fallback_answer(self, tool_calls, tool_results):
        if not tool_calls:
            return "No tool result."

        if len(tool_calls) == 1:
            tc = tool_calls[0]
            result = tool_results[0]
            name = tc["function"]["name"]

            if name == "add" and isinstance(result, dict) and "result" in result:
                return str(result["result"])

            if name == "get_time" and isinstance(result, dict):
                return result.get("readable") or result.get("iso") or json.dumps(result, ensure_ascii=False)

            return json.dumps(result, ensure_ascii=False)

        lines = []
        for tc, result in zip(tool_calls, tool_results):
            name = tc["function"]["name"]
            if isinstance(result, dict) and "result" in result:
                lines.append(f"{name}: {result['result']}")
            else:
                lines.append(f"{name}: {json.dumps(result, ensure_ascii=False)}")
        return "\n".join(lines)

    async def ask(self, prompt):
        base_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Use tools when needed. "
                    "When a tool is useful, call it instead of guessing."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # First pass must be non-streaming for reliable tool_calls.
        first = await self._chat_once(
            base_messages,
            stream=False,
            tools=TOOLS,
            tool_choice="auto",
        )

        if not first["tool_calls"]:
            print("=== ANSWER ===")
            print(first["content"])
            return {
                "answer": first["content"],
                "used_tools": False,
            }

        print("=== TOOL EXECUTION ===")
        tool_results = []

        for tc in first["tool_calls"]:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]

            try:
                result = execute_tool(name, raw_args)
            except Exception as e:
                result = {
                    "error": str(e),
                    "tool": name,
                }

            tool_results.append(result)
            print(f"{name}({raw_args}) -> {json.dumps(result, ensure_ascii=False)}")

        # Try a few message-history variants because local servers differ.
        variants = (1, 2, 3)
        last_error = None

        for variant in variants:
            try:
                followup_messages = list(base_messages)
                followup_messages.extend(
                    self._build_tool_messages_variant(
                        first_content=first["content"],
                        tool_calls=first["tool_calls"],
                        tool_results=tool_results,
                        variant=variant,
                    )
                )

                print(f"\n=== ANSWER (variant {variant}) ===")
                final = await self._chat_once(
                    followup_messages,
                    stream=True,
                    tools=None,
                )

                if final["content"]:
                    return {
                        "answer": final["content"],
                        "used_tools": True,
                    }

            except Exception as e:
                last_error = e

        # Final non-stream fallback attempt
        try:
            followup_messages = list(base_messages)
            followup_messages.extend(
                self._build_tool_messages_variant(
                    first_content=first["content"],
                    tool_calls=first["tool_calls"],
                    tool_results=tool_results,
                    variant=1,
                )
            )

            print("\n=== ANSWER (non-stream fallback) ===")
            final = await self._chat_once(
                followup_messages,
                stream=False,
                tools=None,
            )

            if final["content"]:
                print(final["content"])
                return {
                    "answer": final["content"],
                    "used_tools": True,
                }

        except Exception as e:
            last_error = e

        # Absolute fallback: do not crash if server rejects post-tool history.
        fallback = self._local_fallback_answer(first["tool_calls"], tool_results)
        print("\n=== LOCAL FALLBACK ANSWER ===")
        print(fallback)

        return {
            "answer": fallback,
            "used_tools": True,
            "warning": f"Server rejected tool follow-up request: {last_error}",
        }

    async def close(self):
        if self.session:
            await self.session.close()


async def main():
    llm = LLM()
    try:
        await llm.start()
        result = await llm.ask("what is 10 ^ 2.4?")
        # print(result)
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())