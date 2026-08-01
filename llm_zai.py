import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from llm.default_plugins.openai_models import (
    _attachment as openai_attachment,
    combine_chunks,
    remove_dict_none_values,
)
from pydantic import Field
from typing import Optional, List, Dict, Any, Iterator, Union, AsyncGenerator
from dataclasses import is_dataclass, asdict
import json

DEFAULT_API_BASE = "https://api.z.ai/api/paas/v4"
CODING_API_BASE = "https://api.z.ai/api/coding/paas/v4"

# Models known to be available on Z.AI's OpenAI-compatible PaaS endpoint.
# (vision, supports_schema, supports_tools)
MODELS = [
    # GLM-5 series
    ("glm-5.2", False, True, True),
    ("glm-5.1", False, True, True),
    ("glm-5", False, True, True),
    ("glm-5-turbo", False, True, True),
    # Vision / multimodal
    ("glm-ocr", True, False, False),
    ("glm-5v-turbo", True, False, False),
    # Long-context / agents
    
    ("autoglm-phone-multilingual", False, False, False),
]

# Coding-plan models. The API expects the real model id without the prefix.
CODING_MODELS = [
    ("glm-5.2-coding", "glm-5.2", True, True, True),
    ("glm-5.1-coding", "glm-5.1", True, True, True),
]

CODING_ALIASES = {alias for alias, *_ in CODING_MODELS}

# GLM-5.2 supports thinking-effort levels high and max.
EFFORT_MODELS = {"glm-5.2", "glm-5.2-coding", "glm-5.1", "glm-5.1-coding"}

API_BASE_FOR_MODEL = {}
for name, _, _, _ in MODELS:
    API_BASE_FOR_MODEL[name] = DEFAULT_API_BASE
for alias, real_name, _, _, _ in CODING_MODELS:
    API_BASE_FOR_MODEL[alias] = CODING_API_BASE


def _reasoning_content_from_response_json(rj: Any, conversation=None) -> Optional[str]:
    """Return a string reasoning_content, or None.

    llm condenses logged response_json by replacing repeated substrings
    (often the assistant response text) with objects like {"$": "r:<id>"}
    or {"$r": [...]}. Z.AI rejects non-string reasoning_content in
    assistant messages with HTTP 400 code 1210, so normalize/expand any
    condensed form back to text before sending it.
    """
    if not isinstance(rj, dict):
        return None

    rc = rj.get("reasoning_content")
    if rc is None and "choices" in rj:
        for c in rj["choices"]:
            msg = c.get("message", {}) if isinstance(c, dict) else {}
            rc = msg.get("reasoning_content")
            if rc:
                break

    if isinstance(rc, str):
        return rc

    # DB rows may contain condensed JSON. Try to expand it using the
    # conversation's response texts (replacement ids "r:<response_id>").
    if rc is not None and conversation is not None:
        replacements = {}
        for resp in getattr(conversation, "responses", []) or []:
            rid = getattr(resp, "id", None)
            if not rid:
                continue
            try:
                text = resp.text_or_raise()
            except Exception:
                text = "".join(getattr(resp, "_chunks", []) or [])
            if text:
                replacements[f"r:{rid}"] = text
        try:
            from condense_json import uncondense_json

            expanded = uncondense_json(rc, replacements)
            if isinstance(expanded, str):
                return expanded
        except Exception:
            pass

    return None


def _batch_get_reasoning(conversation):
    """Get reasoning_content for all responses in a conversation.

    Tries response_json first (fast path), then falls back to querying
    the SQLite logs DB's reasoning column directly — the only reliable
    source when ReasoningParts are lost during DB round-trip via from_row.

    Returns a dict of {response_id: reasoning_text}.
    """
    if not conversation or not hasattr(conversation, "responses"):
        return {}

    reasoning_map = {}
    missing_ids = []

    for resp in conversation.responses:
        rj = getattr(resp, "response_json", None) or {}
        rc = _reasoning_content_from_response_json(rj, conversation)
        if rc:
            reasoning_map[resp.id] = rc
        else:
            missing_ids.append(resp.id)

    # Batch query the DB's reasoning column for responses where
    # response_json didn't have reasoning_content.
    if missing_ids:
        try:
            import sqlite3
            db_path = str(llm.user_dir() / "logs.db")
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            placeholders = ",".join("?" * len(missing_ids))
            cur = conn.execute(
                f"SELECT id, reasoning FROM responses WHERE id IN ({placeholders})",
                missing_ids,
            )
            for row in cur:
                if row[1]:
                    reasoning_map[row[0]] = row[1]
            conn.close()
        except Exception:
            pass

    return reasoning_map


def _combine_chunks_with_reasoning(chunks: List) -> dict:
    """Like combine_chunks but also captures reasoning_content from deltas.

    The upstream combine_chunks only looks at delta.content; for preserved
    thinking we need reasoning_content stored in response_json so it can be
    round-tripped back to the API on subsequent turns.
    """
    combined = combine_chunks(chunks)
    reasoning_content = ""
    for item in chunks:
        for choice in item.choices:
            delta = getattr(choice, "delta", None)
            if delta:
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_content += rc
    if reasoning_content:
        combined["reasoning_content"] = reasoning_content
    return combined


class ZaiOptionsMixin:
    class Options(Chat.Options):
        coding: Optional[bool] = Field(
            None,
            description="Use the GLM Coding Plan endpoint (https://api.z.ai/api/coding/paas/v4)",
        )
        thinking_type: Optional[str] = Field(
            "enabled",
            description="Thinking mode: enabled or disabled",
        )
        effort: Optional[str] = Field(
            "max",
            description="Thinking effort for supported models (glm-5.2): high or max",
        )
        clear_thinking: Optional[bool] = Field(
            None,
            description=(
                "Preserve (false) or clear (true) reasoning history across turns. "
                "Defaults to false on coding endpoints, true on standard endpoints."
            ),
        )


class _ZaiShared:
    """Shared logic for sync and async Z.AI GLM chat models."""

    needs_key = "zai"
    key_env_var = "GLM_API_KEY"

    def __str__(self):
        return "Z.AI GLM: {}".format(self.model_id)

    def _is_coding(self) -> bool:
        return getattr(self, "_coding", False) or self.model_id in CODING_ALIASES

    def _append_zai_message(self, out, message, current_system, image_detail=None):
        """Translate one llm.Message into Z.AI-compatible OpenAI message dicts.

        Key ordering for preserved/interleaved thinking:
          assistant messages include reasoning_content alongside content and
          tool_calls, matching the sequence the model originally produced.
        """
        from llm.parts import (
            TextPart,
            ReasoningPart,
            AttachmentPart,
            ToolCallPart,
            ToolResultPart,
        )

        text_bits = []
        reasoning_bits = []
        attachment_items = []
        tool_calls = []
        tool_results = []

        for part in message.parts:
            if isinstance(part, TextPart):
                text_bits.append(part.text)
            elif isinstance(part, ReasoningPart):
                if part.text:
                    reasoning_bits.append(part.text)
            elif isinstance(part, AttachmentPart) and part.attachment:
                attachment_items.append(
                    openai_attachment(part.attachment, image_detail=image_detail)
                )
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    {
                        "type": "function",
                        "id": part.tool_call_id,
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(part.arguments),
                        },
                    }
                )
            elif isinstance(part, ToolResultPart):
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": part.output,
                    }
                )

        # Tool messages are emitted as separate role="tool" entries.
        if message.role == "tool":
            out.extend(tool_results)
            return current_system

        if message.role == "system":
            text = "".join(text_bits)
            if text == current_system:
                return current_system
            current_system = text

        # Build the message entry in the order Z.AI expects:
        # role -> content -> reasoning_content -> tool_calls
        if attachment_items:
            content = []
            if text_bits:
                content.append({"type": "text", "text": "".join(text_bits)})
            content.extend(attachment_items)
            entry = {"role": message.role, "content": content}
        else:
            entry = {
                "role": message.role,
                "content": "".join(text_bits) if text_bits else None,
            }

        # Preserved / interleaved thinking: include reasoning_content on
        # assistant messages. Consecutive reasoning parts are concatenated.
        if reasoning_bits and message.role == "assistant":
            entry["reasoning_content"] = "".join(reasoning_bits)

        if tool_calls:
            entry["tool_calls"] = tool_calls
            if not text_bits:
                entry["content"] = None
        elif entry["content"] is None and message.role != "assistant":
            return current_system

        out.append(entry)
        return current_system

    def build_messages(self, prompt, conversation, image_detail=None):
        messages = []
        current_system = None
        if image_detail is not None:
            image_detail = image_detail.value

        # Preserved thinking: collect reasoning_content from prior responses.
        # prompt.messages is authoritative in llm >= 0.32; conversation is used
        # only to recover reasoning lost during DB round-trips.
        reasoning_map = _batch_get_reasoning(conversation)
        reasoning_by_assistant_position = []
        if conversation is not None and hasattr(conversation, "responses"):
            for resp in conversation.responses:
                reasoning_by_assistant_position.append(
                    reasoning_map.get(getattr(resp, "id", None))
                )

        assistant_message_indexes = []
        for msg in prompt.messages:
            current_system = self._append_zai_message(
                messages, msg, current_system, image_detail=image_detail
            )
            if msg.role == "assistant":
                assistant_message_indexes.append(len(messages) - 1)

        if reasoning_by_assistant_position:
            for position, message_index in enumerate(assistant_message_indexes):
                if position >= len(reasoning_by_assistant_position):
                    break
                reasoning = reasoning_by_assistant_position[position]
                entry = messages[message_index]
                if (
                    reasoning
                    and entry.get("role") == "assistant"
                    and "reasoning_content" not in entry
                ):
                    # Preserve key order: reasoning_content before tool_calls.
                    if "tool_calls" in entry:
                        tool_calls = entry.pop("tool_calls")
                        entry["reasoning_content"] = reasoning
                        entry["tool_calls"] = tool_calls
                    else:
                        entry["reasoning_content"] = reasoning

        return messages

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("coding", None)
        thinking_type = kwargs.pop("thinking_type", None)
        effort = kwargs.pop("effort", None)
        clear_thinking = kwargs.pop("clear_thinking", None)

        # Default clear_thinking based on endpoint:
        # Coding endpoint preserves (False), standard endpoint clears (True).
        if clear_thinking is None:
            clear_thinking = False if self._is_coding() else True

        # Build the thinking object with clear_thinking INSIDE it (per Z.AI spec).
        thinking: Dict[str, Any] = {}
        if thinking_type:
            thinking["type"] = thinking_type
        thinking["clear_thinking"] = clear_thinking
        if effort and self.model_id in EFFORT_MODELS:
            thinking["effort"] = effort

        extra_body = kwargs.get("extra_body") or {}
        extra_body["thinking"] = thinking
        kwargs["extra_body"] = extra_body
        return kwargs


class ZaiChat(_ZaiShared, Chat):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reasoning", False)
        super().__init__(*args, **kwargs)
        self.Options = ZaiOptionsMixin.Options

    def execute(
        self, prompt, stream, response, conversation=None, key=None
    ) -> Iterator[Union[str, Any]]:
        from llm.parts import StreamEvent

        if getattr(prompt.options, "coding", None) or self._is_coding():
            self.api_base = CODING_API_BASE

        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key)
        usage = None
        emitted_reasoning = False

        if stream:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            tool_calls = {}
            for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta

                    # Interleaved thinking: emit reasoning_content deltas.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        emitted_reasoning = True
                        yield StreamEvent(type="reasoning", chunk=reasoning)

                    for tool_call in delta.tool_calls or []:
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ""
                        idx = tool_call.index
                        if idx not in tool_calls:
                            tool_calls[idx] = tool_call
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=tool_call.function.name or "",
                                tool_call_id=tool_call.id,
                            )
                        else:
                            tool_calls[
                                idx
                            ].function.arguments += tool_call.function.arguments
                        if tool_call.function.arguments:
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=tool_call.function.arguments,
                                tool_call_id=tool_calls[idx].id,
                            )

                    content = delta.content
                    if content:
                        yield StreamEvent(type="text", chunk=content)

            # Store response_json with reasoning_content for preserved thinking
            response.response_json = remove_dict_none_values(
                _combine_chunks_with_reasoning(chunks)
            )
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.id,
                            name=value.function.name,
                            arguments=json.loads(value.function.arguments),
                        )
                    )
        else:
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump()
            response.response_json = remove_dict_none_values(completion.model_dump())
            message = completion.choices[0].message

            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                emitted_reasoning = True
                yield StreamEvent(type="reasoning", chunk=reasoning)

            for tool_call in message.tool_calls or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    )
                )
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=tool_call.function.name or "",
                    tool_call_id=tool_call.id,
                )
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=tool_call.function.arguments or "",
                    tool_call_id=tool_call.id,
                )

            if message.content is not None:
                yield StreamEvent(type="text", chunk=message.content)

        self.set_usage(response, usage)
        if (
            usage
            and (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            and not emitted_reasoning
        ):
            yield StreamEvent(type="reasoning", chunk="", redacted=True)
        response._prompt_json = {"messages": messages}


class ZaiAsyncChat(_ZaiShared, AsyncChat):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reasoning", False)
        super().__init__(*args, **kwargs)
        self.Options = ZaiOptionsMixin.Options

    async def execute(
        self, prompt, stream, response, conversation=None, key=None
    ) -> AsyncGenerator[Union[str, Any], None]:
        from llm.parts import StreamEvent

        if getattr(prompt.options, "coding", None) or self._is_coding():
            self.api_base = CODING_API_BASE

        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key, async_=True)
        usage = None
        emitted_reasoning = False

        if stream:
            completion = await client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=True,
                **kwargs,
            )
            chunks = []
            tool_calls = {}
            async for chunk in completion:
                chunks.append(chunk)
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta

                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        emitted_reasoning = True
                        yield StreamEvent(type="reasoning", chunk=reasoning)

                    for tool_call in delta.tool_calls or []:
                        if tool_call.function.arguments is None:
                            tool_call.function.arguments = ""
                        idx = tool_call.index
                        if idx not in tool_calls:
                            tool_calls[idx] = tool_call
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=tool_call.function.name or "",
                                tool_call_id=tool_call.id,
                            )
                        else:
                            tool_calls[
                                idx
                            ].function.arguments += tool_call.function.arguments
                        if tool_call.function.arguments:
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=tool_call.function.arguments,
                                tool_call_id=tool_calls[idx].id,
                            )

                    content = delta.content
                    if content:
                        yield StreamEvent(type="text", chunk=content)

            response.response_json = remove_dict_none_values(
                _combine_chunks_with_reasoning(chunks)
            )
            if tool_calls:
                for value in tool_calls.values():
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=value.id,
                            name=value.function.name,
                            arguments=json.loads(value.function.arguments),
                        )
                    )
        else:
            completion = await client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=False,
                **kwargs,
            )
            usage = completion.usage.model_dump()
            response.response_json = remove_dict_none_values(completion.model_dump())
            message = completion.choices[0].message

            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                emitted_reasoning = True
                yield StreamEvent(type="reasoning", chunk=reasoning)

            for tool_call in message.tool_calls or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    )
                )
                yield StreamEvent(
                    type="tool_call_name",
                    chunk=tool_call.function.name or "",
                    tool_call_id=tool_call.id,
                )
                yield StreamEvent(
                    type="tool_call_args",
                    chunk=tool_call.function.arguments or "",
                    tool_call_id=tool_call.id,
                )

            if message.content is not None:
                yield StreamEvent(type="text", chunk=message.content)

        self.set_usage(response, usage)
        if (
            usage
            and (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            and not emitted_reasoning
        ):
            yield StreamEvent(type="reasoning", chunk="", redacted=True)
        response._prompt_json = {"messages": messages}


def _model_caps(model_id):
    vision = "v" in model_id.lower() or "ocr" in model_id.lower()
    supports_schema = not vision
    supports_tools = not vision and "-128k" not in model_id
    return vision, supports_schema, supports_tools


def _fetch_models_from_api(api_base, key):
    import urllib.request

    req = urllib.request.Request(
        f"{api_base}/models",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept-Language": "en-US,en",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("data", [])


def _refresh_models():
    import click

    key = llm.get_key("zai")
    if not key:
        key = llm.get_key_from_env("GLM_API_KEY")
    if not key:
        raise click.ClickException(
            "No Z.AI API key found. Set GLM_API_KEY or run 'llm keys set zai'."
        )

    all_models = {}
    for model_name, vision, supports_schema, supports_tools in MODELS:
        all_models[model_name] = (vision, supports_schema, supports_tools)

    for alias, real_name, vision, supports_schema, supports_tools in CODING_MODELS:
        all_models[alias] = (vision, supports_schema, supports_tools)

    for api_base in (DEFAULT_API_BASE, CODING_API_BASE):
        try:
            for item in _fetch_models_from_api(api_base, key):
                model_id = item.get("id") or ""
                if not model_id or model_id in all_models:
                    continue
                if model_id.startswith("glm-"):
                    all_models[model_id] = _model_caps(model_id)
        except Exception:
            pass

    models_path = llm.user_dir() / "zai_models.json"
    models_path.write_text(json.dumps(list(all_models.items()), indent=2))


def _merged_models(default_models=None):
    if default_models is None:
        default_models = MODELS + [
            (alias, vision, supports_schema, supports_tools)
            for alias, real_name, vision, supports_schema, supports_tools in CODING_MODELS
        ]
    models_path = llm.user_dir() / "zai_models.json"
    merged = list(default_models)
    seen = {name for name, *_ in merged}
    if models_path.exists():
        try:
            for model_name, caps in json.loads(models_path.read_text()):
                if model_name not in seen:
                    merged.append((model_name, *caps))
                    seen.add(model_name)
        except Exception:
            pass
    return merged


@llm.hookimpl
def register_models(register):
    for model_name, vision, supports_schema, supports_tools in _merged_models():
        is_coding_alias = model_name in CODING_ALIASES
        kwargs = dict(
            model_id=model_name,
            model_name=(
                model_name
                if not is_coding_alias
                else next(
                    real for alias, real, *_ in CODING_MODELS if alias == model_name
                )
            ),
            api_base=(
                CODING_API_BASE if is_coding_alias else DEFAULT_API_BASE
            ),
            vision=vision,
            supports_schema=supports_schema,
            supports_tools=supports_tools,
            headers={"Accept-Language": "en-US,en"},
        )
        if is_coding_alias:
            sync_model = ZaiChat(**kwargs)
            async_model = ZaiAsyncChat(**kwargs)
            sync_model._coding = True
            async_model._coding = True
            register(sync_model, async_model)
        else:
            register(ZaiChat(**kwargs), ZaiAsyncChat(**kwargs))


@llm.hookimpl
def register_commands(cli):
    import click

    @cli.group(name="zai")
    def zai():
        "Commands for the llm-zai plugin"

    @zai.command(name="models")
    def list_models():
        "List Z.AI GLM models registered by this plugin"
        for model_name, vision, schema, tools in _merged_models():
            flags = []
            if vision:
                flags.append("vision")
            if schema:
                flags.append("schema")
            if tools:
                flags.append("tools")
            print(f"{model_name:<35} {','.join(flags) or '-'}")
        for alias, real_name, vision, schema, tools in CODING_MODELS:
            flags = ["coding"]
            if vision:
                flags.append("vision")
            if schema:
                flags.append("schema")
            if tools:
                flags.append("tools")
            print(f"{alias:<35} {','.join(flags)}  -> {real_name}")

    @zai.command(name="refresh")
    @click.option("--key", help="Z.AI API key")
    def refresh_models(key):
        "Refresh the list of models from the Z.AI API"
        _refresh_models()
        click.echo("Models refreshed from Z.AI API.")
