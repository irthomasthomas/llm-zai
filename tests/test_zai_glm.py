"""Tests for llm-zai-glm plugin.

Inspired by simonw/llm-anthropic test structure, but focused on the
unique features of the Z.AI GLM plugin:

- Preserved thinking (reasoning_content injection from DB round-trip)
- Interleaved thinking (reasoning_content alongside tool_calls)
- Message key ordering (role -> content -> reasoning_content -> tool_calls)
- clear_thinking defaults (coding=False, standard=True) and placement
  inside the thinking object in extra_body
- thinking_type and effort options (effort only for EFFORT_MODELS)
- Coding endpoint routing (api_base switching)
- _combine_chunks_with_reasoning helper
- Model registration and capabilities
"""
import json
import os
import datetime
import sqlite_utils
import pytest
from unittest.mock import MagicMock, patch

import llm
import llm_zai_glm as zai
from llm_zai_glm import (
    ZaiGlmChat,
    ZaiGlmAsyncChat,
    ZaiGlmOptionsMixin,
    _combine_chunks_with_reasoning,
    _model_caps,
    MODELS,
    CODING_MODELS,
    CODING_ALIASES,
    EFFORT_MODELS,
    DEFAULT_API_BASE,
    CODING_API_BASE,
)
from llm.parts import (
    StreamEvent,
    TextPart,
    ReasoningPart,
    ToolCallPart,
    ToolResultPart,
    AttachmentPart,
)
from llm import user, assistant, tool_message, system


ZAI_API_KEY = os.environ.get("PYTEST_ZAI_API_KEY", None) or "test-key-..."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(model_id="glm-5.1", key="test-key"):
    """Get a registered ZAI GLM model instance."""
    model = llm.get_model(model_id)
    model.key = key
    return model


def make_prompt(model, prompt_text="hi", **kwargs):
    """Build a Prompt without hitting the API."""
    options = kwargs.pop("options", model.Options())
    return llm.Prompt(prompt_text, model=model, options=options, **kwargs)


def build_messages_for(model_id, messages=None, **prompt_kwargs):
    """Invoke build_messages on a one-shot Prompt without hitting the API.

    llm >= 0.32 models translate prompt.messages, so the helper must pass an
    explicit messages= chain (not just prompt= text) to Prompt.
    """
    model = make_model(model_id)
    options = prompt_kwargs.pop("options", model.Options())
    p = llm.Prompt(
        None,
        model=model,
        options=options,
        messages=messages if messages is not None else [],
        **prompt_kwargs,
    )
    return model.build_messages(p, None)


# ---------------------------------------------------------------------------
# Model registration and capabilities
# ---------------------------------------------------------------------------

class TestModelRegistration:
    """Verify models are registered correctly with proper capabilities."""

    @pytest.mark.parametrize(
        "model_id,vision,schema,tools",
        MODELS,
        ids=[m[0] for m in MODELS],
    )
    def test_standard_models_registered(self, model_id, vision, schema, tools):
        model = llm.get_model(model_id)
        assert model is not None
        assert model.vision == vision
        assert model.supports_schema == schema
        assert model.supports_tools == tools

    @pytest.mark.parametrize(
        "alias,real_name,vision,schema,tools",
        CODING_MODELS,
        ids=[m[0] for m in CODING_MODELS],
    )
    def test_coding_models_registered(self, alias, real_name, vision, schema, tools):
        model = llm.get_model(alias)
        assert model is not None
        assert isinstance(model, ZaiGlmChat)
        assert model._coding is True
        assert model.model_name == real_name
        assert model.api_base == CODING_API_BASE

    def test_standard_models_use_default_api_base(self):
        for model_name, *_ in MODELS:
            model = llm.get_model(model_name)
            assert model.api_base == DEFAULT_API_BASE

    def test_coding_aliases_use_coding_api_base(self):
        for alias, *_ in CODING_MODELS:
            model = llm.get_model(alias)
            assert model.api_base == CODING_API_BASE

    def test_effort_models_set(self):
        assert EFFORT_MODELS == {"glm-5.2", "glm-5.2-coding", "glm-5.1", "glm-5.1-coding"}

    def test_coding_aliases_set(self):
        assert CODING_ALIASES == {alias for alias, *_ in CODING_MODELS}

    def test_sync_and_async_registered(self):
        model = llm.get_model("glm-5.2")
        async_model = llm.get_async_model("glm-5.2")
        assert isinstance(model, ZaiGlmChat)
        assert isinstance(async_model, ZaiGlmAsyncChat)

    def test_coding_sync_and_async_registered(self):
        model = llm.get_model("glm-5.2-coding")
        async_model = llm.get_async_model("glm-5.2-coding")
        assert isinstance(model, ZaiGlmChat)
        assert isinstance(async_model, ZaiGlmAsyncChat)
        assert model._coding is True
        assert async_model._coding is True

    def test_model_str(self):
        model = llm.get_model("glm-5.2")
        assert str(model) == "Z.AI GLM: glm-5.2"

    def test_model_needs_key(self):
        model = llm.get_model("glm-5.2")
        assert model.needs_key == "zai"
        assert model.key_env_var == "GLM_API_KEY"

    def test_model_headers_include_accept_language(self):
        model = llm.get_model("glm-5.2")
        assert model.headers.get("Accept-Language") == "en-US,en"


class TestModelCaps:
    """Test the _model_caps heuristic for auto-discovered models."""

    def test_vision_model(self):
        assert _model_caps("glm-5.1v") == (True, False, False)

    def test_vision_flash(self):
        assert _model_caps("glm-4.6v-flash") == (True, False, False)

    def test_ocr_model(self):
        assert _model_caps("glm-ocr") == (True, False, False)

    def test_text_model(self):
        assert _model_caps("glm-5.1") == (False, True, True)

    def test_128k_model_no_tools(self):
        assert _model_caps("glm-4-32b-0414-128k") == (False, True, False)


# ---------------------------------------------------------------------------
# build_messages -- basic structure
# ---------------------------------------------------------------------------

class TestBuildMessagesBasic:
    """Verify build_messages produces correct OpenAI-compatible message structure."""

    def test_simple_user_text(self):
        msgs = build_messages_for("glm-5.1", messages=[user("hello")])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_multi_turn(self):
        msgs = build_messages_for(
            "glm-5.2",
            messages=[user("hi"), assistant(TextPart("hello!"))],
        )
        assert msgs == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello!"},
        ]

    def test_system_message_included_in_messages(self):
        """System messages stay in the messages list (OpenAI-compatible format)."""
        msgs = build_messages_for(
            "glm-5.2",
            messages=[system("be helpful"), user("hi")],
        )
        assert msgs == [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ]

    def test_empty_user_text_kept(self):
        """User messages with empty string content are kept (not None)."""
        msgs = build_messages_for(
            "glm-5.2",
            messages=[user("")],
        )
        assert msgs == [{"role": "user", "content": ""}]

    def test_assistant_none_content_preserved(self):
        """Assistant with only tool_calls gets content=None (not skipped)."""
        msgs = build_messages_for(
            "glm-5.2",
            messages=[
                user("what time?"),
                assistant(
                    ToolCallPart(name="clock", arguments={}, tool_call_id="c1"),
                ),
            ],
        )
        assert msgs[-1]["role"] == "assistant"
        assert msgs[-1]["content"] is None
        assert "tool_calls" in msgs[-1]


# ---------------------------------------------------------------------------
# build_messages -- tool calls and tool results
# ---------------------------------------------------------------------------

class TestBuildMessagesTools:
    """Verify tool call / tool result message construction."""

    def test_assistant_with_text_and_tool_call(self):
        msgs = build_messages_for(
            "glm-5.2",
            messages=[
                user("what time?"),
                assistant(
                    TextPart(text="Let me check"),
                    ToolCallPart(name="clock", arguments={}, tool_call_id="c1"),
                ),
            ],
        )
        assert msgs == [
            {"role": "user", "content": "what time?"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "c1",
                        "function": {"name": "clock", "arguments": "{}"},
                    }
                ],
            },
        ]

    def test_tool_result_emitted_as_separate_message(self):
        msgs = build_messages_for(
            "glm-5.2",
            messages=[
                user("what time?"),
                assistant(
                    ToolCallPart(name="clock", arguments={}, tool_call_id="c1"),
                ),
                tool_message(
                    ToolResultPart(name="clock", output="noon", tool_call_id="c1"),
                ),
            ],
        )
        assert msgs[-1] == {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "noon",
        }

    def test_tool_result_ordering(self):
        """Tool result should appear after the assistant tool_use."""
        msgs = build_messages_for(
            "glm-5.2",
            messages=[
                user("q"),
                assistant(
                    ToolCallPart(name="fn", arguments={"x": 1}, tool_call_id="c1"),
                ),
                tool_message(
                    ToolResultPart(name="fn", output="42", tool_call_id="c1"),
                ),
            ],
        )
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "tool"]

    def test_tool_call_arguments_json_serialized(self):
        msgs = build_messages_for(
            "glm-5.2",
            messages=[
                user("q"),
                assistant(
                    ToolCallPart(
                        name="search", arguments={"query": "cats"}, tool_call_id="c1"
                    ),
                ),
            ],
        )
        tool_call = msgs[1]["tool_calls"][0]
        assert json.loads(tool_call["function"]["arguments"]) == {"query": "cats"}


# ---------------------------------------------------------------------------
# build_messages -- preserved thinking (DB round-trip injection)
# ---------------------------------------------------------------------------

class TestPreservedThinking:
    """Verify reasoning_content is preserved across DB round-trips."""

    def test_reasoning_injected_from_conversation_response_json_streaming(self):
        """reasoning_content stored flat in response_json (streaming path)."""
        model = make_model("glm-5.1")

        conversation = MagicMock()
        response_mock = MagicMock()
        response_mock.id = "resp-1"
        response_mock.response_json = {
            "content": "answer",
            "reasoning_content": "I should think about this...",
        }
        conversation.responses = [response_mock]

        prompt = llm.Prompt(
            "follow up",
            model=model,
            options=model.Options(),
            messages=[
                assistant(TextPart(text="answer")),
                user("follow up"),
            ],
        )

        msgs = model.build_messages(prompt, conversation)
        assistant_msg = msgs[0]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["reasoning_content"] == "I should think about this..."

    def test_reasoning_injected_from_conversation_response_json_nonstreaming(self):
        """reasoning_content nested in choices[0].message (non-streaming path)."""
        model = make_model("glm-5.1")

        conversation = MagicMock()
        response_mock = MagicMock()
        response_mock.id = "resp-1"
        response_mock.response_json = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "Deep thoughts",
                    }
                }
            ]
        }
        conversation.responses = [response_mock]

        prompt = llm.Prompt(
            "follow up",
            model=model,
            options=model.Options(),
            messages=[
                assistant(TextPart(text="answer")),
                user("follow up"),
            ],
        )

        msgs = model.build_messages(prompt, conversation)
        assert msgs[0]["reasoning_content"] == "Deep thoughts"

    def test_reasoning_not_overwritten_if_reasoning_part_present(self):
        """If ReasoningPart already exists, it takes priority over response_json."""
        model = make_model("glm-5.1")

        conversation = MagicMock()
        response_mock = MagicMock()
        response_mock.id = "resp-1"
        response_mock.response_json = {"reasoning_content": "from_db"}
        conversation.responses = [response_mock]

        prompt = llm.Prompt(
            "q",
            model=model,
            options=model.Options(),
            messages=[
                assistant(
                    ReasoningPart(text="from_part"),
                    TextPart(text="answer"),
                ),
                user("q"),
            ],
        )

        msgs = model.build_messages(prompt, conversation)
        assert msgs[0]["reasoning_content"] == "from_part"

    def test_no_crash_when_no_conversation(self):
        model = make_model("glm-5.1")
        prompt = llm.Prompt("hi", model=model, options=model.Options())
        msgs = model.build_messages(prompt, None)
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_no_crash_when_no_responses(self):
        model = make_model("glm-5.1")

        conversation = MagicMock()
        conversation.responses = []

        prompt = llm.Prompt("hi", model=model, options=model.Options())
        msgs = model.build_messages(prompt, conversation)
        assert msgs == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# build_messages -- interleaved thinking with tool calls
# ---------------------------------------------------------------------------

class TestInterleavedThinking:
    """Verify reasoning_content appears alongside tool_calls in assistant messages."""

    def test_reasoning_before_tool_call(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[
                user("q"),
                assistant(
                    ReasoningPart(text="I need to search for this."),
                    ToolCallPart(name="search", arguments={"q": "test"}, tool_call_id="c1"),
                ),
            ],
        )
        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["reasoning_content"] == "I need to search for this."
        assert "tool_calls" in assistant_msg

    def test_reasoning_with_text_and_tool_call(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[
                user("q"),
                assistant(
                    ReasoningPart(text="Thinking step"),
                    TextPart(text="Let me look that up."),
                    ToolCallPart(name="lookup", arguments={}, tool_call_id="c1"),
                ),
            ],
        )
        assistant_msg = msgs[1]
        assert assistant_msg["content"] == "Let me look that up."
        assert assistant_msg["reasoning_content"] == "Thinking step"
        assert len(assistant_msg["tool_calls"]) == 1

    def test_multiple_reasoning_parts_concatenated(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[
                user("q"),
                assistant(
                    ReasoningPart(text="Part 1. "),
                    ReasoningPart(text="Part 2."),
                    TextPart(text="Answer"),
                ),
            ],
        )
        assert msgs[1]["reasoning_content"] == "Part 1. Part 2."

    def test_reasoning_only_on_assistant(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[user("q")],
        )
        assert "reasoning_content" not in msgs[0]


# ---------------------------------------------------------------------------
# build_messages -- message key ordering
# ---------------------------------------------------------------------------

class TestMessageKeyOrdering:
    """Verify message keys follow Z.AI's expected order:
    role -> content -> reasoning_content -> tool_calls."""

    def test_key_order_with_reasoning_and_tool_calls(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[
                user("q"),
                assistant(
                    ReasoningPart(text="thinking"),
                    TextPart(text="response"),
                    ToolCallPart(name="fn", arguments={}, tool_call_id="c1"),
                ),
            ],
        )
        keys = list(msgs[1].keys())
        assert keys.index("role") < keys.index("content")
        assert keys.index("content") < keys.index("reasoning_content")
        assert keys.index("reasoning_content") < keys.index("tool_calls")

    def test_key_order_reasoning_no_tool_calls(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[
                user("q"),
                assistant(
                    ReasoningPart(text="thinking"),
                    TextPart(text="response"),
                ),
            ],
        )
        keys = list(msgs[1].keys())
        assert keys == ["role", "content", "reasoning_content"]

    def test_key_order_tool_calls_no_reasoning(self):
        model = make_model("glm-5.1")
        msgs = build_messages_for(
            "glm-5.1",
            messages=[
                user("q"),
                assistant(
                    TextPart(text="response"),
                    ToolCallPart(name="fn", arguments={}, tool_call_id="c1"),
                ),
            ],
        )
        keys = list(msgs[1].keys())
        assert keys == ["role", "content", "tool_calls"]


# ---------------------------------------------------------------------------
# build_kwargs -- thinking configuration
# ---------------------------------------------------------------------------

class TestBuildKwargs:
    """Verify build_kwargs configures thinking, clear_thinking, and effort correctly."""

    def test_default_thinking_enabled(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "hi")
        kwargs = model.build_kwargs(prompt, stream=True)
        thinking = kwargs["extra_body"]["thinking"]
        assert thinking["type"] == "enabled"
        assert thinking["clear_thinking"] is True

    def test_default_clear_thinking_false_on_coding(self):
        model = make_model("glm-5.1-coding")
        prompt = make_prompt(model, "hi")
        kwargs = model.build_kwargs(prompt, stream=True)
        thinking = kwargs["extra_body"]["thinking"]
        assert thinking["clear_thinking"] is False

    def test_clear_thinking_inside_thinking_object(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "hi")
        kwargs = model.build_kwargs(prompt, stream=True)
        thinking = kwargs["extra_body"]["thinking"]
        assert "clear_thinking" in thinking
        assert "clear_thinking" not in kwargs["extra_body"]

    def test_explicit_clear_thinking_true(self):
        model = make_model("glm-5.1-coding")
        prompt = make_prompt(model, "hi", options=model.Options(clear_thinking=True))
        kwargs = model.build_kwargs(prompt, stream=True)
        assert kwargs["extra_body"]["thinking"]["clear_thinking"] is True

    def test_explicit_clear_thinking_false(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "hi", options=model.Options(clear_thinking=False))
        kwargs = model.build_kwargs(prompt, stream=True)
        assert kwargs["extra_body"]["thinking"]["clear_thinking"] is False

    def test_thinking_type_disabled(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(
            model, "hi", options=model.Options(thinking_type="disabled")
        )
        kwargs = model.build_kwargs(prompt, stream=True)
        assert kwargs["extra_body"]["thinking"]["type"] == "disabled"

    def test_effort_only_for_supported_models(self):
        model = make_model("glm-5.2")
        prompt = make_prompt(model, "hi")
        kwargs = model.build_kwargs(prompt, stream=True)
        assert kwargs["extra_body"]["thinking"]["effort"] == "max"

    def test_effort_omitted_for_unsupported_models(self):
        # glm-5.1 is now in EFFORT_MODELS; glm-5 is not.
        model = make_model("glm-5")
        prompt = make_prompt(model, "hi")
        kwargs = model.build_kwargs(prompt, stream=True)
        assert "effort" not in kwargs["extra_body"]["thinking"]

    def test_coding_model_effort_included(self):
        model = make_model("glm-5.2-coding")
        prompt = make_prompt(model, "hi")
        kwargs = model.build_kwargs(prompt, stream=True)
        assert kwargs["extra_body"]["thinking"]["effort"] == "max"

    def test_coding_option_removed_from_kwargs(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "hi", options=model.Options(coding=True))
        kwargs = model.build_kwargs(prompt, stream=True)
        assert "coding" not in kwargs

    def test_consumed_options_removed_from_kwargs(self):
        model = make_model("glm-5.2")
        prompt = make_prompt(
            model,
            "hi",
            options=model.Options(
                thinking_type="enabled", effort="high", clear_thinking=False
            ),
        )
        kwargs = model.build_kwargs(prompt, stream=True)
        assert "thinking_type" not in kwargs
        assert "effort" not in kwargs
        assert "clear_thinking" not in kwargs
        t = kwargs["extra_body"]["thinking"]
        assert t["type"] == "enabled"
        assert t["effort"] == "high"
        assert t["clear_thinking"] is False


# ---------------------------------------------------------------------------
# build_kwargs -- coding endpoint routing
# ---------------------------------------------------------------------------

class TestCodingEndpointRouting:
    """Verify coding option/alias switches api_base to CODING_API_BASE."""

    def test_coding_alias_uses_coding_api_base(self):
        model = make_model("glm-5.1-coding")
        assert model._is_coding() is True
        assert model.api_base == CODING_API_BASE

    def test_standard_model_default_api_base(self):
        model = make_model("glm-5.1")
        assert model._is_coding() is False
        assert model.api_base == DEFAULT_API_BASE

    def test_coding_alias_maps_to_real_model_name(self):
        for alias, real_name, *_ in CODING_MODELS:
            model = make_model(alias)
            assert model.model_name == real_name


# ---------------------------------------------------------------------------
# _combine_chunks_with_reasoning
# ---------------------------------------------------------------------------

class TestCombineChunksWithReasoning:
    """Verify the streaming chunk combiner captures reasoning_content."""

    def _make_chunk(self, content=None, reasoning_content=None):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning_content
        delta.tool_calls = None

        choice = MagicMock()
        choice.delta = delta
        choice.index = 0

        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = None
        return chunk

    def test_text_only(self):
        chunks = [
            self._make_chunk(content="Hello"),
            self._make_chunk(content=" world"),
        ]
        result = _combine_chunks_with_reasoning(chunks)
        assert "content" in result
        assert "reasoning_content" not in result

    def test_reasoning_captured(self):
        chunks = [
            self._make_chunk(reasoning_content="Thinking..."),
            self._make_chunk(reasoning_content=" More thinking."),
            self._make_chunk(content="Answer"),
        ]
        result = _combine_chunks_with_reasoning(chunks)
        assert result.get("reasoning_content") == "Thinking... More thinking."

    def test_reasoning_without_text(self):
        chunks = [self._make_chunk(reasoning_content="Only thinking")]
        result = _combine_chunks_with_reasoning(chunks)
        assert result.get("reasoning_content") == "Only thinking"


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    """Verify ZaiGlmOptionsMixin.Options exposes the right fields."""

    def test_options_has_coding_field(self):
        model = make_model("glm-5.1")
        opts = model.Options()
        assert hasattr(opts, "coding")
        assert opts.coding is None

    def test_options_has_thinking_type_field(self):
        model = make_model("glm-5.1")
        opts = model.Options()
        assert opts.thinking_type == "enabled"

    def test_options_has_effort_field(self):
        model = make_model("glm-5.1")
        opts = model.Options()
        assert opts.effort == "max"

    def test_options_has_clear_thinking_field(self):
        model = make_model("glm-5.1")
        opts = model.Options()
        assert opts.clear_thinking is None

    def test_options_set_all_fields(self):
        model = make_model("glm-5.2")
        opts = model.Options(
            coding=True,
            thinking_type="disabled",
            effort="high",
            clear_thinking=False,
        )
        assert opts.coding is True
        assert opts.thinking_type == "disabled"
        assert opts.effort == "high"
        assert opts.clear_thinking is False


# ---------------------------------------------------------------------------
# DB round-trip: conversation loaded from SQLite preserves reasoning
# ---------------------------------------------------------------------------

class TestConversationRoundTrip:
    """Regression: reasoning_content survives a full DB round-trip."""

    def test_reasoning_preserved_after_db_load(self, tmp_path):
        model = make_model("glm-5.1")

        conversation = model.conversation()
        first = llm.Response(
            llm.Prompt("q1", model=model, options=model.Options()),
            model,
            stream=False,
            conversation=conversation,
        )
        first._chunks = ["answer"]
        first._stream_events = [StreamEvent(type="text", chunk="answer")]
        first.response_json = {
            "content": "answer",
            "reasoning_content": "I deliberated on this.",
        }
        first._done = True
        first._start = 0.0
        first._end = 0.0
        first._start_utcnow = datetime.datetime.now(datetime.timezone.utc)

        db = sqlite_utils.Database(str(tmp_path / "logs.db"))
        from llm.migrations import migrate
        migrate(db)
        first.log_to_db(db)
        conversation.responses.append(first)

        # Simulate what conversation.prompt() does: messages include prior turns
        prompt2 = llm.Prompt(
            "q2",
            model=model,
            options=model.Options(),
            messages=[
                assistant(TextPart(text="answer")),
                user("q2"),
            ],
        )
        msgs = model.build_messages(prompt2, conversation)

        assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        assert any(
            m.get("reasoning_content") == "I deliberated on this."
            for m in assistant_msgs
        )

    def test_tool_chain_round_trip(self, tmp_path):
        model = make_model("glm-5.1")

        conversation = model.conversation()
        first = llm.Response(
            llm.Prompt("q1", model=model, tools=[], options=model.Options()),
            model,
            stream=False,
            conversation=conversation,
        )
        first.add_tool_call(
            llm.ToolCall(name="clock", arguments={}, tool_call_id="c1")
        )
        first._chunks = []
        first._stream_events = [
            StreamEvent(type="tool_call_name", chunk="clock", tool_call_id="c1"),
            StreamEvent(type="tool_call_args", chunk="{}", tool_call_id="c1"),
        ]
        first.response_json = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "clock", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        first._done = True
        first._start = 0.0
        first._end = 0.0
        first._start_utcnow = datetime.datetime.now(datetime.timezone.utc)

        db = sqlite_utils.Database(str(tmp_path / "logs.db"))
        from llm.migrations import migrate
        migrate(db)
        first.log_to_db(db)
        conversation.responses.append(first)

        # Simulate conversation with prior tool call messages
        prompt2 = llm.Prompt(
            "q2",
            model=model,
            options=model.Options(),
            messages=[
                user("q1"),
                assistant(
                    ToolCallPart(name="clock", arguments={}, tool_call_id="c1"),
                ),
                tool_message(
                    ToolResultPart(name="clock", output="noon", tool_call_id="c1"),
                ),
                user("q2"),
            ],
        )
        msgs = model.build_messages(prompt2, conversation)

        assistant_with_tools = [
            m for m in msgs if m.get("role") == "assistant" and "tool_calls" in m
        ]
        assert len(assistant_with_tools) >= 1


# ---------------------------------------------------------------------------
# StreamEvent emission (mocked execute)
# ---------------------------------------------------------------------------

class TestStreamEventEmission:
    """Verify execute() yields correct StreamEvent types using mocked API calls."""

    def _mock_chunk(self, content=None, reasoning=None, tool_calls=None,
                    reasoning_tokens=0):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning
        delta.tool_calls = tool_calls

        choice = MagicMock()
        choice.delta = delta
        choice.index = 0

        chunk = MagicMock()
        chunk.choices = [choice]
        chunk.usage = MagicMock()
        chunk.usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        }
        return chunk

    def test_streaming_yields_reasoning_then_text(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "q")

        mock_chunks = [
            self._mock_chunk(reasoning="Let me think..."),
            self._mock_chunk(content="Answer"),
        ]

        with patch.object(model, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter(mock_chunks)
            mock_get_client.return_value = mock_client

            response = llm.Response(
                prompt, model, stream=True, conversation=model.conversation()
            )
            events = list(model.execute(prompt, True, response))

        reasoning_events = [e for e in events if e.type == "reasoning"]
        text_events = [e for e in events if e.type == "text"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].chunk == "Let me think..."
        assert len(text_events) == 1
        assert text_events[0].chunk == "Answer"
        assert events.index(reasoning_events[0]) < events.index(text_events[0])

    def test_streaming_yields_redacted_reasoning_when_no_content(self):
        """When reasoning_tokens > 0 but no reasoning_content, emit redacted."""
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "q")

        mock_chunk = self._mock_chunk(
            content="answer", reasoning_tokens=15
        )

        with patch.object(model, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([mock_chunk])
            mock_get_client.return_value = mock_client

            response = llm.Response(
                prompt, model, stream=True, conversation=model.conversation()
            )
            events = list(model.execute(prompt, True, response))

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].redacted is True
        assert reasoning_events[0].chunk == ""

    def test_no_redacted_reasoning_when_content_emitted(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "q")

        mock_chunk = self._mock_chunk(
            content="answer", reasoning="I thought.", reasoning_tokens=15
        )

        with patch.object(model, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([mock_chunk])
            mock_get_client.return_value = mock_client

            response = llm.Response(
                prompt, model, stream=True, conversation=model.conversation()
            )
            events = list(model.execute(prompt, True, response))

        reasoning_events = [e for e in events if e.type == "reasoning"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].redacted is False


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------

class TestAsyncExecution:
    """Verify async execute path works and emits correct events."""

    @pytest.mark.asyncio
    async def test_async_streaming_yields_reasoning_then_text(self):
        model = llm.get_async_model("glm-5.1")
        model.key = "test-key"
        prompt = make_prompt(model, "q")

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.content = "answer"
        mock_chunk.choices[0].delta.reasoning_content = "thinking"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.model_dump.return_value = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

        async def mock_create(*args, **kwargs):
            async def _gen():
                yield mock_chunk
            return _gen()

        with patch.object(model, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_get_client.return_value = mock_client

            response = llm.Response(
                prompt, model, stream=True, conversation=model.conversation()
            )
            events = []
            async for event in model.execute(prompt, True, response):
                events.append(event)

        reasoning_events = [e for e in events if e.type == "reasoning"]
        text_events = [e for e in events if e.type == "text"]
        assert len(reasoning_events) == 1
        assert reasoning_events[0].chunk == "thinking"
        assert len(text_events) == 1
        assert text_events[0].chunk == "answer"


# ---------------------------------------------------------------------------
# Coding endpoint in execute
# ---------------------------------------------------------------------------

class TestExecuteCodingEndpoint:
    """Verify execute() routes to coding API base when coding flag is set."""

    def test_coding_option_switches_api_base_in_execute(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "q", options=model.Options(coding=True))

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.content = "answer"
        mock_chunk.choices[0].delta.reasoning_content = None
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.usage = MagicMock()
        mock_chunk.usage.model_dump.return_value = {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        }

        with patch.object(model, "get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = iter([mock_chunk])
            mock_get_client.return_value = mock_client

            response = llm.Response(
                prompt, model, stream=True, conversation=model.conversation()
            )
            list(model.execute(prompt, True, response))

        assert model.api_base == CODING_API_BASE


# ---------------------------------------------------------------------------
# Integration: full build_messages + build_kwargs flow
# ---------------------------------------------------------------------------

class TestIntegrationFlow:
    """End-to-ish-end tests combining build_messages and build_kwargs."""

    def test_coding_model_preserved_thinking_round_trip(self):
        model = make_model("glm-5.1-coding")

        conversation = MagicMock()
        resp_mock = MagicMock()
        resp_mock.id = "r1"
        resp_mock.response_json = {"reasoning_content": "preserved thought"}
        conversation.responses = [resp_mock]

        prompt = llm.Prompt(
            "follow up",
            model=model,
            options=model.Options(),
            messages=[
                assistant(TextPart(text="prev answer")),
                user("follow up"),
            ],
        )

        msgs = model.build_messages(prompt, conversation)
        kwargs = model.build_kwargs(prompt, stream=True)

        assert msgs[0].get("reasoning_content") == "preserved thought"
        assert kwargs["extra_body"]["thinking"]["clear_thinking"] is False

    def test_standard_model_clears_thinking_by_default(self):
        model = make_model("glm-5.1")
        prompt = make_prompt(model, "q")
        kwargs = model.build_kwargs(prompt, stream=True)
        assert kwargs["extra_body"]["thinking"]["clear_thinking"] is True

    def test_effort_model_with_coding_and_preserved_thinking(self):
        model = make_model("glm-5.2-coding")

        conversation = MagicMock()
        resp_mock = MagicMock()
        resp_mock.id = "r1"
        resp_mock.response_json = {"reasoning_content": "deep thoughts"}
        conversation.responses = [resp_mock]

        prompt = llm.Prompt(
            "q",
            model=model,
            options=model.Options(),
            messages=[
                assistant(TextPart(text="answer")),
                user("q"),
            ],
        )

        msgs = model.build_messages(prompt, conversation)
        kwargs = model.build_kwargs(prompt, stream=True)

        assert kwargs["extra_body"]["thinking"]["effort"] == "max"
        assert kwargs["extra_body"]["thinking"]["clear_thinking"] is False
        assert msgs[0].get("reasoning_content") == "deep thoughts"
        assert model.api_base == CODING_API_BASE
