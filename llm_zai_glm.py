import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pydantic import Field
from typing import Optional
import os

DEFAULT_API_BASE = "https://api.z.ai/api/paas/v4"
CODING_API_BASE = "https://api.z.ai/api/coding/paas/v4"

# Models known to be available on Z.AI's OpenAI-compatible PaaS endpoint.
# (vision, supports_schema, supports_tools)
MODELS = [
    # GLM-5 series
    ("glm-5.1", True, True, True),
    ("glm-5", True, True, True),
    ("glm-5-turbo", False, True, True),
    # GLM-4.x text
    ("glm-4.7", False, True, True),
    ("glm-4.7-flash", False, True, True),
    ("glm-4.7-flashx", False, True, True),
    ("glm-4.6", False, True, True),
    ("glm-4.6-flash", False, True, True),
    ("glm-4.6-flashx", False, True, True),
    ("glm-4.5", False, True, True),
    ("glm-4.5-air", False, True, True),
    ("glm-4.5-airx", False, True, True),
    ("glm-4.5-flash", False, True, True),
    ("glm-4.5-x", False, True, True),
    # Vision / multimodal
    ("glm-4.5v", True, False, False),
    ("glm-4.6v", True, False, False),
    ("glm-4.6v-flash", True, False, False),
    ("glm-4.6v-flashx", True, False, False),
    ("glm-ocr", True, False, False),
    ("glm-5v-turbo", True, False, False),
    # Long-context / agents
    ("glm-4-32b-0414-128k", False, True, True),
    ("autoglm-phone-multilingual", False, False, False),
]

# Coding-plan models. The API expects the real model id without the prefix.
CODING_MODELS = [
    ("glm-coding-5.1", "glm-5.1", True, True, True),
    ("glm-coding-5", "glm-5", True, True, True),
]

API_BASE_FOR_MODEL = {}
for name, _, _, _ in MODELS:
    API_BASE_FOR_MODEL[name] = DEFAULT_API_BASE
for alias, real_name, _, _, _ in CODING_MODELS:
    API_BASE_FOR_MODEL[alias] = CODING_API_BASE


class ZaiGlmOptionsMixin:
    class Options(Chat.Options):
        coding: Optional[bool] = Field(
            None,
            description="Use the GLM Coding Plan endpoint (https://api.z.ai/api/coding/paas/v4)",
        )


class ZaiGlmChat(ZaiGlmOptionsMixin, Chat):
    needs_key = "zai"
    key_env_var = "ZAI_API_KEY"

    def __str__(self):
        return "Z.AI GLM: {}".format(self.model_id)

    def execute(self, prompt, stream, response, conversation=None, key=None):
        # Honor --coding on any model (real time endpoint switch)
        if getattr(prompt.options, "coding", None) or getattr(self, "_coding", False):
            self.api_base = CODING_API_BASE
        return super().execute(prompt, stream, response, conversation, key)

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("coding", None)
        return kwargs


class ZaiGlmAsyncChat(ZaiGlmOptionsMixin, AsyncChat):
    needs_key = "zai"
    key_env_var = "ZAI_API_KEY"

    def __str__(self):
        return "Z.AI GLM: {}".format(self.model_id)

    async def execute(self, prompt, stream, response, conversation=None, key=None):
        if getattr(prompt.options, "coding", None) or getattr(self, "_coding", False):
            self.api_base = CODING_API_BASE
        async for chunk in super().execute(prompt, stream, response, conversation, key):
            yield chunk

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        kwargs.pop("coding", None)
        return kwargs


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "zai", "ZAI_API_KEY") or os.environ.get("ZAI_API_KEY")
    if not key:
        return
    for model_name, vision, supports_schema, supports_tools in MODELS:
        kwargs = dict(
            model_id=model_name,
            model_name=model_name,
            api_base=DEFAULT_API_BASE,
            vision=vision,
            supports_schema=supports_schema,
            supports_tools=supports_tools,
            headers={"Accept-Language": "en-US,en"},
        )
        register(ZaiGlmChat(**kwargs), ZaiGlmAsyncChat(**kwargs))

    for alias, real_name, vision, supports_schema, supports_tools in CODING_MODELS:
        sync_model = ZaiGlmChat(
            model_id=alias,
            model_name=real_name,
            api_base=CODING_API_BASE,
            vision=vision,
            supports_schema=supports_schema,
            supports_tools=supports_tools,
            headers={"Accept-Language": "en-US,en"},
        )
        async_model = ZaiGlmAsyncChat(
            model_id=alias,
            model_name=real_name,
            api_base=CODING_API_BASE,
            vision=vision,
            supports_schema=supports_schema,
            supports_tools=supports_tools,
            headers={"Accept-Language": "en-US,en"},
        )
        sync_model._coding = True
        async_model._coding = True
        register(sync_model, async_model)


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="zai-glm")
    def zai_glm():
        "Commands for the llm-zai-glm plugin"

    @zai_glm.command(name="models")
    def list_models():
        "List Z.AI GLM models registered by this plugin"
        for model_name, vision, schema, tools in MODELS:
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
