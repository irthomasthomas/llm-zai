# llm-zai-glm

LLM CLI plugin for [Z.AI](https://z.ai/) / GLM models, including the GLM Coding Plan endpoint.

## Installation

```bash
llm install llm-zai-glm
```

Set your API key:

```bash
export ZAI_API_KEY="your-zai-api-key"
```

Or use:

```bash
llm keys set zai
```

## Usage

```bash
# Basic model invocation
llm -m glm-5.1 "hello"
llm -m glm-5.2 "welcome message"
llm -m glm-5 "generic response"

# Use the Coding Plan endpoint with any GLM model (e.g., --coding flag)
llm -m glm-5.1 --coding "implement a binary search in rust"
llm -m glm-5.2-coding "review this repo"

# Vision models
llm -m glm-5v-turbo "describe this image" -a image.png

# Long‑context / agent models
llm -m autoglm-phone-multilingual "converse about my project"
```

## Options

You can configure the chat behavior via options:

- `--thinking-type`: enabled or disabled (controls whether reasoning is kept)
- `--effort`: max or high effort for supported models (e.g., glm-5.2, glm-5.1-coding)
- `--coding`: use the GLM Coding Plan endpoint (`--coding true` or `--coding false`)
- `--clear-thinking`: preserve reasoning history across turns (true) or clear it (false)

## Models

The plugin registers the following GLM models on Z.AI's OpenAI-compatible endpoints:

### Chat models
- `glm-5.2`
- `glm-5.1`
- `glm-5`
- `glm-5-turbo`
- `glm-ocr`
- `glm-5v-turbo`

### Coding plan variants (use `--coding` flag)
- `glm-5.2-coding` → `glm-5.2`
- `glm-5.1-coding` → `glm-5.1`

### Long‑context / agent models
- `autoglm-phone-multilingual`
- `glm-4-32b-0414-128k`

### Vision models
- `glm-5v-turbo`
- `glm-ocr`

## Listing registered models

```bash
llm zai-glm models
```

## License

Apache-2.0
