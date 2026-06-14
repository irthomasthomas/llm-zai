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
llm -m glm-5.1 "hello"
llm -m glm-4.6 "write a python web server"
llm -m glm-5v-turbo "describe this image" -a image.png
llm -m glm-coding-5.1 "review this repo"
```

Use the Coding Plan endpoint explicitly with any GLM model:

```bash
llm -m glm-5.1 --coding "implement a binary search in rust"
```

## Models

The plugin registers GLM chat models on Z.AI's OpenAI-compatible endpoints:

- `glm-5.1`, `glm-5`, `glm-5-turbo`
- `glm-4.7`, `glm-4.7-flash`, `glm-4.7-flashx`
- `glm-4.6`, `glm-4.6-flash`, `glm-4.6-flashx`
- `glm-4.5`, `glm-4.5-air`, `glm-4.5-airx`, `glm-4.5-flash`, `glm-4.5-x`
- Vision: `glm-5v-turbo`, `glm-4.5v`, `glm-4.6v`, `glm-4.6v-flash`, `glm-4.6v-flashx`, `glm-ocr`
- Long-context/agent: `glm-4-32b-0414-128k`, `autoglm-phone-multilingual`
- Coding aliases: `glm-coding-5.1 -> glm-5.1`, `glm-coding-5 -> glm-5`

List models:

```bash
llm zai-glm models
```

## License

Apache-2.0
