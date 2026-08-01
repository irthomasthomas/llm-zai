# Retained Reasoning (Preserved Thinking) — Investigation & Verification

## Overview

This document summarizes the investigation, root-cause analysis, and
verification of **retained reasoning** (also called *preserved thinking*)
for the `llm-zai` plugin. Retained reasoning means that when a GLM
model produces a `reasoning_content` field during inference, that content
is stored, injected back into subsequent conversation turns, and
accessible to the model.

---

## Timeline

| Date       | Event |
|------------|-------|
| 2026-06-23 | Commit `a7273ff` — Initial reasoning_content + coding plan endpoints |
| 2026-06-25 | Agent session `01kvzbwv2z5jf67bvn28astp1a` — Root cause identified, `_batch_get_reasoning()` written, fix left uncommitted |
| 2026-06-27 | Verification session — Fix reconstructed, tested, confirmed working |

---

## The Problem

When `llm` loads conversation history from SQLite via `from_row`,
`ReasoningPart` objects are lost. The `_build_parts()` fallback in
`llm/models.py` only creates `TextPart` from `self._chunks` (the plain
text response column). Additionally, `response_json` stored in the DB
is condensed by `condense_json`, which replaces repeated substrings
with objects like `{"$": "r:<response_id>"}`.

This means that on turn 2+ of a conversation, the assistant message
sent to the Z.AI API was missing its `reasoning_content` field entirely.
The model had no access to its prior thinking.

### The condense_json red herring

The `prompt_json` column in the DB is also condensed — `content` appears
as a dict like `{"$r": [{"$": "r:<id>"}, "..."]}` instead of a plain
string. This is **cosmetic only**: the condensation happens in
`log_to_db()` *after* the API call. The actual messages sent to the API
contained plain-string `content` from `TextPart.text` (which comes from
the `response` DB column, not the condensed `response_json`).

---

## The Fix

Two new functions were added to `llm_zai.py` (+101/-21 lines vs
commit `a7273ff`):

### 1. `_reasoning_content_from_response_json(rj, conversation=None)`

Extracts `reasoning_content` from a response JSON dict, handling:
- **Streaming format**: flat `reasoning_content` key
- **Non-streaming format**: nested in `choices[].message.reasoning_content`
- **Condensed JSON**: attempts to expand condensed references using
  `condense_json.uncondense_json` with conversation response texts

### 2. `_batch_get_reasoning(conversation)`

The authoritative fix. Returns a `{response_id: reasoning_text}` dict:

1. **Fast path**: tries `response_json` via `_reasoning_content_from_response_json()`
2. **Fallback**: batch-queries the SQLite `responses.reasoning` column directly:
   ```sql
   SELECT id, reasoning FROM responses WHERE id IN (?, ?, ...)
   ```
   The `reasoning` column is populated by `llm`'s logging and is the
   only reliable source when `ReasoningPart` objects are lost during
   DB round-trip.

### 3. Modified `build_messages()`

- Calls `_batch_get_reasoning(conversation)` to build the reasoning map
- Injects `reasoning_content` on assistant messages that lost their
  `ReasoningPart` during DB serialization
- Guards with `entry.get("content")` to only inject on text-bearing
  assistant messages (not tool-call-only entries)

### 4. Modified `_merged_models()`

Added coding model aliases to the default model list so they appear
in `llm models list`.

---

## Verification

### Test 1: Mathematical calculation (DEFINITIVE PASS)

**Turn 1**: "What is 127 * 8? Show only the final answer."
- Visible response: `1016` (just the number)
- Stored reasoning: 371 chars containing step-by-step calculation
  (`7*8=56`, `carry over 5`, `2*8=16+5=21`, `1*8=8+2=10`)

**Turn 2**: "What intermediate calculation step did you take?"
- Model recalled: *"Hundreds column: I multiplied 1 by 8, which equals 8.
  I then added the carried-over 2, making it 10"*
- Also recalled: *"distributive property: 100 by 8 [800], 20 by 8 [160],
  7 by 8 [56]"*

**Conclusion**: These calculation details existed ONLY in
`reasoning_content`. The visible response was just `1016`. The model
could not have known this without accessing its injected reasoning.

**prompt_json verification**: `msg[1].reasoning_content` = 371 chars,
confirmed to contain `7 * 8 = 56` and `carry`.

### Test 2: Haiku drafting (PASS)

**Turn 1**: "Write a haiku about the ocean. Think carefully."
- Reasoning: 6031 chars with multiple draft attempts

**Turn 2**: "What was the very first haiku draft you considered?"
- Model correctly quoted: *"Vast and deep blue sea / Hiding secrets
  in the dark / Whispers to the shore"*

### Test 3: France capital (PARTIAL — false negative)

The model quoted visible content instead of reasoning. This was a test
design issue: the visible response contained similar-looking step-by-step
text, so the model's answer appeared wrong. The plumbing was confirmed
correct (`reasoning_content` was present in `prompt_json`).

---

## Architecture

```
Turn 1:
  GLM API → streaming deltas with reasoning_content
    ↓
  ZaiChat.execute() captures reasoning via StreamEvent(type="reasoning")
    ↓
  llm stores reasoning in:
    - response.reasoning column (SQLite)  ← authoritative
    - response_json.reasoning_content (condensed)
    ↓
Turn 2:
  from_row loads conversation (no ReasoningPart in _build_parts)
    ↓
  build_messages() calls _batch_get_reasoning(conversation)
    ↓
  _batch_get_reasoning:
    1. Tries response_json → may fail (condensed)
    2. Falls back to: SELECT reasoning FROM responses WHERE id IN (...)
    ↓
  Injects reasoning_content on assistant messages
    ↓
  Sends to API: {"role":"assistant", "content":"...", "reasoning_content":"..."}
    ↓
  Model accesses its prior thinking ✅
```

---

## Files Modified

| File | Change |
|------|--------|
| `llm_zai.py` | +101/-21 lines (uncommitted) |

### Key functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_reasoning_content_from_response_json()` | ~50 | Extract reasoning from response JSON, handle condensed format |
| `_batch_get_reasoning()` | ~35 | Batch DB query for reasoning column |
| `build_messages()` | modified | Use `_batch_get_reasoning()` and inject results |
| `_merged_models()` | modified | Include coding aliases in default model list |

---

## Dependencies

This fix depends on:
- `llm` core's `reasoning` column in the SQLite `responses` table
  (populated by `StreamEvent(type="reasoning")` capture)
- Local changes to `llm/default_plugins/openai_models.py` that capture
  `reasoning_content` from `delta.reasoning_content` in streaming and
  non-streaming paths (uncommitted in the `llm` fork)
- The `condense_json` library for response_json expansion (optional,
  falls back gracefully)

---

## Unresolved Items

1. **The fix is uncommitted** — the working tree has +101/-21 lines
   not yet committed to git
2. **`llm` core changes also uncommitted** — the `openai_models.py`
   changes that capture `delta.reasoning_content` are in the local fork
   but not pushed
3. **Condensed content in prompt_json** — cosmetic only (doesn't affect
   API calls) but makes debugging confusing. A future improvement could
   uncondense `prompt_json` when reading from DB for display purposes
4. **Backup files cleaned up** — all `.bak` files removed
