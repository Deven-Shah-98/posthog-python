"""
Tests for posthog.ai.gemini.gemini_converter module.

Covers: _format_parts_as_content_blocks, _format_dict_message,
_format_object_message, format_gemini_response, extract_gemini_stop_reason,
extract_gemini_stop_reason_from_chunk, extract_gemini_system_instruction,
extract_gemini_tools, format_gemini_input_with_system, format_gemini_input,
extract_gemini_web_search_count, _extract_usage_from_metadata,
extract_gemini_usage_from_response, extract_gemini_usage_from_chunk,
extract_gemini_content_from_chunk, format_gemini_streaming_output,
extract_gemini_embedding_token_count.
"""

import base64
from types import SimpleNamespace

import pytest

from posthog.ai.gemini.gemini_converter import (
    _extract_usage_from_metadata,
    _format_dict_message,
    _format_object_message,
    _format_parts_as_content_blocks,
    extract_gemini_content_from_chunk,
    extract_gemini_embedding_token_count,
    extract_gemini_stop_reason,
    extract_gemini_stop_reason_from_chunk,
    extract_gemini_system_instruction,
    extract_gemini_tools,
    extract_gemini_usage_from_chunk,
    extract_gemini_usage_from_response,
    extract_gemini_web_search_count,
    format_gemini_input,
    format_gemini_input_with_system,
    format_gemini_response,
    format_gemini_streaming_output,
)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# _format_parts_as_content_blocks
# ---------------------------------------------------------------------------

class TestFormatPartsAsContentBlocks:
    def test_dict_text(self):
        result = _format_parts_as_content_blocks([{"text": "hello"}])
        assert result == [{"type": "text", "text": "hello"}]

    def test_string_part(self):
        result = _format_parts_as_content_blocks(["hello"])
        assert result == [{"type": "text", "text": "hello"}]

    def test_dict_inline_data_image(self):
        result = _format_parts_as_content_blocks([
            {"inline_data": {"mime_type": "image/png", "data": "base64data"}}
        ])
        assert result[0]["type"] == "image"
        assert result[0]["inline_data"]["mime_type"] == "image/png"

    def test_dict_inline_data_document(self):
        result = _format_parts_as_content_blocks([
            {"inline_data": {"mime_type": "application/pdf", "data": "base64data"}}
        ])
        assert result[0]["type"] == "document"

    def test_object_text(self):
        part = _ns(text="world")
        result = _format_parts_as_content_blocks([part])
        assert result == [{"type": "text", "text": "world"}]

    def test_object_empty_text(self):
        part = _ns(text="")
        result = _format_parts_as_content_blocks([part])
        assert result == []

    def test_object_inline_data_with_attrs(self):
        inline = _ns(mime_type="image/jpeg", data="imgdata")
        part = _ns(inline_data=inline)
        result = _format_parts_as_content_blocks([part])
        assert result[0]["type"] == "image"
        assert result[0]["inline_data"]["data"] == "imgdata"

    def test_object_inline_data_document(self):
        inline = _ns(mime_type="application/pdf", data="docdata")
        part = _ns(inline_data=inline)
        result = _format_parts_as_content_blocks([part])
        assert result[0]["type"] == "document"

    def test_object_inline_data_no_attrs(self):
        part = _ns(inline_data="raw_data")
        result = _format_parts_as_content_blocks([part])
        assert result[0]["type"] == "image"
        assert result[0]["inline_data"] == "raw_data"

    def test_empty_list(self):
        assert _format_parts_as_content_blocks([]) == []

    def test_mixed_parts(self):
        parts = [{"text": "hello"}, "world", {"inline_data": {"mime_type": "image/png", "data": "x"}}]
        result = _format_parts_as_content_blocks(parts)
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "text"
        assert result[2]["type"] == "image"


# ---------------------------------------------------------------------------
# _format_dict_message
# ---------------------------------------------------------------------------

class TestFormatDictMessage:
    def test_parts_format(self):
        msg = {"role": "user", "parts": [{"text": "hi"}]}
        result = _format_dict_message(msg)
        assert result["role"] == "user"
        assert result["content"] == [{"type": "text", "text": "hi"}]

    def test_content_string(self):
        msg = {"role": "assistant", "content": "answer"}
        result = _format_dict_message(msg)
        assert result["content"] == "answer"

    def test_content_list(self):
        msg = {"content": [{"text": "hi"}]}
        result = _format_dict_message(msg)
        assert result["content"] == [{"type": "text", "text": "hi"}]

    def test_content_non_string(self):
        msg = {"content": 42}
        result = _format_dict_message(msg)
        assert result["content"] == "42"

    def test_text_field(self):
        msg = {"text": "hello"}
        result = _format_dict_message(msg)
        assert result["content"] == "hello"

    def test_fallback(self):
        msg = {"unknown": "field"}
        result = _format_dict_message(msg)
        assert result["role"] == "user"
        assert "unknown" in result["content"]

    def test_default_role(self):
        msg = {"content": "hi"}
        result = _format_dict_message(msg)
        assert result["role"] == "user"


# ---------------------------------------------------------------------------
# _format_object_message
# ---------------------------------------------------------------------------

class TestFormatObjectMessage:
    def test_parts_attr(self):
        item = _ns(parts=[_ns(text="hello")], role="model")
        result = _format_object_message(item)
        assert result["role"] == "model"
        assert result["content"] == [{"type": "text", "text": "hello"}]

    def test_text_attr(self):
        item = _ns(text="world", role="user")
        result = _format_object_message(item)
        assert result["content"] == "world"

    def test_content_attr_string(self):
        item = _ns(content="data", role="assistant")
        result = _format_object_message(item)
        assert result["content"] == "data"

    def test_content_attr_list(self):
        item = _ns(content=[{"text": "hi"}], role="user")
        result = _format_object_message(item)
        assert result["content"] == [{"type": "text", "text": "hi"}]

    def test_content_attr_non_string(self):
        item = _ns(content=42, role="user")
        result = _format_object_message(item)
        assert result["content"] == "42"

    def test_fallback(self):
        item = _ns(unknown="value")
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_non_string_role(self):
        item = _ns(text="hi", role=123)
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_no_role(self):
        item = _ns(text="hi")
        result = _format_object_message(item)
        assert result["role"] == "user"


# ---------------------------------------------------------------------------
# format_gemini_response
# ---------------------------------------------------------------------------

class TestFormatGeminiResponse:
    def test_none_response(self):
        assert format_gemini_response(None) == []

    def test_text_candidate(self):
        part = _ns(text="Hello")
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_empty_text_skipped(self):
        part = _ns(text="")
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        assert format_gemini_response(resp) == []

    def test_function_call(self):
        fc = _ns(name="get_weather", args={"city": "NY"})
        part = _ns(function_call=fc)
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"][0]["type"] == "function"
        assert result[0]["content"][0]["function"]["name"] == "get_weather"

    def test_inline_data_audio(self):
        raw_bytes = b"audio_data"
        inline_data = _ns(mime_type="audio/pcm", data=raw_bytes)
        part = _ns(inline_data=inline_data)
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"][0]["type"] == "audio"
        assert result[0]["content"][0]["data"] == base64.b64encode(raw_bytes).decode("utf-8")

    def test_inline_data_already_base64(self):
        part = _ns(inline_data=_ns(mime_type="audio/wav", data="already_base64"))
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"][0]["data"] == "already_base64"

    def test_candidate_text_fallback(self):
        candidate = _ns(text="fallback text")
        resp = _ns(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"][0]["text"] == "fallback text"

    def test_response_text_fallback(self):
        resp = _ns(text="top-level text")
        result = format_gemini_response(resp)
        assert result[0]["content"][0]["text"] == "top-level text"

    def test_empty_candidates(self):
        resp = _ns(candidates=[])
        assert format_gemini_response(resp) == []

    def test_no_candidates_no_text(self):
        resp = _ns()
        assert format_gemini_response(resp) == []

    def test_multiple_candidates(self):
        p1 = _ns(text="a")
        p2 = _ns(text="b")
        c1 = _ns(content=_ns(parts=[p1]))
        c2 = _ns(content=_ns(parts=[p2]))
        resp = _ns(candidates=[c1, c2])
        result = format_gemini_response(resp)
        assert len(result) == 2

    def test_candidate_no_content_no_text(self):
        candidate = _ns()
        resp = _ns(candidates=[candidate])
        assert format_gemini_response(resp) == []

    def test_mixed_parts(self):
        parts = [_ns(text="hi"), _ns(function_call=_ns(name="fn", args={}))]
        content = _ns(parts=parts)
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        result = format_gemini_response(resp)
        assert len(result[0]["content"]) == 2


# ---------------------------------------------------------------------------
# extract_gemini_stop_reason
# ---------------------------------------------------------------------------

class TestExtractGeminiStopReason:
    def test_enum_finish_reason(self):
        candidate = _ns(finish_reason=_ns(name="STOP"))
        resp = _ns(candidates=[candidate])
        assert extract_gemini_stop_reason(resp) == "STOP"

    def test_string_finish_reason(self):
        candidate = _ns(finish_reason="MAX_TOKENS")
        resp = _ns(candidates=[candidate])
        assert extract_gemini_stop_reason(resp) == "MAX_TOKENS"

    def test_none_response(self):
        assert extract_gemini_stop_reason(None) is None

    def test_no_candidates(self):
        resp = _ns(candidates=[])
        assert extract_gemini_stop_reason(resp) is None


class TestExtractGeminiStopReasonFromChunk:
    def test_delegates_to_stop_reason(self):
        candidate = _ns(finish_reason=_ns(name="STOP"))
        chunk = _ns(candidates=[candidate])
        assert extract_gemini_stop_reason_from_chunk(chunk) == "STOP"


# ---------------------------------------------------------------------------
# extract_gemini_system_instruction
# ---------------------------------------------------------------------------

class TestExtractGeminiSystemInstruction:
    def test_none_config(self):
        assert extract_gemini_system_instruction(None) is None

    def test_attr(self):
        config = _ns(system_instruction="Be helpful")
        assert extract_gemini_system_instruction(config) == "Be helpful"

    def test_dict_system_instruction(self):
        config = {"system_instruction": "Be concise"}
        assert extract_gemini_system_instruction(config) == "Be concise"

    def test_dict_camelCase(self):
        config = {"systemInstruction": "Be polite"}
        assert extract_gemini_system_instruction(config) == "Be polite"

    def test_dict_no_key(self):
        assert extract_gemini_system_instruction({}) is None


# ---------------------------------------------------------------------------
# extract_gemini_tools
# ---------------------------------------------------------------------------

class TestExtractGeminiTools:
    def test_config_with_tools(self):
        config = _ns(tools=[{"name": "search"}])
        result = extract_gemini_tools({"config": config})
        assert result == [{"name": "search"}]

    def test_no_config(self):
        assert extract_gemini_tools({"model": "gemini"}) is None

    def test_config_no_tools(self):
        assert extract_gemini_tools({"config": _ns()}) is None


# ---------------------------------------------------------------------------
# format_gemini_input_with_system
# ---------------------------------------------------------------------------

class TestFormatGeminiInputWithSystem:
    def test_adds_system_from_config(self):
        contents = "Hello"
        config = _ns(system_instruction="Be helpful")
        result = format_gemini_input_with_system(contents, config)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"
        assert result[1]["role"] == "user"

    def test_no_duplicate_system(self):
        contents = [{"role": "system", "content": "existing"}, {"role": "user", "content": "hi"}]
        config = _ns(system_instruction="Be helpful")
        result = format_gemini_input_with_system(contents, config)
        # System already exists, shouldn't add another
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 1

    def test_no_config(self):
        result = format_gemini_input_with_system("hi")
        assert len(result) == 1

    def test_config_no_system(self):
        result = format_gemini_input_with_system("hi", {})
        assert len(result) == 1


# ---------------------------------------------------------------------------
# format_gemini_input
# ---------------------------------------------------------------------------

class TestFormatGeminiInput:
    def test_string_input(self):
        result = format_gemini_input("hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_list_of_strings(self):
        result = format_gemini_input(["a", "b"])
        assert len(result) == 2
        assert result[0]["content"] == "a"

    def test_list_of_dicts(self):
        result = format_gemini_input([{"role": "user", "content": "hi"}])
        assert result[0]["role"] == "user"

    def test_list_of_objects(self):
        result = format_gemini_input([_ns(text="hi", role="user")])
        assert result[0]["content"] == "hi"

    def test_single_dict(self):
        result = format_gemini_input({"role": "user", "content": "hi"})
        assert len(result) == 1

    def test_single_object(self):
        result = format_gemini_input(_ns(text="hi"))
        assert len(result) == 1

    def test_mixed_list(self):
        result = format_gemini_input(["str", {"content": "dict"}, _ns(text="obj")])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# extract_gemini_web_search_count
# ---------------------------------------------------------------------------

class TestExtractGeminiWebSearchCount:
    def test_no_candidates(self):
        resp = _ns()
        assert extract_gemini_web_search_count(resp) == 0

    def test_grounding_with_queries(self):
        grounding = _ns(web_search_queries=["query1"])
        candidate = _ns(grounding_metadata=grounding, content=_ns(parts=[]))
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_grounding_with_chunks(self):
        grounding = _ns(grounding_chunks=[{"chunk": "data"}])
        candidate = _ns(grounding_metadata=grounding, content=_ns(parts=[]))
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_google_search_function_call(self):
        fc = _ns(name="google_search", args={})
        part = _ns(function_call=fc)
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_grounding_function_call(self):
        fc = _ns(name="grounding_tool", args={})
        part = _ns(function_call=fc)
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_no_grounding(self):
        part = _ns(text="hi")
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 0

    def test_empty_grounding_queries(self):
        grounding = _ns(web_search_queries=[])
        candidate = _ns(grounding_metadata=grounding, content=_ns(parts=[]))
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 0

    def test_none_grounding_queries(self):
        grounding = _ns(web_search_queries=None)
        candidate = _ns(grounding_metadata=grounding, content=_ns(parts=[]))
        resp = _ns(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 0


# ---------------------------------------------------------------------------
# _extract_usage_from_metadata
# ---------------------------------------------------------------------------

class TestExtractUsageFromMetadata:
    def test_basic(self):
        metadata = _ns(prompt_token_count=100, candidates_token_count=50)
        metadata.model_dump = lambda: {}
        result = _extract_usage_from_metadata(metadata)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_with_cache(self):
        metadata = _ns(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=20,
        )
        metadata.model_dump = lambda: {}
        result = _extract_usage_from_metadata(metadata)
        assert result["cache_read_input_tokens"] == 20

    def test_zero_cache_not_added(self):
        metadata = _ns(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=0,
        )
        metadata.model_dump = lambda: {}
        result = _extract_usage_from_metadata(metadata)
        assert "cache_read_input_tokens" not in result

    def test_with_reasoning(self):
        metadata = _ns(
            prompt_token_count=100,
            candidates_token_count=50,
            thoughts_token_count=15,
        )
        metadata.model_dump = lambda: {}
        result = _extract_usage_from_metadata(metadata)
        assert result["reasoning_tokens"] == 15

    def test_raw_usage_captured(self):
        metadata = _ns(prompt_token_count=10, candidates_token_count=5)
        metadata.model_dump = lambda: {"prompt_token_count": 10}
        result = _extract_usage_from_metadata(metadata)
        assert "raw_usage" in result


# ---------------------------------------------------------------------------
# extract_gemini_usage_from_response
# ---------------------------------------------------------------------------

class TestExtractGeminiUsageFromResponse:
    def test_no_metadata(self):
        resp = _ns()
        result = extract_gemini_usage_from_response(resp)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_none_metadata(self):
        resp = _ns(usage_metadata=None)
        result = extract_gemini_usage_from_response(resp)
        assert result["input_tokens"] == 0

    def test_with_metadata(self):
        metadata = _ns(prompt_token_count=200, candidates_token_count=100)
        metadata.model_dump = lambda: {}
        resp = _ns(usage_metadata=metadata)
        result = extract_gemini_usage_from_response(resp)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 100

    def test_with_web_search(self):
        metadata = _ns(prompt_token_count=10, candidates_token_count=5)
        metadata.model_dump = lambda: {}
        grounding = _ns(web_search_queries=["q1"])
        candidate = _ns(grounding_metadata=grounding, content=_ns(parts=[]))
        resp = _ns(usage_metadata=metadata, candidates=[candidate])
        result = extract_gemini_usage_from_response(resp)
        assert result["web_search_count"] == 1


# ---------------------------------------------------------------------------
# extract_gemini_usage_from_chunk
# ---------------------------------------------------------------------------

class TestExtractGeminiUsageFromChunk:
    def test_no_metadata(self):
        chunk = _ns()
        result = extract_gemini_usage_from_chunk(chunk)
        assert result == {}

    def test_with_metadata(self):
        metadata = _ns(prompt_token_count=10, candidates_token_count=5)
        metadata.model_dump = lambda: {}
        chunk = _ns(usage_metadata=metadata)
        result = extract_gemini_usage_from_chunk(chunk)
        assert result["input_tokens"] == 10

    def test_web_search_in_chunk(self):
        grounding = _ns(web_search_queries=["q1"])
        candidate = _ns(grounding_metadata=grounding, content=_ns(parts=[]))
        chunk = _ns(candidates=[candidate])
        result = extract_gemini_usage_from_chunk(chunk)
        assert result["web_search_count"] == 1


# ---------------------------------------------------------------------------
# extract_gemini_content_from_chunk
# ---------------------------------------------------------------------------

class TestExtractGeminiContentFromChunk:
    def test_text_chunk(self):
        chunk = _ns(text="hello")
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {"type": "text", "text": "hello"}

    def test_function_call_in_candidates(self):
        fc = _ns(name="search", args={"q": "test"})
        part = _ns(function_call=fc)
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        chunk = _ns(candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"

    def test_text_in_candidate_parts(self):
        part = _ns(text="hi")
        content = _ns(parts=[part])
        candidate = _ns(content=content)
        chunk = _ns(candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {"type": "text", "text": "hi"}

    def test_empty_chunk(self):
        chunk = _ns()
        assert extract_gemini_content_from_chunk(chunk) is None

    def test_empty_text(self):
        chunk = _ns(text="")
        assert extract_gemini_content_from_chunk(chunk) is None


# ---------------------------------------------------------------------------
# format_gemini_streaming_output
# ---------------------------------------------------------------------------

class TestFormatGeminiStreamingOutput:
    def test_string_input(self):
        result = format_gemini_streaming_output("hello")
        assert result[0]["content"][0]["text"] == "hello"

    def test_list_of_strings(self):
        result = format_gemini_streaming_output(["a", "b"])
        assert result[0]["content"][0]["text"] == "ab"

    def test_list_of_text_blocks(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        result = format_gemini_streaming_output(blocks)
        assert result[0]["content"][0]["text"] == "ab"

    def test_function_call_block(self):
        blocks = [
            {"type": "text", "text": "thinking"},
            {"type": "function", "function": {"name": "fn", "arguments": {}}},
            {"type": "text", "text": "done"},
        ]
        result = format_gemini_streaming_output(blocks)
        content = result[0]["content"]
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "thinking"
        assert content[1]["type"] == "function"
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "done"

    def test_empty_string(self):
        result = format_gemini_streaming_output("")
        assert result[0]["content"][0]["text"] == ""

    def test_empty_list(self):
        result = format_gemini_streaming_output([])
        assert result[0]["content"][0]["text"] == ""

    def test_mixed_strings_and_blocks(self):
        items = ["a", {"type": "text", "text": "b"}]
        result = format_gemini_streaming_output(items)
        assert result[0]["content"][0]["text"] == "ab"


# ---------------------------------------------------------------------------
# extract_gemini_embedding_token_count
# ---------------------------------------------------------------------------

class TestExtractGeminiEmbeddingTokenCount:
    def test_no_embeddings(self):
        resp = _ns()
        assert extract_gemini_embedding_token_count(resp) == 0

    def test_with_token_counts(self):
        e1 = _ns(statistics=_ns(token_count=10))
        e2 = _ns(statistics=_ns(token_count=20))
        resp = _ns(embeddings=[e1, e2])
        assert extract_gemini_embedding_token_count(resp) == 30

    def test_no_statistics(self):
        e1 = _ns(statistics=None)
        resp = _ns(embeddings=[e1])
        assert extract_gemini_embedding_token_count(resp) == 0

    def test_none_token_count(self):
        e1 = _ns(statistics=_ns(token_count=None))
        resp = _ns(embeddings=[e1])
        assert extract_gemini_embedding_token_count(resp) == 0

    def test_empty_embeddings(self):
        resp = _ns(embeddings=[])
        assert extract_gemini_embedding_token_count(resp) == 0
