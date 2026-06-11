"""Unit tests for posthog.ai.gemini.gemini_converter."""

import base64

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Obj:
    """Simple namespace for mock objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ===========================================================================
# _format_parts_as_content_blocks
# ===========================================================================


class TestFormatPartsAsContentBlocks:
    def test_dict_text(self):
        parts = [{"text": "Hello"}]
        result = _format_parts_as_content_blocks(parts)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_string_part(self):
        result = _format_parts_as_content_blocks(["raw string"])
        assert result == [{"type": "text", "text": "raw string"}]

    def test_dict_inline_data_image(self):
        parts = [{"inline_data": {"mime_type": "image/png", "data": "base64data"}}]
        result = _format_parts_as_content_blocks(parts)
        assert result[0]["type"] == "image"
        assert result[0]["inline_data"]["mime_type"] == "image/png"

    def test_dict_inline_data_document(self):
        parts = [{"inline_data": {"mime_type": "application/pdf", "data": "pdfdata"}}]
        result = _format_parts_as_content_blocks(parts)
        assert result[0]["type"] == "document"

    def test_object_with_text_attr(self):
        part = _Obj(text="obj text")
        result = _format_parts_as_content_blocks([part])
        assert result == [{"type": "text", "text": "obj text"}]

    def test_object_with_empty_text(self):
        part = _Obj(text="")
        result = _format_parts_as_content_blocks([part])
        assert result == []

    def test_object_with_inline_data_typed(self):
        inline = _Obj(mime_type="image/jpeg", data="imgdata")
        part = _Obj(inline_data=inline)
        result = _format_parts_as_content_blocks([part])
        assert result[0]["type"] == "image"
        assert result[0]["inline_data"]["mime_type"] == "image/jpeg"
        assert result[0]["inline_data"]["data"] == "imgdata"

    def test_object_with_inline_data_document(self):
        inline = _Obj(mime_type="application/pdf", data="pdfdata")
        part = _Obj(inline_data=inline)
        result = _format_parts_as_content_blocks([part])
        assert result[0]["type"] == "document"

    def test_object_with_inline_data_untyped(self):
        """inline_data without mime_type/data attributes → fallback."""
        inline = {"raw": "stuff"}
        part = _Obj(inline_data=inline)
        result = _format_parts_as_content_blocks([part])
        assert result[0]["type"] == "image"
        assert result[0]["inline_data"] == {"raw": "stuff"}


# ===========================================================================
# _format_dict_message
# ===========================================================================


class TestFormatDictMessage:
    def test_with_parts(self):
        msg = {"role": "user", "parts": [{"text": "Hello"}]}
        result = _format_dict_message(msg)
        assert result["role"] == "user"
        assert result["content"] == [{"type": "text", "text": "Hello"}]

    def test_with_content_string(self):
        msg = {"role": "assistant", "content": "text"}
        result = _format_dict_message(msg)
        assert result == {"role": "assistant", "content": "text"}

    def test_with_content_list(self):
        msg = {"content": [{"text": "a"}, {"text": "b"}]}
        result = _format_dict_message(msg)
        assert result["role"] == "user"
        assert len(result["content"]) == 2

    def test_with_content_non_string(self):
        msg = {"content": 42}
        result = _format_dict_message(msg)
        assert result["content"] == "42"

    def test_with_text_field(self):
        msg = {"text": "fallback text", "role": "model"}
        result = _format_dict_message(msg)
        assert result["content"] == "fallback text"
        assert result["role"] == "model"

    def test_fallback(self):
        msg = {"data": 123}
        result = _format_dict_message(msg)
        assert result["role"] == "user"
        assert "123" in result["content"]

    def test_default_role(self):
        msg = {"parts": [{"text": "hi"}]}
        assert _format_dict_message(msg)["role"] == "user"


# ===========================================================================
# _format_object_message
# ===========================================================================


class TestFormatObjectMessage:
    def test_with_parts(self):
        part = _Obj(text="Hello")
        item = _Obj(parts=[part], role="model")
        result = _format_object_message(item)
        assert result["role"] == "model"
        assert result["content"] == [{"type": "text", "text": "Hello"}]

    def test_with_text(self):
        item = _Obj(text="Direct text", role="assistant")
        result = _format_object_message(item)
        assert result["content"] == "Direct text"
        assert result["role"] == "assistant"

    def test_with_content_string(self):
        item = _Obj(content="content string", role="user")
        result = _format_object_message(item)
        assert result["content"] == "content string"

    def test_with_content_list(self):
        item = _Obj(content=[{"text": "a"}], role="user")
        result = _format_object_message(item)
        assert isinstance(result["content"], list)

    def test_with_content_non_string(self):
        item = _Obj(content=123, role="user")
        result = _format_object_message(item)
        assert result["content"] == "123"

    def test_fallback_str(self):
        item = _Obj(data=42)
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_non_string_role_defaults(self):
        """Non-string role attribute → defaults to 'user'."""
        item = _Obj(text="hi", role=123)
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_parts_non_string_role(self):
        part = _Obj(text="hi")
        item = _Obj(parts=[part], role=None)
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_content_non_string_role(self):
        item = _Obj(content="hi", role=True)
        result = _format_object_message(item)
        assert result["role"] == "user"


# ===========================================================================
# format_gemini_response
# ===========================================================================


class TestFormatGeminiResponse:
    def test_none(self):
        assert format_gemini_response(None) == []

    def test_text_candidate(self):
        part = _Obj(text="Hello")
        content = _Obj(parts=[part])
        candidate = _Obj(content=content)
        resp = _Obj(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_function_call_candidate(self):
        fn = _Obj(name="get_weather", args={"city": "NYC"})
        part = _Obj(text=None, function_call=fn)
        content = _Obj(parts=[part])
        candidate = _Obj(content=content)
        resp = _Obj(candidates=[candidate])
        result = format_gemini_response(resp)
        items = result[0]["content"]
        assert items[0]["type"] == "function"
        assert items[0]["function"]["name"] == "get_weather"

    def test_inline_data_audio(self):
        raw_bytes = b"\x00\x01\x02"
        inline = _Obj(mime_type="audio/pcm", data=raw_bytes)
        part = _Obj(text=None, function_call=None, inline_data=inline)
        content = _Obj(parts=[part])
        candidate = _Obj(content=content)
        resp = _Obj(candidates=[candidate])
        result = format_gemini_response(resp)
        items = result[0]["content"]
        assert items[0]["type"] == "audio"
        assert items[0]["data"] == base64.b64encode(raw_bytes).decode("utf-8")

    def test_inline_data_string(self):
        inline = _Obj(mime_type="audio/pcm", data="already_base64")
        part = _Obj(text=None, function_call=None, inline_data=inline)
        content = _Obj(parts=[part])
        candidate = _Obj(content=content)
        resp = _Obj(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"][0]["data"] == "already_base64"

    def test_candidate_text_fallback(self):
        """Candidate has .text but no .content."""
        candidate = _Obj(text="fallback text")
        resp = _Obj(candidates=[candidate])
        result = format_gemini_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "fallback text"}]

    def test_response_text_fallback(self):
        """No candidates, but response has .text."""
        resp = _Obj(text="top level text", candidates=None)
        result = format_gemini_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "top level text"}]

    def test_empty_candidates(self):
        resp = _Obj(candidates=[])
        assert format_gemini_response(resp) == []

    def test_candidate_no_content_no_text(self):
        candidate = _Obj(content=None)
        resp = _Obj(candidates=[candidate])
        assert format_gemini_response(resp) == []

    def test_empty_parts(self):
        content = _Obj(parts=[])
        candidate = _Obj(content=content)
        resp = _Obj(candidates=[candidate])
        assert format_gemini_response(resp) == []


# ===========================================================================
# extract_gemini_stop_reason
# ===========================================================================


class TestExtractGeminiStopReason:
    def test_enum_stop_reason(self):
        finish = _Obj(name="STOP")
        candidate = _Obj(finish_reason=finish)
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_stop_reason(resp) == "STOP"

    def test_string_stop_reason(self):
        candidate = _Obj(finish_reason="MAX_TOKENS")
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_stop_reason(resp) == "MAX_TOKENS"

    def test_no_candidates(self):
        resp = _Obj(candidates=[])
        assert extract_gemini_stop_reason(resp) is None

    def test_none_response(self):
        assert extract_gemini_stop_reason(None) is None

    def test_chunk_delegates(self):
        candidate = _Obj(finish_reason="STOP")
        chunk = _Obj(candidates=[candidate])
        assert extract_gemini_stop_reason_from_chunk(chunk) == "STOP"


# ===========================================================================
# extract_gemini_system_instruction
# ===========================================================================


class TestExtractGeminiSystemInstruction:
    def test_none_config(self):
        assert extract_gemini_system_instruction(None) is None

    def test_object_config(self):
        config = _Obj(system_instruction="Be helpful")
        assert extract_gemini_system_instruction(config) == "Be helpful"

    def test_dict_system_instruction(self):
        config = {"system_instruction": "Be terse"}
        assert extract_gemini_system_instruction(config) == "Be terse"

    def test_dict_camel_case(self):
        config = {"systemInstruction": "Camel"}
        assert extract_gemini_system_instruction(config) == "Camel"

    def test_no_instruction(self):
        assert extract_gemini_system_instruction({}) is None


# ===========================================================================
# extract_gemini_tools
# ===========================================================================


class TestExtractGeminiTools:
    def test_with_tools(self):
        config = _Obj(tools=["tool1"])
        result = extract_gemini_tools({"config": config})
        assert result == ["tool1"]

    def test_no_config(self):
        assert extract_gemini_tools({}) is None

    def test_config_no_tools(self):
        config = _Obj()
        assert extract_gemini_tools({"config": config}) is None


# ===========================================================================
# format_gemini_input_with_system
# ===========================================================================


class TestFormatGeminiInputWithSystem:
    def test_prepends_system(self):
        result = format_gemini_input_with_system(
            "Hello", config={"system_instruction": "Sys"}
        )
        assert result[0] == {"role": "system", "content": "Sys"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_no_duplicate_system(self):
        """If messages already have system role, don't add another."""
        contents = [
            {"role": "system", "content": "existing"},
            {"role": "user", "content": "q"},
        ]
        result = format_gemini_input_with_system(
            contents, config={"system_instruction": "new"}
        )
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 1

    def test_no_config(self):
        result = format_gemini_input_with_system("Hello")
        assert len(result) == 1


# ===========================================================================
# format_gemini_input
# ===========================================================================


class TestFormatGeminiInput:
    def test_string_input(self):
        result = format_gemini_input("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_list_of_strings(self):
        result = format_gemini_input(["a", "b"])
        assert len(result) == 2
        assert result[0]["content"] == "a"

    def test_list_of_dicts(self):
        result = format_gemini_input([{"role": "user", "content": "hi"}])
        assert result[0] == {"role": "user", "content": "hi"}

    def test_list_of_objects(self):
        item = _Obj(text="obj", role="user")
        result = format_gemini_input([item])
        assert result[0]["content"] == "obj"

    def test_single_dict(self):
        result = format_gemini_input({"role": "user", "content": "single"})
        assert result == [{"role": "user", "content": "single"}]

    def test_single_object(self):
        item = _Obj(text="single obj")
        result = format_gemini_input(item)
        assert result[0]["content"] == "single obj"


# ===========================================================================
# extract_gemini_web_search_count
# ===========================================================================


class TestExtractGeminiWebSearchCount:
    def test_no_candidates(self):
        resp = _Obj(candidates=[])
        assert extract_gemini_web_search_count(resp) == 0

    def test_grounding_web_search_queries(self):
        grounding = _Obj(web_search_queries=["query1"])
        candidate = _Obj(grounding_metadata=grounding, content=None)
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_grounding_chunks(self):
        grounding = _Obj(grounding_chunks=["chunk1"])
        candidate = _Obj(grounding_metadata=grounding, content=None)
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_google_search_function_call(self):
        fn = _Obj(name="google_search_tool")
        part = _Obj(function_call=fn, text=None)
        content = _Obj(parts=[part])
        candidate = _Obj(grounding_metadata=None, content=content)
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 1

    def test_empty_queries_and_chunks(self):
        grounding = _Obj(web_search_queries=[], grounding_chunks=[])
        candidate = _Obj(grounding_metadata=grounding, content=None)
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 0

    def test_no_grounding_metadata(self):
        part = _Obj(text="hi", function_call=None)
        content = _Obj(parts=[part])
        candidate = _Obj(grounding_metadata=None, content=content)
        resp = _Obj(candidates=[candidate])
        assert extract_gemini_web_search_count(resp) == 0

    def test_no_candidates_attr(self):
        resp = _Obj()
        assert extract_gemini_web_search_count(resp) == 0


# ===========================================================================
# _extract_usage_from_metadata
# ===========================================================================


class TestExtractUsageFromMetadata:
    def test_basic(self):
        meta = _Obj(prompt_token_count=100, candidates_token_count=50)
        result = _extract_usage_from_metadata(meta)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_with_cache(self):
        meta = _Obj(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=20,
        )
        result = _extract_usage_from_metadata(meta)
        assert result["cache_read_input_tokens"] == 20

    def test_with_reasoning(self):
        meta = _Obj(
            prompt_token_count=100, candidates_token_count=50, thoughts_token_count=10
        )
        result = _extract_usage_from_metadata(meta)
        assert result["reasoning_tokens"] == 10

    def test_zero_cache_not_included(self):
        meta = _Obj(
            prompt_token_count=100,
            candidates_token_count=50,
            cached_content_token_count=0,
        )
        result = _extract_usage_from_metadata(meta)
        assert "cache_read_input_tokens" not in result

    def test_zero_reasoning_not_included(self):
        meta = _Obj(
            prompt_token_count=100, candidates_token_count=50, thoughts_token_count=0
        )
        result = _extract_usage_from_metadata(meta)
        assert "reasoning_tokens" not in result


# ===========================================================================
# extract_gemini_usage_from_response
# ===========================================================================


class TestExtractGeminiUsageFromResponse:
    def test_no_metadata(self):
        resp = _Obj()
        result = extract_gemini_usage_from_response(resp)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_none_metadata(self):
        resp = _Obj(usage_metadata=None)
        result = extract_gemini_usage_from_response(resp)
        assert result["input_tokens"] == 0

    def test_with_metadata(self):
        meta = _Obj(prompt_token_count=200, candidates_token_count=100)
        resp = _Obj(usage_metadata=meta, candidates=[])
        result = extract_gemini_usage_from_response(resp)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 100

    def test_with_web_search(self):
        grounding = _Obj(web_search_queries=["q1"])
        candidate = _Obj(grounding_metadata=grounding, content=None)
        meta = _Obj(prompt_token_count=50, candidates_token_count=25)
        resp = _Obj(usage_metadata=meta, candidates=[candidate])
        result = extract_gemini_usage_from_response(resp)
        assert result["web_search_count"] == 1


# ===========================================================================
# extract_gemini_usage_from_chunk
# ===========================================================================


class TestExtractGeminiUsageFromChunk:
    def test_no_metadata(self):
        chunk = _Obj()
        result = extract_gemini_usage_from_chunk(chunk)
        assert result.get("input_tokens") is None

    def test_none_metadata(self):
        chunk = _Obj(usage_metadata=None)
        result = extract_gemini_usage_from_chunk(chunk)
        assert result.get("input_tokens") is None

    def test_with_metadata(self):
        meta = _Obj(prompt_token_count=50, candidates_token_count=25)
        chunk = _Obj(usage_metadata=meta, candidates=[])
        result = extract_gemini_usage_from_chunk(chunk)
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 25

    def test_with_web_search(self):
        grounding = _Obj(web_search_queries=["q"])
        candidate = _Obj(grounding_metadata=grounding, content=None)
        meta = _Obj(prompt_token_count=50, candidates_token_count=25)
        chunk = _Obj(usage_metadata=meta, candidates=[candidate])
        result = extract_gemini_usage_from_chunk(chunk)
        assert result["web_search_count"] == 1


# ===========================================================================
# extract_gemini_content_from_chunk
# ===========================================================================


class TestExtractGeminiContentFromChunk:
    def test_text_content(self):
        chunk = _Obj(text="hello", candidates=[])
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {"type": "text", "text": "hello"}

    def test_function_call_content(self):
        fn = _Obj(name="calc", args={"x": 1})
        part = _Obj(function_call=fn, text=None)
        content = _Obj(parts=[part])
        candidate = _Obj(content=content)
        chunk = _Obj(text=None, candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result["type"] == "function"
        assert result["function"]["name"] == "calc"

    def test_text_in_candidate_parts(self):
        part = _Obj(function_call=None, text="from parts")
        content = _Obj(parts=[part])
        candidate = _Obj(content=content)
        chunk = _Obj(text=None, candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {"type": "text", "text": "from parts"}

    def test_no_content(self):
        chunk = _Obj(text=None, candidates=[])
        assert extract_gemini_content_from_chunk(chunk) is None

    def test_empty_parts(self):
        content = _Obj(parts=[])
        candidate = _Obj(content=content)
        chunk = _Obj(text=None, candidates=[candidate])
        assert extract_gemini_content_from_chunk(chunk) is None


# ===========================================================================
# format_gemini_streaming_output
# ===========================================================================


class TestFormatGeminiStreamingOutput:
    def test_string(self):
        result = format_gemini_streaming_output("Hello world")
        assert result[0]["content"] == [{"type": "text", "text": "Hello world"}]

    def test_list_of_strings(self):
        result = format_gemini_streaming_output(["Hello", " world"])
        assert result[0]["content"] == [{"type": "text", "text": "Hello world"}]

    def test_list_of_text_blocks(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        result = format_gemini_streaming_output(blocks)
        assert result[0]["content"] == [{"type": "text", "text": "ab"}]

    def test_list_with_function_call(self):
        blocks = [
            {"type": "text", "text": "before"},
            {"type": "function", "function": {"name": "fn"}},
            {"type": "text", "text": "after"},
        ]
        result = format_gemini_streaming_output(blocks)
        content = result[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "before"
        assert content[1]["type"] == "function"
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "after"

    def test_empty_list(self):
        result = format_gemini_streaming_output([])
        assert result[0]["content"] == [{"type": "text", "text": ""}]

    def test_empty_string(self):
        result = format_gemini_streaming_output("")
        assert result[0]["content"] == [{"type": "text", "text": ""}]


# ===========================================================================
# extract_gemini_embedding_token_count
# ===========================================================================


class TestExtractGeminiEmbeddingTokenCount:
    def test_no_embeddings(self):
        resp = _Obj()
        assert extract_gemini_embedding_token_count(resp) == 0

    def test_empty_embeddings(self):
        resp = _Obj(embeddings=[])
        assert extract_gemini_embedding_token_count(resp) == 0

    def test_with_token_counts(self):
        emb1 = _Obj(statistics=_Obj(token_count=10))
        emb2 = _Obj(statistics=_Obj(token_count=15))
        resp = _Obj(embeddings=[emb1, emb2])
        assert extract_gemini_embedding_token_count(resp) == 25

    def test_no_statistics(self):
        emb = _Obj(statistics=None)
        resp = _Obj(embeddings=[emb])
        assert extract_gemini_embedding_token_count(resp) == 0

    def test_none_token_count(self):
        emb = _Obj(statistics=_Obj(token_count=None))
        resp = _Obj(embeddings=[emb])
        assert extract_gemini_embedding_token_count(resp) == 0
