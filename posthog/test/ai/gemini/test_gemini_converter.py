"""
Unit tests for posthog.ai.gemini.gemini_converter module.

Tests conversion of Gemini API responses/inputs into standardized PostHog formats.
Covers response formatting, input formatting, streaming, multimodal content,
tool calls, usage extraction, and edge cases.
"""

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


# =======================
# Mock Helpers
# =======================


class MockObj:
    """Generic mock object that sets attributes from kwargs."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# =======================
# _format_parts_as_content_blocks
# =======================


class TestFormatPartsAsContentBlocks:
    def test_dict_with_text(self):
        parts = [{"text": "Hello"}]
        result = _format_parts_as_content_blocks(parts)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_string_parts(self):
        parts = ["Hello", "World"]
        result = _format_parts_as_content_blocks(parts)
        assert result == [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]

    def test_dict_with_inline_data_image(self):
        parts = [{"inline_data": {"mime_type": "image/png", "data": "base64data"}}]
        result = _format_parts_as_content_blocks(parts)
        assert result == [
            {
                "type": "image",
                "inline_data": {"mime_type": "image/png", "data": "base64data"},
            }
        ]

    def test_dict_with_inline_data_document(self):
        parts = [{"inline_data": {"mime_type": "application/pdf", "data": "pdfdata"}}]
        result = _format_parts_as_content_blocks(parts)
        assert result == [
            {
                "type": "document",
                "inline_data": {"mime_type": "application/pdf", "data": "pdfdata"},
            }
        ]

    def test_object_with_text_attr(self):
        part = MockObj(text="Object text")
        result = _format_parts_as_content_blocks([part])
        assert result == [{"type": "text", "text": "Object text"}]

    def test_object_with_empty_text_attr(self):
        part = MockObj(text="")
        result = _format_parts_as_content_blocks([part])
        assert result == []

    def test_object_with_inline_data_attr(self):
        inline_data = MockObj(mime_type="image/jpeg", data="jpegdata")
        part = MockObj(inline_data=inline_data)
        result = _format_parts_as_content_blocks([part])
        assert result == [
            {
                "type": "image",
                "inline_data": {"mime_type": "image/jpeg", "data": "jpegdata"},
            }
        ]

    def test_object_with_inline_data_no_mime(self):
        """inline_data object without mime_type/data attrs."""
        inline_data = MockObj()
        part = MockObj(inline_data=inline_data)
        result = _format_parts_as_content_blocks([part])
        assert result == [{"type": "image", "inline_data": inline_data}]

    def test_object_with_inline_data_document_type(self):
        inline_data = MockObj(mime_type="application/pdf", data="pdfbytes")
        part = MockObj(inline_data=inline_data)
        result = _format_parts_as_content_blocks([part])
        assert result == [
            {
                "type": "document",
                "inline_data": {"mime_type": "application/pdf", "data": "pdfbytes"},
            }
        ]

    def test_empty_parts(self):
        assert _format_parts_as_content_blocks([]) == []


# =======================
# _format_dict_message
# =======================


class TestFormatDictMessage:
    def test_parts_format(self):
        item = {"role": "user", "parts": [{"text": "Hello"}]}
        result = _format_dict_message(item)
        assert result == {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }

    def test_content_string(self):
        item = {"role": "assistant", "content": "Response"}
        result = _format_dict_message(item)
        assert result == {"role": "assistant", "content": "Response"}

    def test_content_list(self):
        item = {"role": "user", "content": [{"text": "Hello"}]}
        result = _format_dict_message(item)
        assert result == {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        }

    def test_content_non_string(self):
        item = {"role": "user", "content": 42}
        result = _format_dict_message(item)
        assert result == {"role": "user", "content": "42"}

    def test_text_field(self):
        item = {"role": "user", "text": "Hello text"}
        result = _format_dict_message(item)
        assert result == {"role": "user", "content": "Hello text"}

    def test_fallback(self):
        item = {"unknown": "field"}
        result = _format_dict_message(item)
        assert result == {"role": "user", "content": str(item)}

    def test_missing_role(self):
        item = {"parts": [{"text": "No role"}]}
        result = _format_dict_message(item)
        assert result["role"] == "user"


# =======================
# _format_object_message
# =======================


class TestFormatObjectMessage:
    def test_object_with_parts(self):
        part = MockObj(text="Part text")
        item = MockObj(parts=[part], role="model")
        result = _format_object_message(item)
        assert result == {
            "role": "model",
            "content": [{"type": "text", "text": "Part text"}],
        }

    def test_object_with_text(self):
        item = MockObj(text="Hello", role="user")
        result = _format_object_message(item)
        assert result == {"role": "user", "content": "Hello"}

    def test_object_with_content_string(self):
        item = MockObj(content="Content string", role="assistant")
        result = _format_object_message(item)
        assert result == {"role": "assistant", "content": "Content string"}

    def test_object_with_content_list(self):
        item = MockObj(content=[{"text": "Listed"}], role="user")
        result = _format_object_message(item)
        assert result == {
            "role": "user",
            "content": [{"type": "text", "text": "Listed"}],
        }

    def test_object_with_content_non_string(self):
        item = MockObj(content=123, role="user")
        result = _format_object_message(item)
        assert result == {"role": "user", "content": "123"}

    def test_object_fallback(self):
        item = MockObj(unknown="x")
        result = _format_object_message(item)
        assert result["role"] == "user"
        assert result["content"] == str(item)

    def test_object_non_string_role(self):
        """Role that is not a string defaults to 'user'."""
        item = MockObj(text="Hi", role=123)
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_object_parts_non_string_role(self):
        part = MockObj(text="Part")
        item = MockObj(parts=[part], role=None)
        result = _format_object_message(item)
        assert result["role"] == "user"

    def test_object_content_non_string_role(self):
        item = MockObj(content="Hi", role=False)
        result = _format_object_message(item)
        assert result["role"] == "user"


# =======================
# format_gemini_response
# =======================


class TestFormatGeminiResponse:
    def test_none_response(self):
        assert format_gemini_response(None) == []

    def test_basic_text_response(self):
        part = MockObj(text="Hello Gemini")
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello Gemini"}]}
        ]

    def test_function_call_response(self):
        func_call = MockObj(name="get_weather", args={"city": "NYC"})
        part = MockObj(function_call=func_call, text=None)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        },
                    }
                ],
            }
        ]

    def test_inline_data_audio(self):
        raw_bytes = b"audio_bytes"
        inline_data = MockObj(mime_type="audio/pcm", data=raw_bytes)
        part = MockObj(text=None, function_call=None, inline_data=inline_data)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result[0]["content"][0]["type"] == "audio"
        assert result[0]["content"][0]["mime_type"] == "audio/pcm"
        assert result[0]["content"][0]["data"] == base64.b64encode(raw_bytes).decode(
            "utf-8"
        )

    def test_inline_data_already_base64(self):
        inline_data = MockObj(mime_type="audio/wav", data="already_base64_string")
        part = MockObj(text=None, function_call=None, inline_data=inline_data)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result[0]["content"][0]["data"] == "already_base64_string"

    def test_candidate_with_text_attr(self):
        """Candidate has text attribute but no content.parts."""
        candidate = MockObj(text="Direct text", content=None)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Direct text"}]}
        ]

    def test_response_text_attr(self):
        """Response has text attribute but no candidates."""
        response = MockObj(text="Top-level text", candidates=None)
        result = format_gemini_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Top-level text"}],
            }
        ]

    def test_empty_candidates(self):
        response = MockObj(candidates=[])
        result = format_gemini_response(response)
        assert result == []

    def test_candidate_no_content(self):
        candidate = MockObj(content=None, text=None)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result == []

    def test_candidate_content_no_parts(self):
        content = MockObj(parts=None)
        candidate = MockObj(content=content, text=None)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result == []

    def test_empty_text_part(self):
        """Parts with empty text are skipped."""
        part = MockObj(text="", function_call=None, inline_data=None)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        response = MockObj(candidates=[candidate])
        result = format_gemini_response(response)
        assert result == []


# =======================
# extract_gemini_stop_reason
# =======================


class TestExtractGeminiStopReason:
    def test_with_enum_finish_reason(self):
        finish_reason = MockObj(name="STOP")
        candidate = MockObj(finish_reason=finish_reason)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_stop_reason(response) == "STOP"

    def test_with_string_finish_reason(self):
        candidate = MockObj(finish_reason="MAX_TOKENS")
        response = MockObj(candidates=[candidate])
        assert extract_gemini_stop_reason(response) == "MAX_TOKENS"

    def test_no_candidates(self):
        response = MockObj(candidates=None)
        assert extract_gemini_stop_reason(response) is None

    def test_none_response(self):
        assert extract_gemini_stop_reason(None) is None

    def test_no_finish_reason(self):
        candidate = MockObj(finish_reason=None)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_stop_reason(response) is None

    def test_from_chunk(self):
        """extract_gemini_stop_reason_from_chunk delegates to extract_gemini_stop_reason."""
        finish_reason = MockObj(name="STOP")
        candidate = MockObj(finish_reason=finish_reason)
        chunk = MockObj(candidates=[candidate])
        assert extract_gemini_stop_reason_from_chunk(chunk) == "STOP"


# =======================
# extract_gemini_system_instruction
# =======================


class TestExtractGeminiSystemInstruction:
    def test_none_config(self):
        assert extract_gemini_system_instruction(None) is None

    def test_object_with_system_instruction(self):
        config = MockObj(system_instruction="Be helpful")
        assert extract_gemini_system_instruction(config) == "Be helpful"

    def test_dict_system_instruction(self):
        config = {"system_instruction": "Be concise"}
        assert extract_gemini_system_instruction(config) == "Be concise"

    def test_dict_camel_case(self):
        config = {"systemInstruction": "Be creative"}
        assert extract_gemini_system_instruction(config) == "Be creative"

    def test_no_system_instruction(self):
        config = {"other": "value"}
        assert extract_gemini_system_instruction(config) is None


# =======================
# extract_gemini_tools
# =======================


class TestExtractGeminiTools:
    def test_tools_in_config(self):
        config = MockObj(tools=[{"name": "fn"}])
        kwargs = {"config": config}
        assert extract_gemini_tools(kwargs) == [{"name": "fn"}]

    def test_no_config(self):
        assert extract_gemini_tools({}) is None

    def test_config_no_tools(self):
        config = MockObj()
        kwargs = {"config": config}
        assert extract_gemini_tools(kwargs) is None


# =======================
# format_gemini_input_with_system
# =======================


class TestFormatGeminiInputWithSystem:
    def test_with_system_instruction(self):
        contents = "Hello"
        config = {"system_instruction": "Be helpful"}
        result = format_gemini_input_with_system(contents, config)
        assert result[0] == {"role": "system", "content": "Be helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_without_system_instruction(self):
        contents = "Hello"
        result = format_gemini_input_with_system(contents)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_already_present(self):
        """If messages already have a system message, don't duplicate."""
        contents = [
            {"role": "system", "content": "Existing"},
            {"role": "user", "content": "Hi"},
        ]
        config = {"system_instruction": "New system"}
        result = format_gemini_input_with_system(contents, config)
        # Should NOT add a duplicate system message
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 1


# =======================
# format_gemini_input
# =======================


class TestFormatGeminiInput:
    def test_string_input(self):
        result = format_gemini_input("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_list_of_strings(self):
        result = format_gemini_input(["Hello", "World"])
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]

    def test_list_of_dicts(self):
        result = format_gemini_input([{"role": "user", "parts": [{"text": "Hi"}]}])
        assert result == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]

    def test_list_of_objects(self):
        item = MockObj(text="Object", role="user")
        result = format_gemini_input([item])
        assert result == [{"role": "user", "content": "Object"}]

    def test_single_dict(self):
        result = format_gemini_input({"role": "user", "text": "Single"})
        assert result == [{"role": "user", "content": "Single"}]

    def test_single_object(self):
        item = MockObj(text="Single obj", role="model")
        result = format_gemini_input(item)
        assert result == [{"role": "model", "content": "Single obj"}]


# =======================
# extract_gemini_web_search_count
# =======================


class TestExtractGeminiWebSearchCount:
    def test_no_candidates(self):
        response = MockObj(candidates=None)
        assert extract_gemini_web_search_count(response) == 0

    def test_no_grounding_metadata(self):
        candidate = MockObj(content=MockObj(parts=[]))
        response = MockObj(candidates=[candidate])
        assert extract_gemini_web_search_count(response) == 0

    def test_web_search_queries(self):
        grounding_metadata = MockObj(web_search_queries=["test query"])
        candidate = MockObj(grounding_metadata=grounding_metadata, content=None)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_web_search_count(response) == 1

    def test_grounding_chunks(self):
        grounding_metadata = MockObj(grounding_chunks=[{"uri": "http://example.com"}])
        candidate = MockObj(grounding_metadata=grounding_metadata, content=None)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_web_search_count(response) == 1

    def test_empty_web_search_queries(self):
        grounding_metadata = MockObj(web_search_queries=[])
        candidate = MockObj(grounding_metadata=grounding_metadata, content=None)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_web_search_count(response) == 0

    def test_google_search_function_call(self):
        func_call = MockObj(name="google_search_tool")
        part = MockObj(function_call=func_call, text=None)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content, grounding_metadata=None)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_web_search_count(response) == 1

    def test_no_function_call_parts(self):
        part = MockObj(text="Hello", function_call=None)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content, grounding_metadata=None)
        response = MockObj(candidates=[candidate])
        assert extract_gemini_web_search_count(response) == 0


# =======================
# _extract_usage_from_metadata
# =======================


class TestExtractUsageFromMetadata:
    def test_basic_usage(self):
        metadata = MockObj(prompt_token_count=50, candidates_token_count=30)
        result = _extract_usage_from_metadata(metadata)
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 30

    def test_with_cache(self):
        metadata = MockObj(
            prompt_token_count=50,
            candidates_token_count=30,
            cached_content_token_count=10,
        )
        result = _extract_usage_from_metadata(metadata)
        assert result["cache_read_input_tokens"] == 10

    def test_with_reasoning(self):
        metadata = MockObj(
            prompt_token_count=50, candidates_token_count=30, thoughts_token_count=15
        )
        result = _extract_usage_from_metadata(metadata)
        assert result["reasoning_tokens"] == 15

    def test_zero_cache_not_included(self):
        metadata = MockObj(
            prompt_token_count=50,
            candidates_token_count=30,
            cached_content_token_count=0,
        )
        result = _extract_usage_from_metadata(metadata)
        assert "cache_read_input_tokens" not in result

    def test_zero_reasoning_not_included(self):
        metadata = MockObj(
            prompt_token_count=50, candidates_token_count=30, thoughts_token_count=0
        )
        result = _extract_usage_from_metadata(metadata)
        assert "reasoning_tokens" not in result


# =======================
# extract_gemini_usage_from_response
# =======================


class TestExtractGeminiUsageFromResponse:
    def test_no_usage_metadata(self):
        response = MockObj()
        result = extract_gemini_usage_from_response(response)
        assert result == {"input_tokens": 0, "output_tokens": 0}

    def test_none_usage_metadata(self):
        response = MockObj(usage_metadata=None)
        result = extract_gemini_usage_from_response(response)
        assert result == {"input_tokens": 0, "output_tokens": 0}

    def test_basic_usage(self):
        metadata = MockObj(prompt_token_count=100, candidates_token_count=50)
        response = MockObj(usage_metadata=metadata, candidates=None)
        result = extract_gemini_usage_from_response(response)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_with_web_search(self):
        metadata = MockObj(prompt_token_count=100, candidates_token_count=50)
        grounding_metadata = MockObj(web_search_queries=["query"])
        candidate = MockObj(grounding_metadata=grounding_metadata, content=None)
        response = MockObj(usage_metadata=metadata, candidates=[candidate])
        result = extract_gemini_usage_from_response(response)
        assert result["web_search_count"] == 1


# =======================
# extract_gemini_usage_from_chunk
# =======================


class TestExtractGeminiUsageFromChunk:
    def test_no_usage_metadata(self):
        chunk = MockObj(candidates=None)
        result = extract_gemini_usage_from_chunk(chunk)
        assert result == {}

    def test_none_usage_metadata(self):
        chunk = MockObj(usage_metadata=None, candidates=None)
        result = extract_gemini_usage_from_chunk(chunk)
        assert result == {}

    def test_with_usage_metadata(self):
        metadata = MockObj(prompt_token_count=20, candidates_token_count=10)
        chunk = MockObj(usage_metadata=metadata, candidates=None)
        result = extract_gemini_usage_from_chunk(chunk)
        assert result["input_tokens"] == 20
        assert result["output_tokens"] == 10

    def test_with_web_search_in_chunk(self):
        metadata = MockObj(prompt_token_count=20, candidates_token_count=10)
        grounding_metadata = MockObj(web_search_queries=["q"])
        candidate = MockObj(grounding_metadata=grounding_metadata, content=None)
        chunk = MockObj(usage_metadata=metadata, candidates=[candidate])
        result = extract_gemini_usage_from_chunk(chunk)
        assert result["web_search_count"] == 1


# =======================
# extract_gemini_content_from_chunk
# =======================


class TestExtractGeminiContentFromChunk:
    def test_text_content(self):
        chunk = MockObj(text="Streaming text", candidates=None)
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {"type": "text", "text": "Streaming text"}

    def test_empty_text(self):
        chunk = MockObj(text="", candidates=None)
        result = extract_gemini_content_from_chunk(chunk)
        assert result is None

    def test_function_call_in_candidate(self):
        func_call = MockObj(name="tool_fn", args={"k": "v"})
        part = MockObj(function_call=func_call, text=None)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        chunk = MockObj(text=None, candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {
            "type": "function",
            "function": {"name": "tool_fn", "arguments": {"k": "v"}},
        }

    def test_text_in_candidate_parts(self):
        part = MockObj(text="Part text", function_call=None)
        content = MockObj(parts=[part])
        candidate = MockObj(content=content)
        chunk = MockObj(text=None, candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result == {"type": "text", "text": "Part text"}

    def test_no_content(self):
        chunk = MockObj(text=None, candidates=None)
        result = extract_gemini_content_from_chunk(chunk)
        assert result is None

    def test_candidate_no_content(self):
        candidate = MockObj(content=None)
        chunk = MockObj(text=None, candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result is None

    def test_candidate_content_no_parts(self):
        content = MockObj(parts=None)
        candidate = MockObj(content=content)
        chunk = MockObj(text=None, candidates=[candidate])
        result = extract_gemini_content_from_chunk(chunk)
        assert result is None


# =======================
# format_gemini_streaming_output
# =======================


class TestFormatGeminiStreamingOutput:
    def test_string_input(self):
        result = format_gemini_streaming_output("Hello streaming")
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello streaming"}],
            }
        ]

    def test_list_of_strings(self):
        result = format_gemini_streaming_output(["Hello", " World"])
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello World"}]}
        ]

    def test_list_of_text_blocks(self):
        result = format_gemini_streaming_output(
            [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
        )
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "AB"}]}
        ]

    def test_list_with_function_call(self):
        content = [
            {"type": "text", "text": "Before"},
            {"type": "function", "function": {"name": "fn", "arguments": {}}},
            {"type": "text", "text": "After"},
        ]
        result = format_gemini_streaming_output(content)
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0] == {"type": "text", "text": "Before"}
        assert result[0]["content"][1] == {
            "type": "function",
            "function": {"name": "fn", "arguments": {}},
        }
        assert result[0]["content"][2] == {"type": "text", "text": "After"}

    def test_empty_list(self):
        result = format_gemini_streaming_output([])
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]

    def test_list_empty_strings(self):
        result = format_gemini_streaming_output(["", ""])
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]

    def test_non_list_non_string(self):
        """Fallback for unexpected input type."""
        result = format_gemini_streaming_output(42)
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]


# =======================
# extract_gemini_embedding_token_count
# =======================


class TestExtractGeminiEmbeddingTokenCount:
    def test_with_token_counts(self):
        stats1 = MockObj(token_count=10)
        stats2 = MockObj(token_count=15)
        emb1 = MockObj(statistics=stats1)
        emb2 = MockObj(statistics=stats2)
        response = MockObj(embeddings=[emb1, emb2])
        assert extract_gemini_embedding_token_count(response) == 25

    def test_no_embeddings(self):
        response = MockObj(embeddings=None)
        assert extract_gemini_embedding_token_count(response) == 0

    def test_empty_embeddings(self):
        response = MockObj(embeddings=[])
        assert extract_gemini_embedding_token_count(response) == 0

    def test_no_statistics(self):
        emb = MockObj(statistics=None)
        response = MockObj(embeddings=[emb])
        assert extract_gemini_embedding_token_count(response) == 0

    def test_no_token_count(self):
        stats = MockObj()
        emb = MockObj(statistics=stats)
        response = MockObj(embeddings=[emb])
        assert extract_gemini_embedding_token_count(response) == 0
