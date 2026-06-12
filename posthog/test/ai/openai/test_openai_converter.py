"""
Tests for posthog.ai.openai.openai_converter module.

Covers: format_openai_response, format_openai_input, extract_openai_tools,
format_openai_streaming_content, extract_openai_web_search_count,
extract_openai_stop_reason, extract_openai_usage_from_response,
extract_openai_usage_from_chunk, extract_openai_content_from_chunk,
extract_openai_tool_calls_from_chunk, accumulate_openai_tool_calls,
format_openai_streaming_output, format_openai_streaming_input.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from posthog.ai.openai.openai_converter import (
    accumulate_openai_tool_calls,
    extract_openai_content_from_chunk,
    extract_openai_stop_reason,
    extract_openai_tool_calls_from_chunk,
    extract_openai_tools,
    extract_openai_usage_from_chunk,
    extract_openai_usage_from_response,
    extract_openai_web_search_count,
    format_openai_input,
    format_openai_response,
    format_openai_streaming_content,
    format_openai_streaming_input,
    format_openai_streaming_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _make_chat_response(
    content="Hello",
    role="assistant",
    tool_calls=None,
    audio=None,
    usage=None,
    finish_reason="stop",
):
    message = _ns(content=content, role=role, tool_calls=tool_calls)
    if audio is not None:
        message.audio = audio
    else:
        # Ensure hasattr check is False
        pass
    choice = _ns(message=message, finish_reason=finish_reason, index=0)
    resp = _ns(choices=[choice])
    if usage is not None:
        resp.usage = usage
    return resp


def _make_responses_api_response(output_items=None, usage=None, status="completed"):
    resp = _ns(output=output_items or [], status=status)
    if usage is not None:
        resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# format_openai_response  -- Chat Completions
# ---------------------------------------------------------------------------

class TestFormatOpenAIResponseChatCompletions:
    def test_none_response(self):
        assert format_openai_response(None) == []

    def test_simple_text(self):
        resp = _make_chat_response(content="hi")
        result = format_openai_response(resp)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "hi"}]

    def test_no_content_returns_empty(self):
        resp = _make_chat_response(content=None)
        assert format_openai_response(resp) == []

    def test_custom_role(self):
        resp = _make_chat_response(content="ok", role="system")
        result = format_openai_response(resp)
        assert result[0]["role"] == "system"

    def test_tool_calls(self):
        tc = _ns(
            id="tc_1",
            function=_ns(name="get_weather", arguments='{"city":"NY"}'),
        )
        resp = _make_chat_response(content=None, tool_calls=[tc])
        result = format_openai_response(resp)
        assert len(result) == 1
        fc = result[0]["content"][0]
        assert fc["type"] == "function"
        assert fc["id"] == "tc_1"
        assert fc["function"]["name"] == "get_weather"

    def test_text_and_tool_calls(self):
        tc = _ns(id="tc_2", function=_ns(name="fn", arguments="{}"))
        resp = _make_chat_response(content="thinking", tool_calls=[tc])
        result = format_openai_response(resp)
        assert len(result[0]["content"]) == 2

    def test_audio_output(self):
        audio_model = _ns(id="audio_1", data="base64data", transcript="hello")
        audio_model.model_dump = lambda: {"id": "audio_1", "data": "base64data", "transcript": "hello"}
        resp = _make_chat_response(content=None)
        resp.choices[0].message.audio = audio_model
        result = format_openai_response(resp)
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "audio"
        assert result[0]["content"][0]["transcript"] == "hello"

    def test_multiple_choices(self):
        m1 = _ns(content="a", role="assistant", tool_calls=None)
        m2 = _ns(content="b", role="assistant", tool_calls=None)
        resp = _ns(choices=[_ns(message=m1, index=0), _ns(message=m2, index=1)])
        result = format_openai_response(resp)
        # Both choices' content merged into one message
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_empty_choices(self):
        resp = _ns(choices=[])
        assert format_openai_response(resp) == []

    def test_message_with_none_role(self):
        resp = _make_chat_response(content="hi", role=None)
        result = format_openai_response(resp)
        # role stays "assistant" (fallback at line 43)
        assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# format_openai_response  -- Responses API
# ---------------------------------------------------------------------------

class TestFormatOpenAIResponseResponsesAPI:
    def test_output_text(self):
        content_item = _ns(type="output_text", text="world")
        msg = _ns(type="message", role="assistant", content=[content_item])
        resp = _make_responses_api_response([msg])
        result = format_openai_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "world"}]

    def test_text_fallback(self):
        content_item = _ns(type="other", text="fallback")
        msg = _ns(type="message", role="assistant", content=[content_item])
        resp = _make_responses_api_response([msg])
        result = format_openai_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "fallback"}]

    def test_input_image(self):
        content_item = _ns(type="input_image", image_url="https://img.png")
        msg = _ns(type="message", role="assistant", content=[content_item])
        resp = _make_responses_api_response([msg])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["type"] == "image"
        assert result[0]["content"][0]["image"] == "https://img.png"

    def test_function_call_output(self):
        fc = _ns(type="function_call", call_id="fc_1", name="search", arguments='{"q":"x"}')
        resp = _make_responses_api_response([fc])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["type"] == "function"
        assert result[0]["content"][0]["id"] == "fc_1"

    def test_function_call_id_fallback(self):
        fc = _ns(type="function_call", id="fc_2", name="search", arguments="{}")
        # No call_id attr
        resp = _make_responses_api_response([fc])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["id"] == "fc_2"

    def test_non_list_content(self):
        msg = _ns(type="message", role="user", content="plain string")
        resp = _make_responses_api_response([msg])
        result = format_openai_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "plain string"}]

    def test_empty_output(self):
        resp = _make_responses_api_response([])
        assert format_openai_response(resp) == []


# ---------------------------------------------------------------------------
# format_openai_input
# ---------------------------------------------------------------------------

class TestFormatOpenAIInput:
    def test_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        result = format_openai_input(messages=msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Hi"

    def test_missing_role_defaults_user(self):
        result = format_openai_input(messages=[{"content": "test"}])
        assert result[0]["role"] == "user"

    def test_input_data_string(self):
        result = format_openai_input(input_data="hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_input_data_list_of_dicts(self):
        result = format_openai_input(
            input_data=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        )
        assert len(result) == 2
        assert result[1]["role"] == "assistant"

    def test_input_data_list_of_strings(self):
        result = format_openai_input(input_data=["first", "second"])
        assert result[0]["content"] == "first"
        assert result[1]["content"] == "second"

    def test_input_data_list_of_objects(self):
        result = format_openai_input(input_data=[42])
        assert result[0]["content"] == "42"

    def test_input_data_non_string_non_list(self):
        result = format_openai_input(input_data=42)
        assert result == [{"role": "user", "content": "42"}]

    def test_both_messages_and_input_data(self):
        result = format_openai_input(
            messages=[{"role": "user", "content": "a"}],
            input_data="b",
        )
        assert len(result) == 2

    def test_neither_messages_nor_input_data(self):
        assert format_openai_input() == []


# ---------------------------------------------------------------------------
# extract_openai_tools
# ---------------------------------------------------------------------------

class TestExtractOpenAITools:
    def test_tools_key(self):
        assert extract_openai_tools({"tools": [{"type": "function"}]}) == [{"type": "function"}]

    def test_functions_key(self):
        assert extract_openai_tools({"functions": [{"name": "f"}]}) == [{"name": "f"}]

    def test_tools_takes_precedence(self):
        result = extract_openai_tools({"tools": ["t"], "functions": ["f"]})
        assert result == ["t"]

    def test_no_tools(self):
        assert extract_openai_tools({"model": "gpt-4"}) is None


# ---------------------------------------------------------------------------
# format_openai_streaming_content
# ---------------------------------------------------------------------------

class TestFormatOpenAIStreamingContent:
    def test_text_only(self):
        result = format_openai_streaming_content("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_empty_text(self):
        assert format_openai_streaming_content("") == []

    def test_with_tool_calls(self):
        tc = [{"id": "tc_1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_content("thinking", tc)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "function"

    def test_tool_calls_only(self):
        tc = [{"id": "tc_1", "function": {"name": "fn"}}]
        result = format_openai_streaming_content("", tc)
        assert len(result) == 1
        assert result[0]["type"] == "function"

    def test_empty(self):
        assert format_openai_streaming_content("", None) == []


# ---------------------------------------------------------------------------
# extract_openai_web_search_count
# ---------------------------------------------------------------------------

class TestExtractOpenAIWebSearchCount:
    def test_no_indicators(self):
        resp = _ns(choices=[])
        assert extract_openai_web_search_count(resp) == 0

    def test_responses_api_web_search_calls(self):
        items = [
            _ns(type="web_search_call"),
            _ns(type="message", role="assistant", content=[]),
            _ns(type="web_search_call"),
        ]
        resp = _ns(output=items)
        assert extract_openai_web_search_count(resp) == 2

    def test_perplexity_citations(self):
        resp = _ns(citations=["https://example.com"])
        assert extract_openai_web_search_count(resp) == 1

    def test_perplexity_search_results(self):
        resp = _ns(search_results=[{"title": "x"}])
        assert extract_openai_web_search_count(resp) == 1

    def test_perplexity_search_context_size(self):
        resp = _ns(usage=_ns(search_context_size=100))
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_in_message_annotations_dict(self):
        annotation = {"type": "url_citation", "url": "https://example.com"}
        message = _ns(annotations=[annotation])
        choice = _ns(message=message)
        resp = _ns(choices=[choice])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_in_message_annotations_object(self):
        annotation = _ns(type="url_citation", url="https://example.com")
        message = _ns(annotations=[annotation])
        choice = _ns(message=message)
        resp = _ns(choices=[choice])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_in_delta_annotations(self):
        annotation = _ns(type="url_citation")
        delta = _ns(annotations=[annotation])
        choice = _ns(delta=delta)
        # No message attr
        resp = _ns(choices=[choice])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_in_output_content_annotations(self):
        annotation = _ns(type="url_citation")
        content_item = _ns(annotations=[annotation])
        msg = _ns(type="message", content=[content_item])
        resp = _ns(output=[msg])
        assert extract_openai_web_search_count(resp) == 1

    def test_empty_citations(self):
        resp = _ns(citations=[])
        assert extract_openai_web_search_count(resp) == 0

    def test_search_context_size_none(self):
        resp = _ns(usage=_ns(search_context_size=None))
        assert extract_openai_web_search_count(resp) == 0

    def test_none_annotations(self):
        message = _ns(annotations=None)
        choice = _ns(message=message)
        resp = _ns(choices=[choice])
        assert extract_openai_web_search_count(resp) == 0


# ---------------------------------------------------------------------------
# extract_openai_stop_reason
# ---------------------------------------------------------------------------

class TestExtractOpenAIStopReason:
    def test_chat_completions(self):
        resp = _make_chat_response(finish_reason="stop")
        assert extract_openai_stop_reason(resp) == "stop"

    def test_responses_api(self):
        resp = _ns(status="completed")
        assert extract_openai_stop_reason(resp) == "completed"

    def test_none(self):
        assert extract_openai_stop_reason(_ns()) is None


# ---------------------------------------------------------------------------
# extract_openai_usage_from_response
# ---------------------------------------------------------------------------

class TestExtractOpenAIUsageFromResponse:
    def test_no_usage(self):
        resp = _ns()
        result = extract_openai_usage_from_response(resp)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_chat_completions_usage(self):
        usage = _ns(
            prompt_tokens=100,
            completion_tokens=50,
            input_tokens=100,
            output_tokens=50,
        )
        # Add details attrs
        usage.prompt_tokens_details = _ns(cached_tokens=10)
        usage.completion_tokens_details = _ns(reasoning_tokens=5)
        usage.input_tokens_details = _ns(cached_tokens=10)
        usage.output_tokens_details = _ns(reasoning_tokens=5)
        usage.model_dump = lambda: {"prompt_tokens": 100, "completion_tokens": 50}
        resp = _ns(usage=usage, choices=[])
        result = extract_openai_usage_from_response(resp)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_read_input_tokens"] == 10
        assert result["reasoning_tokens"] == 5
        assert "raw_usage" in result

    def test_responses_api_usage(self):
        usage = _ns(
            input_tokens=200,
            output_tokens=75,
        )
        usage.input_tokens_details = _ns(cached_tokens=0)
        usage.output_tokens_details = _ns(reasoning_tokens=0)
        usage.model_dump = lambda: {"input_tokens": 200, "output_tokens": 75}
        resp = _ns(usage=usage)
        result = extract_openai_usage_from_response(resp)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 75

    def test_web_search_count_in_usage(self):
        usage = _ns(input_tokens=10, output_tokens=5)
        usage.model_dump = lambda: {}
        resp = _ns(usage=usage, citations=["url"])
        result = extract_openai_usage_from_response(resp)
        assert result["web_search_count"] == 1


# ---------------------------------------------------------------------------
# extract_openai_usage_from_chunk
# ---------------------------------------------------------------------------

class TestExtractOpenAIUsageFromChunk:
    def test_chat_no_usage(self):
        chunk = _ns(choices=[])
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result == {}

    def test_chat_with_usage(self):
        usage = _ns(prompt_tokens=20, completion_tokens=10)
        chunk = _ns(usage=usage, choices=[])
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result["input_tokens"] == 20
        assert result["output_tokens"] == 10

    def test_chat_with_cached_and_reasoning(self):
        usage = _ns(
            prompt_tokens=20,
            completion_tokens=10,
            prompt_tokens_details=_ns(cached_tokens=5),
            completion_tokens_details=_ns(reasoning_tokens=3),
        )
        usage.model_dump = lambda: {}
        chunk = _ns(usage=usage, choices=[])
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result["cache_read_input_tokens"] == 5
        assert result["reasoning_tokens"] == 3

    def test_responses_completed_event(self):
        response_usage = _ns(input_tokens=50, output_tokens=25)
        response_usage.model_dump = lambda: {}
        response = _ns(usage=response_usage, output=[])
        chunk = _ns(type="response.completed", response=response)
        result = extract_openai_usage_from_chunk(chunk, "responses")
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 25

    def test_responses_non_completed_event(self):
        chunk = _ns(type="response.output_item.added")
        result = extract_openai_usage_from_chunk(chunk, "responses")
        assert result == {}

    def test_chat_web_search_in_chunk(self):
        chunk = _ns(choices=[], citations=["url"])
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result["web_search_count"] == 1


# ---------------------------------------------------------------------------
# extract_openai_content_from_chunk
# ---------------------------------------------------------------------------

class TestExtractOpenAIContentFromChunk:
    def test_chat_delta_content(self):
        delta = _ns(content="hello")
        choice = _ns(delta=delta)
        chunk = _ns(choices=[choice])
        assert extract_openai_content_from_chunk(chunk, "chat") == "hello"

    def test_chat_no_content(self):
        delta = _ns(content=None)
        choice = _ns(delta=delta)
        chunk = _ns(choices=[choice])
        assert extract_openai_content_from_chunk(chunk, "chat") is None

    def test_chat_empty_choices(self):
        chunk = _ns(choices=[])
        assert extract_openai_content_from_chunk(chunk, "chat") is None

    def test_responses_completed(self):
        output_item = _ns(type="message")
        response = _ns(output=[output_item])
        chunk = _ns(type="response.completed", response=response)
        assert extract_openai_content_from_chunk(chunk, "responses") == output_item

    def test_responses_non_completed(self):
        chunk = _ns(type="response.in_progress")
        assert extract_openai_content_from_chunk(chunk, "responses") is None


# ---------------------------------------------------------------------------
# extract_openai_tool_calls_from_chunk
# ---------------------------------------------------------------------------

class TestExtractOpenAIToolCallsFromChunk:
    def test_with_tool_calls(self):
        fn = _ns(name="fn", arguments='{"a":1}')
        tc = _ns(index=0, id="tc_1", type="function", function=fn)
        delta = _ns(tool_calls=[tc])
        choice = _ns(delta=delta)
        chunk = _ns(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "fn"

    def test_no_tool_calls(self):
        delta = _ns(content="text", tool_calls=None)
        choice = _ns(delta=delta)
        chunk = _ns(choices=[choice])
        assert extract_openai_tool_calls_from_chunk(chunk) is None

    def test_partial_tool_call(self):
        fn = _ns(name=None, arguments='{"partial')
        tc = _ns(index=0, id=None, type=None, function=fn)
        delta = _ns(tool_calls=[tc])
        choice = _ns(delta=delta)
        chunk = _ns(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert len(result) == 1
        assert "id" not in result[0]
        assert result[0]["function"]["arguments"] == '{"partial'


# ---------------------------------------------------------------------------
# accumulate_openai_tool_calls
# ---------------------------------------------------------------------------

class TestAccumulateOpenAIToolCalls:
    def test_initial_accumulation(self):
        acc: Dict[int, Dict[str, Any]] = {}
        chunk_tcs = [
            {"index": 0, "id": "tc_1", "type": "function", "function": {"name": "fn", "arguments": '{"a":'}}
        ]
        accumulate_openai_tool_calls(acc, chunk_tcs)
        assert acc[0]["id"] == "tc_1"
        assert acc[0]["function"]["name"] == "fn"
        assert acc[0]["function"]["arguments"] == '{"a":'

    def test_incremental_arguments(self):
        acc: Dict[int, Dict[str, Any]] = {
            0: {"id": "tc_1", "type": "function", "function": {"name": "fn", "arguments": '{"a":'}}
        }
        chunk_tcs = [{"index": 0, "function": {"arguments": "1}"}}]
        accumulate_openai_tool_calls(acc, chunk_tcs)
        assert acc[0]["function"]["arguments"] == '{"a":1}'

    def test_missing_index_skipped(self):
        acc: Dict[int, Dict[str, Any]] = {}
        accumulate_openai_tool_calls(acc, [{"index": None, "id": "tc"}])
        assert len(acc) == 0

    def test_multiple_tool_calls(self):
        acc: Dict[int, Dict[str, Any]] = {}
        accumulate_openai_tool_calls(acc, [
            {"index": 0, "id": "tc_0", "type": "function", "function": {"name": "f1", "arguments": ""}},
            {"index": 1, "id": "tc_1", "type": "function", "function": {"name": "f2", "arguments": ""}},
        ])
        assert len(acc) == 2
        assert acc[0]["function"]["name"] == "f1"
        assert acc[1]["function"]["name"] == "f2"


# ---------------------------------------------------------------------------
# format_openai_streaming_output
# ---------------------------------------------------------------------------

class TestFormatOpenAIStreamingOutput:
    def test_chat_text_only(self):
        result = format_openai_streaming_output("hello", "chat")
        assert result == [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]

    def test_chat_with_tool_calls(self):
        tcs = [{"id": "tc_1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_output("", "chat", tcs)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "function"

    def test_chat_text_and_tools(self):
        tcs = [{"id": "tc_1", "function": {"name": "fn"}}]
        result = format_openai_streaming_output("msg", "chat", tcs)
        assert len(result[0]["content"]) == 2

    def test_chat_empty(self):
        result = format_openai_streaming_output("", "chat")
        assert result == [{"role": "assistant", "content": []}]

    def test_chat_list_content(self):
        result = format_openai_streaming_output(["a", "b"], "chat")
        assert result[0]["content"][0]["text"] == "ab"

    def test_responses_list(self):
        items = [{"role": "assistant", "content": "data"}]
        result = format_openai_streaming_output(items, "responses")
        assert result == items

    def test_responses_string(self):
        result = format_openai_streaming_output("text", "responses")
        assert result[0]["content"][0]["text"] == "text"

    def test_responses_empty_list(self):
        result = format_openai_streaming_output([], "responses")
        # Fallback
        assert result[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# format_openai_streaming_input
# ---------------------------------------------------------------------------

class TestFormatOpenAIStreamingInput:
    def test_basic_call(self):
        kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4"}
        result = format_openai_streaming_input(kwargs)
        # Should return formatted input via merge_system_prompt
        assert result is not None
