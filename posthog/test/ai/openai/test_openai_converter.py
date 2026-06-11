"""Unit tests for posthog.ai.openai.openai_converter."""

import pytest

from posthog.ai.openai.openai_converter import (
    accumulate_openai_tool_calls,
    extract_openai_content_from_chunk,
    extract_openai_stop_reason,
    extract_openai_tools,
    extract_openai_usage_from_chunk,
    extract_openai_usage_from_response,
    extract_openai_web_search_count,
    extract_openai_tool_calls_from_chunk,
    format_openai_input,
    format_openai_response,
    format_openai_streaming_content,
    format_openai_streaming_input,
    format_openai_streaming_output,
)


# ---------------------------------------------------------------------------
# Helpers – lightweight mock objects
# ---------------------------------------------------------------------------

class _Obj:
    """Simple namespace that converts kwargs to attributes."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_chat_response(
    content="Hello",
    role="assistant",
    finish_reason="stop",
    tool_calls=None,
    audio=None,
    usage=None,
):
    """Return a mock Chat Completions response."""
    msg = _Obj(content=content, role=role, tool_calls=tool_calls, audio=audio)
    choice = _Obj(message=msg, finish_reason=finish_reason, delta=None)
    resp = _Obj(choices=[choice], usage=usage)
    return resp


def _make_responses_api_response(output_items=None, usage=None, status="completed"):
    """Return a mock Responses API response."""
    return _Obj(output=output_items or [], usage=usage, status=status)


# ===========================================================================
# format_openai_response – Chat Completions
# ===========================================================================

class TestFormatOpenAIResponseChatCompletions:
    def test_none_response(self):
        assert format_openai_response(None) == []

    def test_basic_text_response(self):
        resp = _make_chat_response(content="Hi there", role="assistant")
        result = format_openai_response(resp)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there"}]

    def test_custom_role(self):
        resp = _make_chat_response(role="system")
        result = format_openai_response(resp)
        assert result[0]["role"] == "system"

    def test_no_content(self):
        resp = _make_chat_response(content=None)
        result = format_openai_response(resp)
        assert result == []

    def test_tool_calls(self):
        tc = _Obj(
            id="call_1",
            function=_Obj(name="get_weather", arguments='{"city":"NYC"}'),
        )
        resp = _make_chat_response(content=None, tool_calls=[tc])
        result = format_openai_response(resp)
        assert len(result) == 1
        items = result[0]["content"]
        assert items[0]["type"] == "function"
        assert items[0]["id"] == "call_1"
        assert items[0]["function"]["name"] == "get_weather"

    def test_content_with_tool_calls(self):
        tc = _Obj(id="c1", function=_Obj(name="fn", arguments="{}"))
        resp = _make_chat_response(content="text", tool_calls=[tc])
        items = format_openai_response(resp)[0]["content"]
        assert len(items) == 2
        assert items[0]["type"] == "text"
        assert items[1]["type"] == "function"

    def test_audio_output(self):
        audio_model = _Obj(model_dump=lambda: {"id": "a1", "data": "base64"})
        resp = _make_chat_response(content=None, audio=audio_model)
        items = format_openai_response(resp)[0]["content"]
        assert items[0]["type"] == "audio"
        assert items[0]["id"] == "a1"

    def test_message_none(self):
        """choice.message is None → skip."""
        choice = _Obj(message=None, delta=None)
        resp = _Obj(choices=[choice])
        assert format_openai_response(resp) == []

    def test_empty_choices(self):
        resp = _Obj(choices=[])
        assert format_openai_response(resp) == []


# ===========================================================================
# format_openai_response – Responses API
# ===========================================================================

class TestFormatOpenAIResponseResponsesAPI:
    def test_output_text(self):
        text_item = _Obj(type="output_text", text="Response text")
        msg_item = _Obj(type="message", role="assistant", content=[text_item])
        resp = _make_responses_api_response(output_items=[msg_item])
        result = format_openai_response(resp)
        assert result[0]["content"] == [{"type": "text", "text": "Response text"}]

    def test_output_text_with_generic_text_attr(self):
        """Content item has .text but type is not 'output_text'."""
        item = _Obj(type="other", text="fallback")
        msg = _Obj(type="message", role="assistant", content=[item])
        resp = _make_responses_api_response(output_items=[msg])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["text"] == "fallback"

    def test_input_image_content(self):
        img = _Obj(type="input_image", image_url="https://img.png")
        msg = _Obj(type="message", role="assistant", content=[img])
        resp = _make_responses_api_response(output_items=[msg])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["type"] == "image"
        assert result[0]["content"][0]["image"] == "https://img.png"

    def test_content_not_list(self):
        """content is a plain string, not a list."""
        msg = _Obj(type="message", role="assistant", content="plain string")
        resp = _make_responses_api_response(output_items=[msg])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "plain string"

    def test_function_call_item(self):
        fn_item = _Obj(
            type="function_call",
            call_id="fc_1",
            name="do_thing",
            arguments='{"x": 1}',
        )
        resp = _make_responses_api_response(output_items=[fn_item])
        result = format_openai_response(resp)
        items = result[0]["content"]
        assert items[0]["type"] == "function"
        assert items[0]["id"] == "fc_1"
        assert items[0]["function"]["name"] == "do_thing"

    def test_function_call_fallback_id(self):
        """call_id not present, fall back to .id."""
        fn_item = _Obj(type="function_call", id="fallback_id", name="fn", arguments="{}")
        resp = _make_responses_api_response(output_items=[fn_item])
        result = format_openai_response(resp)
        assert result[0]["content"][0]["id"] == "fallback_id"


# ===========================================================================
# format_openai_input
# ===========================================================================

class TestFormatOpenAIInput:
    def test_messages(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        result = format_openai_input(messages=msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_messages_missing_role(self):
        result = format_openai_input(messages=[{"content": "x"}])
        assert result[0]["role"] == "user"

    def test_input_data_string(self):
        result = format_openai_input(input_data="hello")
        assert result == [{"role": "user", "content": "hello"}]

    def test_input_data_list_of_dicts(self):
        data = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        result = format_openai_input(input_data=data)
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_input_data_list_of_strings(self):
        result = format_openai_input(input_data=["a", "b"])
        assert result[0]["content"] == "a"
        assert result[1]["content"] == "b"

    def test_input_data_list_of_objects(self):
        result = format_openai_input(input_data=[42])
        assert result[0]["content"] == "42"

    def test_input_data_non_string_non_list(self):
        result = format_openai_input(input_data=123)
        assert result == [{"role": "user", "content": "123"}]

    def test_both_none(self):
        assert format_openai_input() == []


# ===========================================================================
# extract_openai_tools
# ===========================================================================

class TestExtractOpenAITools:
    def test_tools_key(self):
        assert extract_openai_tools({"tools": ["t1"]}) == ["t1"]

    def test_functions_key(self):
        assert extract_openai_tools({"functions": ["f1"]}) == ["f1"]

    def test_no_tools(self):
        assert extract_openai_tools({}) is None


# ===========================================================================
# format_openai_streaming_content
# ===========================================================================

class TestFormatOpenAIStreamingContent:
    def test_text_only(self):
        result = format_openai_streaming_content("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_empty_text(self):
        assert format_openai_streaming_content("") == []

    def test_tool_calls_only(self):
        tcs = [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_content("", tool_calls=tcs)
        assert len(result) == 1
        assert result[0]["type"] == "function"

    def test_text_and_tool_calls(self):
        tcs = [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_content("hi", tool_calls=tcs)
        assert len(result) == 2

    def test_none_tool_calls(self):
        result = format_openai_streaming_content("hi", tool_calls=None)
        assert len(result) == 1


# ===========================================================================
# extract_openai_web_search_count
# ===========================================================================

class TestExtractOpenAIWebSearchCount:
    def test_no_indicators(self):
        resp = _Obj()
        assert extract_openai_web_search_count(resp) == 0

    def test_responses_api_web_search_call(self):
        items = [_Obj(type="web_search_call"), _Obj(type="web_search_call"), _Obj(type="message")]
        resp = _Obj(output=items)
        assert extract_openai_web_search_count(resp) == 2

    def test_perplexity_citations(self):
        resp = _Obj(citations=["http://example.com"])
        assert extract_openai_web_search_count(resp) == 1

    def test_perplexity_empty_citations(self):
        resp = _Obj(citations=[])
        assert extract_openai_web_search_count(resp) == 0

    def test_search_results(self):
        resp = _Obj(search_results=[{"url": "http://example.com"}])
        assert extract_openai_web_search_count(resp) == 1

    def test_usage_search_context_size(self):
        resp = _Obj(usage=_Obj(search_context_size=1024))
        assert extract_openai_web_search_count(resp) == 1

    def test_usage_search_context_size_falsy(self):
        resp = _Obj(usage=_Obj(search_context_size=0))
        assert extract_openai_web_search_count(resp) == 0

    def test_url_citation_in_message_annotations(self):
        annotation = _Obj(type="url_citation")
        msg = _Obj(annotations=[annotation])
        choice = _Obj(message=msg)
        resp = _Obj(choices=[choice])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_as_dict(self):
        annotation = {"type": "url_citation"}
        msg = _Obj(annotations=[annotation])
        choice = _Obj(message=msg)
        resp = _Obj(choices=[choice])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_in_delta_annotations(self):
        annotation = _Obj(type="url_citation")
        delta = _Obj(annotations=[annotation])
        choice = _Obj(message=_Obj(annotations=None), delta=delta)
        resp = _Obj(choices=[choice])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_in_responses_api_output(self):
        annotation = _Obj(type="url_citation")
        content_item = _Obj(annotations=[annotation])
        msg = _Obj(type="message", content=[content_item])
        resp = _Obj(output=[msg])
        assert extract_openai_web_search_count(resp) == 1

    def test_url_citation_dict_in_output(self):
        content_item = _Obj(annotations=[{"type": "url_citation"}])
        msg = _Obj(type="message", content=[content_item])
        resp = _Obj(output=[msg])
        assert extract_openai_web_search_count(resp) == 1


# ===========================================================================
# extract_openai_stop_reason
# ===========================================================================

class TestExtractOpenAIStopReason:
    def test_chat_completions(self):
        resp = _make_chat_response(finish_reason="stop")
        assert extract_openai_stop_reason(resp) == "stop"

    def test_responses_api(self):
        resp = _Obj(status="completed")
        assert extract_openai_stop_reason(resp) == "completed"

    def test_none(self):
        assert extract_openai_stop_reason(_Obj()) is None


# ===========================================================================
# extract_openai_usage_from_response
# ===========================================================================

class TestExtractOpenAIUsageFromResponse:
    def test_no_usage(self):
        resp = _Obj()
        result = extract_openai_usage_from_response(resp)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_responses_api_usage(self):
        usage = _Obj(
            input_tokens=100,
            output_tokens=50,
            input_tokens_details=_Obj(cached_tokens=10),
            output_tokens_details=_Obj(reasoning_tokens=5),
        )
        resp = _Obj(usage=usage)
        result = extract_openai_usage_from_response(resp)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_read_input_tokens"] == 10
        assert result["reasoning_tokens"] == 5

    def test_chat_completions_usage_overrides(self):
        """Chat Completions fields (prompt_tokens) override Responses API fields."""
        usage = _Obj(
            input_tokens=100,
            output_tokens=50,
            prompt_tokens=200,
            completion_tokens=80,
            prompt_tokens_details=_Obj(cached_tokens=20),
            completion_tokens_details=_Obj(reasoning_tokens=10),
            input_tokens_details=_Obj(cached_tokens=10),
            output_tokens_details=_Obj(reasoning_tokens=5),
        )
        resp = _Obj(usage=usage)
        result = extract_openai_usage_from_response(resp)
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 80
        assert result["cache_read_input_tokens"] == 20
        assert result["reasoning_tokens"] == 10

    def test_zero_cached_tokens_not_included(self):
        usage = _Obj(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=_Obj(cached_tokens=0),
            completion_tokens_details=_Obj(reasoning_tokens=0),
            input_tokens=100,
            output_tokens=50,
            input_tokens_details=_Obj(cached_tokens=0),
            output_tokens_details=_Obj(reasoning_tokens=0),
        )
        resp = _Obj(usage=usage)
        result = extract_openai_usage_from_response(resp)
        assert "cache_read_input_tokens" not in result
        assert "reasoning_tokens" not in result

    def test_raw_usage_serialized(self):
        usage = _Obj(
            input_tokens=10,
            output_tokens=5,
            model_dump=lambda: {"input_tokens": 10, "output_tokens": 5},
        )
        resp = _Obj(usage=usage)
        result = extract_openai_usage_from_response(resp)
        assert result["raw_usage"] == {"input_tokens": 10, "output_tokens": 5}


# ===========================================================================
# extract_openai_usage_from_chunk
# ===========================================================================

class TestExtractOpenAIUsageFromChunk:
    def test_chat_no_usage(self):
        chunk = _Obj()
        result = extract_openai_usage_from_chunk(chunk, provider_type="chat")
        assert result.get("input_tokens") is None

    def test_chat_with_usage(self):
        chunk = _Obj(
            usage=_Obj(
                prompt_tokens=50,
                completion_tokens=25,
            ),
        )
        result = extract_openai_usage_from_chunk(chunk, provider_type="chat")
        assert result["input_tokens"] == 50
        assert result["output_tokens"] == 25

    def test_chat_with_cached_and_reasoning(self):
        chunk = _Obj(
            usage=_Obj(
                prompt_tokens=50,
                completion_tokens=25,
                prompt_tokens_details=_Obj(cached_tokens=10),
                completion_tokens_details=_Obj(reasoning_tokens=5),
            ),
        )
        result = extract_openai_usage_from_chunk(chunk, provider_type="chat")
        assert result["cache_read_input_tokens"] == 10
        assert result["reasoning_tokens"] == 5

    def test_chat_cached_none(self):
        chunk = _Obj(
            usage=_Obj(
                prompt_tokens=50,
                completion_tokens=25,
                prompt_tokens_details=_Obj(cached_tokens=None),
                completion_tokens_details=_Obj(reasoning_tokens=None),
            ),
        )
        result = extract_openai_usage_from_chunk(chunk, provider_type="chat")
        assert "cache_read_input_tokens" not in result
        assert "reasoning_tokens" not in result

    def test_chat_web_search_detected(self):
        chunk = _Obj(citations=["http://example.com"])
        result = extract_openai_usage_from_chunk(chunk, provider_type="chat")
        assert result["web_search_count"] == 1

    def test_responses_completed(self):
        response_usage = _Obj(
            input_tokens=100,
            output_tokens=50,
        )
        resp = _Obj(usage=response_usage, output=[])
        chunk = _Obj(type="response.completed", response=resp)
        result = extract_openai_usage_from_chunk(chunk, provider_type="responses")
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_responses_completed_with_details(self):
        response_usage = _Obj(
            input_tokens=100,
            output_tokens=50,
            input_tokens_details=_Obj(cached_tokens=15),
            output_tokens_details=_Obj(reasoning_tokens=8),
        )
        resp = _Obj(usage=response_usage, output=[])
        chunk = _Obj(type="response.completed", response=resp)
        result = extract_openai_usage_from_chunk(chunk, provider_type="responses")
        assert result["cache_read_input_tokens"] == 15
        assert result["reasoning_tokens"] == 8

    def test_responses_non_completed_event(self):
        chunk = _Obj(type="response.in_progress")
        result = extract_openai_usage_from_chunk(chunk, provider_type="responses")
        assert result.get("input_tokens") is None

    def test_responses_web_search_in_response(self):
        ws = _Obj(type="web_search_call")
        response_usage = _Obj(input_tokens=10, output_tokens=5)
        resp = _Obj(usage=response_usage, output=[ws])
        chunk = _Obj(type="response.completed", response=resp)
        result = extract_openai_usage_from_chunk(chunk, provider_type="responses")
        assert result["web_search_count"] == 1


# ===========================================================================
# extract_openai_content_from_chunk
# ===========================================================================

class TestExtractOpenAIContentFromChunk:
    def test_chat_content(self):
        delta = _Obj(content="hello")
        choice = _Obj(delta=delta)
        chunk = _Obj(choices=[choice])
        assert extract_openai_content_from_chunk(chunk, "chat") == "hello"

    def test_chat_no_content(self):
        chunk = _Obj(choices=[])
        assert extract_openai_content_from_chunk(chunk, "chat") is None

    def test_chat_delta_none_content(self):
        delta = _Obj(content=None)
        choice = _Obj(delta=delta)
        chunk = _Obj(choices=[choice])
        assert extract_openai_content_from_chunk(chunk, "chat") is None

    def test_responses_completed(self):
        output_item = _Obj(type="message")
        resp = _Obj(output=[output_item])
        chunk = _Obj(type="response.completed", response=resp)
        result = extract_openai_content_from_chunk(chunk, "responses")
        assert result is not None

    def test_responses_non_completed(self):
        chunk = _Obj(type="response.in_progress")
        assert extract_openai_content_from_chunk(chunk, "responses") is None


# ===========================================================================
# extract_openai_tool_calls_from_chunk
# ===========================================================================

class TestExtractOpenAIToolCallsFromChunk:
    def test_no_tool_calls(self):
        chunk = _Obj(choices=[])
        assert extract_openai_tool_calls_from_chunk(chunk) is None

    def test_with_tool_calls(self):
        fn = _Obj(name="fn", arguments='{"a":1}')
        tc = _Obj(index=0, id="c1", type="function", function=fn)
        delta = _Obj(tool_calls=[tc], content=None)
        choice = _Obj(delta=delta)
        chunk = _Obj(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert result[0]["id"] == "c1"
        assert result[0]["function"]["name"] == "fn"

    def test_tool_call_no_id(self):
        fn = _Obj(name="fn", arguments=None)
        tc = _Obj(index=0, id=None, type=None, function=fn)
        delta = _Obj(tool_calls=[tc], content=None)
        choice = _Obj(delta=delta)
        chunk = _Obj(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert "id" not in result[0]
        assert "type" not in result[0]

    def test_tool_call_no_function(self):
        tc = _Obj(index=0, id="c1", type="function", function=None)
        delta = _Obj(tool_calls=[tc], content=None)
        choice = _Obj(delta=delta)
        chunk = _Obj(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert "function" not in result[0]


# ===========================================================================
# accumulate_openai_tool_calls
# ===========================================================================

class TestAccumulateOpenAIToolCalls:
    def test_new_tool_call(self):
        acc = {}
        deltas = [{"index": 0, "id": "c1", "type": "function", "function": {"name": "fn", "arguments": '{"a":'}}]
        accumulate_openai_tool_calls(acc, deltas)
        assert acc[0]["id"] == "c1"
        assert acc[0]["function"]["name"] == "fn"
        assert acc[0]["function"]["arguments"] == '{"a":'

    def test_append_arguments(self):
        acc = {0: {"id": "c1", "type": "function", "function": {"name": "fn", "arguments": '{"a":'}}}
        deltas = [{"index": 0, "function": {"arguments": "1}"}}]
        accumulate_openai_tool_calls(acc, deltas)
        assert acc[0]["function"]["arguments"] == '{"a":1}'

    def test_skip_none_index(self):
        acc = {}
        accumulate_openai_tool_calls(acc, [{"index": None}])
        assert acc == {}


# ===========================================================================
# format_openai_streaming_output
# ===========================================================================

class TestFormatOpenAIStreamingOutput:
    def test_chat_text(self):
        result = format_openai_streaming_output("Hello world", provider_type="chat")
        assert result[0]["content"][0]["text"] == "Hello world"

    def test_chat_list_content(self):
        result = format_openai_streaming_output(["a", "b"], provider_type="chat")
        assert result[0]["content"][0]["text"] == "ab"

    def test_chat_empty(self):
        result = format_openai_streaming_output("", provider_type="chat")
        assert result[0]["content"] == []

    def test_chat_with_tool_calls(self):
        tcs = [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_output("", provider_type="chat", tool_calls=tcs)
        assert result[0]["content"][0]["type"] == "function"

    def test_responses_list(self):
        items = [{"role": "assistant", "content": "test"}]
        result = format_openai_streaming_output(items, provider_type="responses")
        assert result == items

    def test_responses_string(self):
        result = format_openai_streaming_output("text", provider_type="responses")
        assert result[0]["content"][0]["text"] == "text"

    def test_unknown_type_fallback(self):
        result = format_openai_streaming_output(42, provider_type="other")
        assert result[0]["content"][0]["text"] == "42"


# ===========================================================================
# format_openai_streaming_input
# ===========================================================================

class TestFormatOpenAIStreamingInput:
    def test_basic_call(self):
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        result = format_openai_streaming_input(kwargs)
        assert result is not None
