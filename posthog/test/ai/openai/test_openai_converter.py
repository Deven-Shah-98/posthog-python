"""
Unit tests for posthog.ai.openai.openai_converter module.

Tests conversion of OpenAI API responses/inputs into standardized PostHog formats.
Covers Chat Completions API, Responses API, streaming, tool calls, and edge cases.
"""

from posthog.ai.openai.openai_converter import (
    accumulate_openai_tool_calls,
    extract_openai_content_from_chunk,
    extract_openai_stop_reason,
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


# =======================
# Mock Helpers
# =======================


class MockObj:
    """Generic mock object that sets attributes from kwargs."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# =======================
# format_openai_response
# =======================


class TestFormatOpenAIResponse:
    def test_none_response(self):
        assert format_openai_response(None) == []

    def test_chat_completion_basic(self):
        msg = MockObj(role="assistant", content="Hello!", tool_calls=None, audio=None)
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        result = format_openai_response(response)
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}
        ]

    def test_chat_completion_no_content(self):
        msg = MockObj(role="assistant", content=None, tool_calls=None, audio=None)
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        result = format_openai_response(response)
        assert result == []

    def test_chat_completion_with_tool_calls(self):
        func = MockObj(name="get_weather", arguments='{"city": "NYC"}')
        tool_call = MockObj(id="call_123", function=func)
        msg = MockObj(
            role="assistant", content="Let me check", tool_calls=[tool_call], audio=None
        )
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        result = format_openai_response(response)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0] == {"type": "text", "text": "Let me check"}
        assert result[0]["content"][1] == {
            "type": "function",
            "id": "call_123",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        }

    def test_chat_completion_with_audio(self):
        audio_obj = MockObj()
        audio_obj.model_dump = lambda: {
            "id": "audio_1",
            "data": "base64data",
            "transcript": "Hi",
        }
        msg = MockObj(role="assistant", content=None, tool_calls=None, audio=audio_obj)
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        result = format_openai_response(response)
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "audio"
        assert result[0]["content"][0]["id"] == "audio_1"

    def test_responses_api_basic(self):
        text_content = MockObj(type="output_text", text="Hello from responses API")
        text_content.annotations = []
        output_msg = MockObj(type="message", role="assistant", content=[text_content])
        response = MockObj(output=[output_msg])
        # no choices attribute
        result = format_openai_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from responses API"}],
            }
        ]

    def test_responses_api_with_text_fallback(self):
        """Content item has text but not output_text type."""
        text_content = MockObj(type="other_text", text="Fallback text")
        output_msg = MockObj(type="message", role="assistant", content=[text_content])
        response = MockObj(output=[output_msg])
        result = format_openai_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Fallback text"}],
            }
        ]

    def test_responses_api_with_image_content(self):
        img_content = MockObj(
            type="input_image", image_url="https://example.com/img.png"
        )
        output_msg = MockObj(type="message", role="assistant", content=[img_content])
        response = MockObj(output=[output_msg])
        result = format_openai_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "image", "image": "https://example.com/img.png"}],
            }
        ]

    def test_responses_api_content_not_list(self):
        """When content is not a list but exists."""
        output_msg = MockObj(
            type="message", role="assistant", content="plain string content"
        )
        response = MockObj(output=[output_msg])
        result = format_openai_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "plain string content"}],
            }
        ]

    def test_responses_api_function_call(self):
        func_item = MockObj(
            type="function_call",
            call_id="fc_1",
            name="search",
            arguments='{"q": "test"}',
        )
        response = MockObj(output=[func_item])
        result = format_openai_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function",
                        "id": "fc_1",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]

    def test_responses_api_function_call_with_id_fallback(self):
        """Function call without call_id falls back to id."""
        func_item = MockObj(
            type="function_call", name="search", arguments="{}", id="id_fallback"
        )
        # Remove call_id so getattr falls back
        response = MockObj(output=[func_item])
        result = format_openai_response(response)
        assert result[0]["content"][0]["id"] == "id_fallback"

    def test_multiple_choices(self):
        msg1 = MockObj(role="assistant", content="First", tool_calls=None, audio=None)
        msg2 = MockObj(role="assistant", content="Second", tool_calls=None, audio=None)
        response = MockObj(choices=[MockObj(message=msg1), MockObj(message=msg2)])
        result = format_openai_response(response)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_choice_without_message(self):
        choice = MockObj(message=None)
        response = MockObj(choices=[choice])
        result = format_openai_response(response)
        assert result == []


# =======================
# format_openai_input
# =======================


class TestFormatOpenAIInput:
    def test_messages_format(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = format_openai_input(messages=messages)
        assert result == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_input_data_string(self):
        result = format_openai_input(input_data="Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_input_data_list_of_dicts(self):
        input_data = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
        ]
        result = format_openai_input(input_data=input_data)
        assert result == [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
        ]

    def test_input_data_list_of_strings(self):
        result = format_openai_input(input_data=["Hello", "World"])
        assert result == [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]

    def test_input_data_list_of_objects(self):
        """Non-dict, non-string items in input_data list are stringified."""
        result = format_openai_input(input_data=[42])
        assert result == [{"role": "user", "content": "42"}]

    def test_input_data_non_string_non_list(self):
        """Non-string, non-list input_data is stringified."""
        result = format_openai_input(input_data=123)
        assert result == [{"role": "user", "content": "123"}]

    def test_both_none(self):
        result = format_openai_input()
        assert result == []

    def test_messages_missing_keys(self):
        messages = [{}]
        result = format_openai_input(messages=messages)
        assert result == [{"role": "user", "content": ""}]


# =======================
# extract_openai_tools
# =======================


class TestExtractOpenAITools:
    def test_tools_present(self):
        kwargs = {"tools": [{"type": "function", "function": {"name": "fn"}}]}
        assert extract_openai_tools(kwargs) == kwargs["tools"]

    def test_functions_present(self):
        kwargs = {"functions": [{"name": "fn"}]}
        assert extract_openai_tools(kwargs) == kwargs["functions"]

    def test_neither_present(self):
        assert extract_openai_tools({}) is None

    def test_tools_takes_priority(self):
        kwargs = {"tools": [{"t": 1}], "functions": [{"f": 1}]}
        assert extract_openai_tools(kwargs) == [{"t": 1}]


# =======================
# format_openai_streaming_content
# =======================


class TestFormatOpenAIStreamingContent:
    def test_text_only(self):
        result = format_openai_streaming_content("Hello world")
        assert result == [{"type": "text", "text": "Hello world"}]

    def test_empty_content(self):
        result = format_openai_streaming_content("")
        assert result == []

    def test_with_tool_calls(self):
        tool_calls = [{"id": "tc_1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_content("", tool_calls=tool_calls)
        assert result == [
            {
                "type": "function",
                "id": "tc_1",
                "function": {"name": "fn", "arguments": "{}"},
            }
        ]

    def test_text_and_tool_calls(self):
        tool_calls = [{"id": "tc_1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_content("Thinking...", tool_calls=tool_calls)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Thinking..."}
        assert result[1]["type"] == "function"


# =======================
# extract_openai_web_search_count
# =======================


class TestExtractOpenAIWebSearchCount:
    def test_no_indicators(self):
        response = MockObj()
        assert extract_openai_web_search_count(response) == 0

    def test_responses_api_web_search_call(self):
        item1 = MockObj(type="web_search_call")
        item2 = MockObj(type="message", content=[])
        item3 = MockObj(type="web_search_call")
        response = MockObj(output=[item1, item2, item3])
        assert extract_openai_web_search_count(response) == 2

    def test_perplexity_citations(self):
        response = MockObj(citations=["http://example.com"])
        assert extract_openai_web_search_count(response) == 1

    def test_perplexity_empty_citations(self):
        response = MockObj(citations=[])
        assert extract_openai_web_search_count(response) == 0

    def test_perplexity_search_results(self):
        response = MockObj(search_results=[{"url": "http://example.com"}])
        assert extract_openai_web_search_count(response) == 1

    def test_search_context_size(self):
        usage = MockObj(search_context_size=1500)
        response = MockObj(usage=usage)
        assert extract_openai_web_search_count(response) == 1

    def test_search_context_size_zero(self):
        usage = MockObj(search_context_size=0)
        response = MockObj(usage=usage)
        assert extract_openai_web_search_count(response) == 0

    def test_url_citation_in_choices_message(self):
        annotation = MockObj(type="url_citation")
        msg = MockObj(annotations=[annotation])
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        assert extract_openai_web_search_count(response) == 1

    def test_url_citation_dict_annotation(self):
        annotation = {"type": "url_citation"}
        msg = MockObj(annotations=[annotation])
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        assert extract_openai_web_search_count(response) == 1

    def test_url_citation_in_delta(self):
        annotation = MockObj(type="url_citation")
        delta = MockObj(annotations=[annotation])
        choice = MockObj(delta=delta, message=MockObj(annotations=None))
        response = MockObj(choices=[choice])
        assert extract_openai_web_search_count(response) == 1

    def test_url_citation_dict_in_delta(self):
        annotation = {"type": "url_citation"}
        delta = MockObj(annotations=[annotation])
        choice = MockObj(delta=delta, message=MockObj(annotations=None))
        response = MockObj(choices=[choice])
        assert extract_openai_web_search_count(response) == 1

    def test_url_citation_in_responses_output(self):
        annotation = MockObj(type="url_citation")
        content_item = MockObj(annotations=[annotation])
        output_msg = MockObj(type="message", content=[content_item])
        response = MockObj(output=[output_msg])
        assert extract_openai_web_search_count(response) == 1

    def test_no_annotations(self):
        msg = MockObj(annotations=None)
        choice = MockObj(message=msg)
        response = MockObj(choices=[choice])
        assert extract_openai_web_search_count(response) == 0


# =======================
# extract_openai_stop_reason
# =======================


class TestExtractOpenAIStopReason:
    def test_chat_completions(self):
        choice = MockObj(finish_reason="stop")
        response = MockObj(choices=[choice])
        assert extract_openai_stop_reason(response) == "stop"

    def test_responses_api(self):
        response = MockObj(status="completed")
        assert extract_openai_stop_reason(response) == "completed"

    def test_neither(self):
        response = MockObj()
        assert extract_openai_stop_reason(response) is None

    def test_empty_choices(self):
        response = MockObj(choices=[])
        assert extract_openai_stop_reason(response) is None


# =======================
# extract_openai_usage_from_response
# =======================


class TestExtractOpenAIUsageFromResponse:
    def test_no_usage(self):
        response = MockObj()
        result = extract_openai_usage_from_response(response)
        assert result == {"input_tokens": 0, "output_tokens": 0}

    def test_chat_completions_format(self):
        usage = MockObj(
            prompt_tokens=100,
            completion_tokens=50,
            prompt_tokens_details=MockObj(cached_tokens=10),
            completion_tokens_details=MockObj(reasoning_tokens=5),
        )
        # Also add responses API fields to test priority
        usage.input_tokens = 0
        usage.output_tokens = 0
        usage.input_tokens_details = MockObj(cached_tokens=0)
        usage.output_tokens_details = MockObj(reasoning_tokens=0)
        response = MockObj(usage=usage)
        # Remove output attr to prevent web_search_count check
        result = extract_openai_usage_from_response(response)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_read_input_tokens"] == 10
        assert result["reasoning_tokens"] == 5

    def test_responses_api_format(self):
        usage = MockObj(
            input_tokens=80,
            output_tokens=40,
            input_tokens_details=MockObj(cached_tokens=5),
            output_tokens_details=MockObj(reasoning_tokens=3),
        )
        response = MockObj(usage=usage)
        result = extract_openai_usage_from_response(response)
        assert result["input_tokens"] == 80
        assert result["output_tokens"] == 40
        assert result["cache_read_input_tokens"] == 5
        assert result["reasoning_tokens"] == 3

    def test_no_cached_or_reasoning(self):
        usage = MockObj(
            input_tokens=10,
            output_tokens=5,
        )
        response = MockObj(usage=usage)
        result = extract_openai_usage_from_response(response)
        assert "cache_read_input_tokens" not in result
        assert "reasoning_tokens" not in result


# =======================
# extract_openai_usage_from_chunk
# =======================


class TestExtractOpenAIUsageFromChunk:
    def test_chat_no_usage(self):
        chunk = MockObj()
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result == {}

    def test_chat_with_usage(self):
        usage = MockObj(
            prompt_tokens=10,
            completion_tokens=5,
        )
        chunk = MockObj(usage=usage)
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5

    def test_chat_with_cached_tokens(self):
        usage = MockObj(
            prompt_tokens=20,
            completion_tokens=10,
            prompt_tokens_details=MockObj(cached_tokens=5),
            completion_tokens_details=MockObj(reasoning_tokens=3),
        )
        chunk = MockObj(usage=usage)
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result["cache_read_input_tokens"] == 5
        assert result["reasoning_tokens"] == 3

    def test_chat_cached_tokens_none(self):
        usage = MockObj(
            prompt_tokens=20,
            completion_tokens=10,
            prompt_tokens_details=MockObj(cached_tokens=None),
            completion_tokens_details=MockObj(reasoning_tokens=None),
        )
        chunk = MockObj(usage=usage)
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert "cache_read_input_tokens" not in result
        assert "reasoning_tokens" not in result

    def test_responses_completed(self):
        response_usage = MockObj(
            input_tokens=30,
            output_tokens=15,
            input_tokens_details=MockObj(cached_tokens=2),
            output_tokens_details=MockObj(reasoning_tokens=1),
        )
        inner_response = MockObj(usage=response_usage, output=[])
        chunk = MockObj(type="response.completed", response=inner_response)
        result = extract_openai_usage_from_chunk(chunk, "responses")
        assert result["input_tokens"] == 30
        assert result["output_tokens"] == 15
        assert result["cache_read_input_tokens"] == 2
        assert result["reasoning_tokens"] == 1

    def test_responses_non_completed(self):
        chunk = MockObj(type="response.output_text.delta")
        result = extract_openai_usage_from_chunk(chunk, "responses")
        assert result == {}

    def test_responses_with_web_search(self):
        ws_item = MockObj(type="web_search_call")
        response_usage = MockObj(input_tokens=10, output_tokens=5)
        inner_response = MockObj(usage=response_usage, output=[ws_item])
        chunk = MockObj(type="response.completed", response=inner_response)
        result = extract_openai_usage_from_chunk(chunk, "responses")
        assert result["web_search_count"] == 1

    def test_chat_web_search_from_chunk(self):
        """Web search indicators detected even without usage."""
        annotation = MockObj(type="url_citation")
        delta = MockObj(annotations=[annotation])
        choice = MockObj(delta=delta, message=MockObj(annotations=None))
        chunk = MockObj(choices=[choice])
        result = extract_openai_usage_from_chunk(chunk, "chat")
        assert result["web_search_count"] == 1


# =======================
# extract_openai_content_from_chunk
# =======================


class TestExtractOpenAIContentFromChunk:
    def test_chat_with_content(self):
        delta = MockObj(content="Hello")
        choice = MockObj(delta=delta)
        chunk = MockObj(choices=[choice])
        assert extract_openai_content_from_chunk(chunk, "chat") == "Hello"

    def test_chat_no_content(self):
        delta = MockObj(content=None)
        choice = MockObj(delta=delta)
        chunk = MockObj(choices=[choice])
        assert extract_openai_content_from_chunk(chunk, "chat") is None

    def test_chat_empty_choices(self):
        chunk = MockObj(choices=[])
        assert extract_openai_content_from_chunk(chunk, "chat") is None

    def test_responses_completed(self):
        output_item = MockObj(type="message")
        inner_response = MockObj(output=[output_item])
        chunk = MockObj(type="response.completed", response=inner_response)
        result = extract_openai_content_from_chunk(chunk, "responses")
        assert result is not None

    def test_responses_non_completed(self):
        chunk = MockObj(type="response.output_text.delta")
        assert extract_openai_content_from_chunk(chunk, "responses") is None

    def test_unknown_provider(self):
        chunk = MockObj()
        assert extract_openai_content_from_chunk(chunk, "unknown") is None


# =======================
# extract_openai_tool_calls_from_chunk
# =======================


class TestExtractOpenAIToolCallsFromChunk:
    def test_with_tool_calls(self):
        from posthog.ai.openai.openai_converter import (
            extract_openai_tool_calls_from_chunk,
        )

        func = MockObj(name="fn", arguments='{"a": 1}')
        tc = MockObj(index=0, id="tc_1", type="function", function=func)
        delta = MockObj(tool_calls=[tc])
        choice = MockObj(delta=delta)
        chunk = MockObj(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert result is not None
        assert result[0]["index"] == 0
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "fn"

    def test_no_tool_calls(self):
        from posthog.ai.openai.openai_converter import (
            extract_openai_tool_calls_from_chunk,
        )

        delta = MockObj(tool_calls=None)
        choice = MockObj(delta=delta)
        chunk = MockObj(choices=[choice])
        assert extract_openai_tool_calls_from_chunk(chunk) is None

    def test_tool_call_without_id(self):
        from posthog.ai.openai.openai_converter import (
            extract_openai_tool_calls_from_chunk,
        )

        func = MockObj(name=None, arguments='{"a": 1}')
        tc = MockObj(index=0, id=None, type=None, function=func)
        delta = MockObj(tool_calls=[tc])
        choice = MockObj(delta=delta)
        chunk = MockObj(choices=[choice])
        result = extract_openai_tool_calls_from_chunk(chunk)
        assert result is not None
        assert "id" not in result[0]


# =======================
# accumulate_openai_tool_calls
# =======================


class TestAccumulateOpenAIToolCalls:
    def test_first_chunk(self):
        accumulated = {}
        chunk_tool_calls = [
            {
                "index": 0,
                "id": "tc_1",
                "type": "function",
                "function": {"name": "fn", "arguments": '{"a"'},
            }
        ]
        accumulate_openai_tool_calls(accumulated, chunk_tool_calls)
        assert accumulated[0]["id"] == "tc_1"
        assert accumulated[0]["function"]["name"] == "fn"
        assert accumulated[0]["function"]["arguments"] == '{"a"'

    def test_subsequent_chunks(self):
        accumulated = {
            0: {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "fn", "arguments": '{"a"'},
            }
        }
        chunk_tool_calls = [{"index": 0, "function": {"arguments": ": 1}"}}]
        accumulate_openai_tool_calls(accumulated, chunk_tool_calls)
        assert accumulated[0]["function"]["arguments"] == '{"a": 1}'

    def test_no_index(self):
        accumulated = {}
        chunk_tool_calls = [{"function": {"name": "fn"}}]
        accumulate_openai_tool_calls(accumulated, chunk_tool_calls)
        assert len(accumulated) == 0

    def test_multiple_tool_calls(self):
        accumulated = {}
        chunk1 = [
            {
                "index": 0,
                "id": "tc_1",
                "type": "function",
                "function": {"name": "fn1", "arguments": ""},
            },
            {
                "index": 1,
                "id": "tc_2",
                "type": "function",
                "function": {"name": "fn2", "arguments": ""},
            },
        ]
        accumulate_openai_tool_calls(accumulated, chunk1)
        assert len(accumulated) == 2
        assert accumulated[0]["function"]["name"] == "fn1"
        assert accumulated[1]["function"]["name"] == "fn2"


# =======================
# format_openai_streaming_output
# =======================


class TestFormatOpenAIStreamingOutput:
    def test_chat_text(self):
        result = format_openai_streaming_output("Hello", "chat")
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
        ]

    def test_chat_empty(self):
        result = format_openai_streaming_output("", "chat")
        assert result == [{"role": "assistant", "content": []}]

    def test_chat_with_tool_calls(self):
        tool_calls = [{"id": "tc_1", "function": {"name": "fn", "arguments": "{}"}}]
        result = format_openai_streaming_output("", "chat", tool_calls=tool_calls)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "function"

    def test_chat_list_content(self):
        result = format_openai_streaming_output(["Hello", " World"], "chat")
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello World"}]}
        ]

    def test_chat_list_empty_items(self):
        result = format_openai_streaming_output([None, "", None], "chat")
        assert result == [{"role": "assistant", "content": []}]

    def test_responses_list(self):
        output = [{"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}]
        result = format_openai_streaming_output(output, "responses")
        assert result == output

    def test_responses_string(self):
        result = format_openai_streaming_output("Hello", "responses")
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
        ]

    def test_responses_empty_list(self):
        result = format_openai_streaming_output([], "responses")
        # Falls through to fallback
        assert result[0]["role"] == "assistant"

    def test_unknown_provider_fallback(self):
        result = format_openai_streaming_output(42, "unknown")
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "42"}]}
        ]


# =======================
# format_openai_streaming_input
# =======================


class TestFormatOpenAIStreamingInput:
    def test_basic_input(self):
        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "model": "gpt-4"}
        result = format_openai_streaming_input(kwargs)
        # Should call merge_system_prompt which handles the formatting
        assert result is not None
