"""
Unit tests for posthog.ai.anthropic.anthropic_converter module.

Tests conversion of Anthropic API responses/inputs into standardized PostHog formats.
Covers response formatting, streaming, tool calls, usage extraction, and edge cases.
"""

from posthog.ai.anthropic.anthropic_converter import (
    extract_anthropic_stop_reason,
    extract_anthropic_tools,
    extract_anthropic_usage_from_event,
    extract_anthropic_usage_from_response,
    extract_anthropic_web_search_count,
    finalize_anthropic_tool_input,
    format_anthropic_input,
    format_anthropic_response,
    format_anthropic_streaming_content,
    format_anthropic_streaming_input,
    format_anthropic_streaming_output_complete,
    handle_anthropic_content_block_start,
    handle_anthropic_text_delta,
    handle_anthropic_tool_delta,
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
# format_anthropic_response
# =======================


class TestFormatAnthropicResponse:
    def test_none_response(self):
        assert format_anthropic_response(None) == []

    def test_basic_text_response(self):
        content_block = MockObj(type="text", text="Hello!")
        response = MockObj(content=[content_block])
        result = format_anthropic_response(response)
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]}
        ]

    def test_empty_text(self):
        content_block = MockObj(type="text", text="")
        response = MockObj(content=[content_block])
        result = format_anthropic_response(response)
        # Empty text is excluded (the condition checks choice.text is truthy)
        assert result == []

    def test_tool_use_response(self):
        tool_block = MockObj(
            type="tool_use", id="toolu_1", name="get_weather", input={"city": "NYC"}
        )
        response = MockObj(content=[tool_block])
        result = format_anthropic_response(response)
        assert result == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function",
                        "id": "toolu_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "NYC"},
                        },
                    }
                ],
            }
        ]

    def test_tool_use_no_input(self):
        tool_block = MockObj(type="tool_use", id="toolu_2", name="list_files")
        # no input attribute
        response = MockObj(content=[tool_block])
        result = format_anthropic_response(response)
        assert result[0]["content"][0]["function"]["arguments"] == {}

    def test_mixed_content(self):
        text_block = MockObj(type="text", text="Let me check")
        tool_block = MockObj(
            type="tool_use", id="toolu_1", name="search", input={"q": "test"}
        )
        response = MockObj(content=[text_block, tool_block])
        result = format_anthropic_response(response)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_unknown_content_type(self):
        block = MockObj(type="image", source={"type": "base64"})
        response = MockObj(content=[block])
        result = format_anthropic_response(response)
        assert result == []

    def test_no_content_attr(self):
        response = MockObj()
        result = format_anthropic_response(response)
        assert result == []


# =======================
# format_anthropic_input
# =======================


class TestFormatAnthropicInput:
    def test_with_system_prompt(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = format_anthropic_input(messages, system="You are helpful")
        assert result == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

    def test_without_system_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = format_anthropic_input(messages)
        assert result == [{"role": "user", "content": "Hi"}]

    def test_empty_messages(self):
        result = format_anthropic_input([])
        assert result == []

    def test_none_messages(self):
        result = format_anthropic_input(None, system="System")
        assert result == [{"role": "system", "content": "System"}]

    def test_messages_missing_keys(self):
        messages = [{}]
        result = format_anthropic_input(messages)
        assert result == [{"role": "user", "content": ""}]


# =======================
# extract_anthropic_tools
# =======================


class TestExtractAnthropicTools:
    def test_tools_present(self):
        kwargs = {"tools": [{"name": "fn", "description": "desc", "input_schema": {}}]}
        assert extract_anthropic_tools(kwargs) == kwargs["tools"]

    def test_no_tools(self):
        assert extract_anthropic_tools({}) is None


# =======================
# format_anthropic_streaming_content
# =======================


class TestFormatAnthropicStreamingContent:
    def test_text_block(self):
        blocks = [{"type": "text", "text": "Hello streaming"}]
        result = format_anthropic_streaming_content(blocks)
        assert result == [{"type": "text", "text": "Hello streaming"}]

    def test_function_block(self):
        blocks = [
            {
                "type": "function",
                "id": "toolu_1",
                "function": {"name": "fn", "arguments": {}},
            }
        ]
        result = format_anthropic_streaming_content(blocks)
        assert result == [
            {
                "type": "function",
                "id": "toolu_1",
                "function": {"name": "fn", "arguments": {}},
            }
        ]

    def test_empty_text(self):
        blocks = [{"type": "text", "text": None}]
        result = format_anthropic_streaming_content(blocks)
        assert result == [{"type": "text", "text": ""}]

    def test_function_no_function_key(self):
        blocks = [{"type": "function", "id": "x"}]
        result = format_anthropic_streaming_content(blocks)
        assert result == [{"type": "function", "id": "x", "function": {}}]

    def test_unknown_type(self):
        blocks = [{"type": "image", "data": "base64"}]
        result = format_anthropic_streaming_content(blocks)
        assert result == []

    def test_empty_blocks(self):
        assert format_anthropic_streaming_content([]) == []


# =======================
# extract_anthropic_web_search_count
# =======================


class TestExtractAnthropicWebSearchCount:
    def test_no_usage(self):
        response = MockObj()
        assert extract_anthropic_web_search_count(response) == 0

    def test_no_server_tool_use(self):
        usage = MockObj()
        response = MockObj(usage=usage)
        assert extract_anthropic_web_search_count(response) == 0

    def test_web_search_requests(self):
        server_tool_use = MockObj(web_search_requests=3)
        usage = MockObj(server_tool_use=server_tool_use)
        response = MockObj(usage=usage)
        assert extract_anthropic_web_search_count(response) == 3

    def test_web_search_requests_zero(self):
        server_tool_use = MockObj(web_search_requests=0)
        usage = MockObj(server_tool_use=server_tool_use)
        response = MockObj(usage=usage)
        assert extract_anthropic_web_search_count(response) == 0

    def test_negative_clamped_to_zero(self):
        server_tool_use = MockObj(web_search_requests=-1)
        usage = MockObj(server_tool_use=server_tool_use)
        response = MockObj(usage=usage)
        assert extract_anthropic_web_search_count(response) == 0

    def test_server_tool_use_no_web_search_attr(self):
        server_tool_use = MockObj()
        usage = MockObj(server_tool_use=server_tool_use)
        response = MockObj(usage=usage)
        assert extract_anthropic_web_search_count(response) == 0


# =======================
# extract_anthropic_stop_reason
# =======================


class TestExtractAnthropicStopReason:
    def test_with_stop_reason(self):
        response = MockObj(stop_reason="end_turn")
        assert extract_anthropic_stop_reason(response) == "end_turn"

    def test_no_stop_reason(self):
        response = MockObj()
        assert extract_anthropic_stop_reason(response) is None


# =======================
# extract_anthropic_usage_from_response
# =======================


class TestExtractAnthropicUsageFromResponse:
    def test_no_usage(self):
        response = MockObj()
        result = extract_anthropic_usage_from_response(response)
        assert result == {"input_tokens": 0, "output_tokens": 0}

    def test_basic_usage(self):
        usage = MockObj(input_tokens=100, output_tokens=50)
        response = MockObj(usage=usage)
        result = extract_anthropic_usage_from_response(response)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_with_cache_read(self):
        usage = MockObj(input_tokens=100, output_tokens=50, cache_read_input_tokens=20)
        response = MockObj(usage=usage)
        result = extract_anthropic_usage_from_response(response)
        assert result["cache_read_input_tokens"] == 20

    def test_with_cache_creation(self):
        usage = MockObj(
            input_tokens=100, output_tokens=50, cache_creation_input_tokens=30
        )
        response = MockObj(usage=usage)
        result = extract_anthropic_usage_from_response(response)
        assert result["cache_creation_input_tokens"] == 30

    def test_cache_zero_not_included(self):
        usage = MockObj(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        response = MockObj(usage=usage)
        result = extract_anthropic_usage_from_response(response)
        assert "cache_read_input_tokens" not in result
        assert "cache_creation_input_tokens" not in result

    def test_with_web_search(self):
        server_tool_use = MockObj(web_search_requests=2)
        usage = MockObj(
            input_tokens=10, output_tokens=5, server_tool_use=server_tool_use
        )
        response = MockObj(usage=usage)
        result = extract_anthropic_usage_from_response(response)
        assert result["web_search_count"] == 2


# =======================
# extract_anthropic_usage_from_event
# =======================


class TestExtractAnthropicUsageFromEvent:
    def test_non_message_event(self):
        event = MockObj(type="content_block_delta")
        result = extract_anthropic_usage_from_event(event)
        assert result == {}

    def test_message_start_event(self):
        msg_usage = MockObj(
            input_tokens=50, cache_creation_input_tokens=10, cache_read_input_tokens=5
        )
        message = MockObj(usage=msg_usage)
        event = MockObj(type="message_start", message=message)
        result = extract_anthropic_usage_from_event(event)
        assert result["input_tokens"] == 50
        assert result["cache_creation_input_tokens"] == 10
        assert result["cache_read_input_tokens"] == 5

    def test_message_delta_event(self):
        usage = MockObj(output_tokens=25)
        event = MockObj(type="message_delta", usage=usage)
        result = extract_anthropic_usage_from_event(event)
        assert result["output_tokens"] == 25

    def test_message_delta_with_web_search(self):
        server_tool_use = MockObj(web_search_requests=1)
        usage = MockObj(output_tokens=10, server_tool_use=server_tool_use)
        event = MockObj(type="message_delta", usage=usage)
        result = extract_anthropic_usage_from_event(event)
        assert result["web_search_count"] == 1

    def test_message_delta_no_web_search_attr(self):
        usage = MockObj(output_tokens=10)
        event = MockObj(type="message_delta", usage=usage)
        result = extract_anthropic_usage_from_event(event)
        assert "web_search_count" not in result

    def test_event_usage_none(self):
        event = MockObj(type="message_delta", usage=None)
        result = extract_anthropic_usage_from_event(event)
        assert result == {}


# =======================
# handle_anthropic_content_block_start
# =======================


class TestHandleAnthropicContentBlockStart:
    def test_text_block(self):
        block = MockObj(type="text")
        event = MockObj(type="content_block_start", content_block=block)
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block == {"type": "text", "text": ""}
        assert tool is None

    def test_tool_use_block(self):
        block = MockObj(type="tool_use", id="toolu_1", name="get_weather")
        event = MockObj(type="content_block_start", content_block=block)
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block == {
            "type": "function",
            "id": "toolu_1",
            "function": {"name": "get_weather", "arguments": {}},
        }
        assert tool is not None
        assert tool["input_string"] == ""

    def test_wrong_event_type(self):
        event = MockObj(type="content_block_delta")
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block is None
        assert tool is None

    def test_no_content_block(self):
        event = MockObj(type="content_block_start")
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block is None
        assert tool is None

    def test_block_no_type(self):
        block = MockObj()  # no type attribute
        event = MockObj(type="content_block_start", content_block=block)
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block is None
        assert tool is None

    def test_unknown_block_type(self):
        block = MockObj(type="image")
        event = MockObj(type="content_block_start", content_block=block)
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block is None
        assert tool is None


# =======================
# handle_anthropic_text_delta
# =======================


class TestHandleAnthropicTextDelta:
    def test_text_delta(self):
        delta = MockObj(text="Hello")
        event = MockObj(delta=delta)
        current_block = {"type": "text", "text": ""}
        result = handle_anthropic_text_delta(event, current_block)
        assert result == "Hello"
        assert current_block["text"] == "Hello"

    def test_text_delta_append(self):
        delta = MockObj(text=" World")
        event = MockObj(delta=delta)
        current_block = {"type": "text", "text": "Hello"}
        result = handle_anthropic_text_delta(event, current_block)
        assert result == " World"
        assert current_block["text"] == "Hello World"

    def test_text_delta_none_text_in_block(self):
        delta = MockObj(text="Start")
        event = MockObj(delta=delta)
        current_block = {"type": "text", "text": None}
        result = handle_anthropic_text_delta(event, current_block)
        assert result == "Start"
        assert current_block["text"] == "Start"

    def test_no_delta_attr(self):
        event = MockObj()
        current_block = {"type": "text", "text": ""}
        result = handle_anthropic_text_delta(event, current_block)
        assert result is None

    def test_delta_no_text(self):
        delta = MockObj()  # no text attribute
        event = MockObj(delta=delta)
        current_block = {"type": "text", "text": ""}
        result = handle_anthropic_text_delta(event, current_block)
        assert result is None

    def test_empty_delta_text(self):
        delta = MockObj(text="")
        event = MockObj(delta=delta)
        current_block = {"type": "text", "text": "Hello"}
        result = handle_anthropic_text_delta(event, current_block)
        assert result == ""
        assert current_block["text"] == "Hello"

    def test_none_current_block(self):
        delta = MockObj(text="text")
        event = MockObj(delta=delta)
        result = handle_anthropic_text_delta(event, None)
        assert result == "text"

    def test_non_text_block(self):
        delta = MockObj(text="text")
        event = MockObj(delta=delta)
        current_block = {"type": "function", "id": "x"}
        result = handle_anthropic_text_delta(event, current_block)
        assert result == "text"


# =======================
# handle_anthropic_tool_delta
# =======================


class TestHandleAnthropicToolDelta:
    def test_tool_input_delta(self):
        delta = MockObj(type="input_json_delta", partial_json='{"key": ')
        event = MockObj(type="content_block_delta", delta=delta, index=0)
        content_blocks = [
            {"type": "function", "id": "toolu_1", "function": {"name": "fn"}}
        ]
        tools_in_progress = {
            "toolu_1": {"block": content_blocks[0], "input_string": ""}
        }
        handle_anthropic_tool_delta(event, content_blocks, tools_in_progress)
        assert tools_in_progress["toolu_1"]["input_string"] == '{"key": '

    def test_wrong_event_type(self):
        event = MockObj(type="content_block_start")
        content_blocks = []
        tools_in_progress = {}
        handle_anthropic_tool_delta(event, content_blocks, tools_in_progress)
        # No error, just returns

    def test_wrong_delta_type(self):
        delta = MockObj(type="text_delta")
        event = MockObj(type="content_block_delta", delta=delta, index=0)
        content_blocks = [{"type": "function", "id": "x"}]
        tools_in_progress = {"x": {"block": content_blocks[0], "input_string": ""}}
        handle_anthropic_tool_delta(event, content_blocks, tools_in_progress)
        assert tools_in_progress["x"]["input_string"] == ""

    def test_index_out_of_range(self):
        delta = MockObj(type="input_json_delta", partial_json="{}")
        event = MockObj(type="content_block_delta", delta=delta, index=5)
        content_blocks = [{"type": "function", "id": "x"}]
        tools_in_progress = {"x": {"block": content_blocks[0], "input_string": ""}}
        handle_anthropic_tool_delta(event, content_blocks, tools_in_progress)
        # No error, index >= len so skipped


# =======================
# finalize_anthropic_tool_input
# =======================


class TestFinalizeAnthropicToolInput:
    def test_finalize_valid_json(self):
        content_blocks = [
            {
                "type": "function",
                "id": "toolu_1",
                "function": {"name": "fn", "arguments": {}},
            }
        ]
        tools_in_progress = {
            "toolu_1": {"block": content_blocks[0], "input_string": '{"city": "NYC"}'}
        }
        event = MockObj(type="content_block_stop", index=0)
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        assert content_blocks[0]["function"]["arguments"] == {"city": "NYC"}
        assert "toolu_1" not in tools_in_progress

    def test_finalize_invalid_json(self):
        content_blocks = [
            {
                "type": "function",
                "id": "toolu_1",
                "function": {"name": "fn", "arguments": {}},
            }
        ]
        tools_in_progress = {
            "toolu_1": {"block": content_blocks[0], "input_string": "not valid json"}
        }
        event = MockObj(type="content_block_stop", index=0)
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        # Keep empty dict on JSON parse failure
        assert content_blocks[0]["function"]["arguments"] == {}

    def test_wrong_event_type(self):
        event = MockObj(type="content_block_delta", index=0)
        content_blocks = [
            {"type": "function", "id": "x", "function": {"arguments": {}}}
        ]
        tools_in_progress = {"x": {"block": content_blocks[0], "input_string": "{}"}}
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        # Not finalized because wrong event type
        assert "x" in tools_in_progress

    def test_index_out_of_range(self):
        event = MockObj(type="content_block_stop", index=5)
        content_blocks = [
            {"type": "function", "id": "x", "function": {"arguments": {}}}
        ]
        tools_in_progress = {"x": {"block": content_blocks[0], "input_string": "{}"}}
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        # Not finalized, still in progress
        assert "x" in tools_in_progress

    def test_text_block_not_finalized(self):
        """Text blocks are not touched by finalize."""
        event = MockObj(type="content_block_stop", index=0)
        content_blocks = [{"type": "text", "text": "hello"}]
        tools_in_progress = {}
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        # No error


# =======================
# format_anthropic_streaming_input
# =======================


class TestFormatAnthropicStreamingInput:
    def test_basic(self):
        kwargs = {"messages": [{"role": "user", "content": "Hi"}], "model": "claude-3"}
        result = format_anthropic_streaming_input(kwargs)
        assert result is not None


# =======================
# format_anthropic_streaming_output_complete
# =======================


class TestFormatAnthropicStreamingOutputComplete:
    def test_with_content_blocks(self):
        content_blocks = [{"type": "text", "text": "Hello"}]
        result = format_anthropic_streaming_output_complete(content_blocks, "fallback")
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}
        ]

    def test_fallback_to_accumulated(self):
        result = format_anthropic_streaming_output_complete([], "Accumulated text")
        assert result == [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Accumulated text"}],
            }
        ]

    def test_empty_both(self):
        result = format_anthropic_streaming_output_complete([], "")
        assert result == [
            {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]
