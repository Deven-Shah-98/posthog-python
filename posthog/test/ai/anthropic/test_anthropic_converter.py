"""
Tests for posthog.ai.anthropic.anthropic_converter module.

Covers: format_anthropic_response, format_anthropic_input,
extract_anthropic_tools, format_anthropic_streaming_content,
extract_anthropic_web_search_count, extract_anthropic_stop_reason,
extract_anthropic_usage_from_response, extract_anthropic_usage_from_event,
handle_anthropic_content_block_start, handle_anthropic_text_delta,
handle_anthropic_tool_delta, finalize_anthropic_tool_input,
format_anthropic_streaming_input, format_anthropic_streaming_output_complete.
"""

from types import SimpleNamespace


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


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# format_anthropic_response
# ---------------------------------------------------------------------------


class TestFormatAnthropicResponse:
    def test_none_response(self):
        assert format_anthropic_response(None) == []

    def test_text_content(self):
        block = _ns(type="text", text="Hello world")
        resp = _ns(content=[block])
        result = format_anthropic_response(resp)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hello world"}]

    def test_empty_text_ignored(self):
        block = _ns(type="text", text="")
        resp = _ns(content=[block])
        assert format_anthropic_response(resp) == []

    def test_tool_use(self):
        block = _ns(type="tool_use", id="tool_1", name="search", input={"q": "test"})
        resp = _ns(content=[block])
        result = format_anthropic_response(resp)
        fc = result[0]["content"][0]
        assert fc["type"] == "function"
        assert fc["id"] == "tool_1"
        assert fc["function"]["name"] == "search"
        assert fc["function"]["arguments"] == {"q": "test"}

    def test_tool_use_no_input(self):
        block = _ns(type="tool_use", id="tool_2", name="fn")
        resp = _ns(content=[block])
        result = format_anthropic_response(resp)
        assert result[0]["content"][0]["function"]["arguments"] == {}

    def test_text_and_tool_use(self):
        blocks = [
            _ns(type="text", text="thinking"),
            _ns(type="tool_use", id="t1", name="fn", input={}),
        ]
        resp = _ns(content=blocks)
        result = format_anthropic_response(resp)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "function"

    def test_unknown_type_ignored(self):
        block = _ns(type="image", data="binary")
        resp = _ns(content=[block])
        assert format_anthropic_response(resp) == []

    def test_no_content_attr(self):
        resp = _ns(id="msg_1")
        assert format_anthropic_response(resp) == []

    def test_empty_content_list(self):
        resp = _ns(content=[])
        assert format_anthropic_response(resp) == []


# ---------------------------------------------------------------------------
# format_anthropic_input
# ---------------------------------------------------------------------------


class TestFormatAnthropicInput:
    def test_messages_only(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = format_anthropic_input(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_with_system(self):
        msgs = [{"role": "user", "content": "q"}]
        result = format_anthropic_input(msgs, system="You are helpful")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"

    def test_empty_messages(self):
        result = format_anthropic_input([])
        assert result == []

    def test_none_messages(self):
        result = format_anthropic_input([], system="sys")
        assert len(result) == 1

    def test_missing_role_defaults(self):
        result = format_anthropic_input([{"content": "test"}])
        assert result[0]["role"] == "user"

    def test_missing_content_defaults(self):
        result = format_anthropic_input([{"role": "user"}])
        assert result[0]["content"] == ""


# ---------------------------------------------------------------------------
# extract_anthropic_tools
# ---------------------------------------------------------------------------


class TestExtractAnthropicTools:
    def test_has_tools(self):
        assert extract_anthropic_tools({"tools": [{"name": "fn"}]}) == [{"name": "fn"}]

    def test_no_tools(self):
        assert extract_anthropic_tools({"model": "claude-3"}) is None


# ---------------------------------------------------------------------------
# format_anthropic_streaming_content
# ---------------------------------------------------------------------------


class TestFormatAnthropicStreamingContent:
    def test_text_block(self):
        blocks = [{"type": "text", "text": "hello"}]
        result = format_anthropic_streaming_content(blocks)
        assert result == [{"type": "text", "text": "hello"}]

    def test_function_block(self):
        blocks = [{"type": "function", "id": "t1", "function": {"name": "fn"}}]
        result = format_anthropic_streaming_content(blocks)
        assert result[0]["type"] == "function"
        assert result[0]["id"] == "t1"

    def test_empty_text(self):
        blocks = [{"type": "text", "text": None}]
        result = format_anthropic_streaming_content(blocks)
        assert result[0]["text"] == ""

    def test_missing_function(self):
        blocks = [{"type": "function", "id": "t1", "function": None}]
        result = format_anthropic_streaming_content(blocks)
        assert result[0]["function"] == {}

    def test_mixed(self):
        blocks = [
            {"type": "text", "text": "hi"},
            {"type": "function", "id": "t1", "function": {"name": "fn"}},
        ]
        result = format_anthropic_streaming_content(blocks)
        assert len(result) == 2

    def test_empty_list(self):
        assert format_anthropic_streaming_content([]) == []

    def test_unknown_type_ignored(self):
        blocks = [{"type": "image", "data": "x"}]
        assert format_anthropic_streaming_content(blocks) == []


# ---------------------------------------------------------------------------
# extract_anthropic_web_search_count
# ---------------------------------------------------------------------------


class TestExtractAnthropicWebSearchCount:
    def test_no_usage(self):
        resp = _ns(id="msg")
        assert extract_anthropic_web_search_count(resp) == 0

    def test_no_server_tool_use(self):
        resp = _ns(usage=_ns(input_tokens=100))
        assert extract_anthropic_web_search_count(resp) == 0

    def test_with_web_search_requests(self):
        resp = _ns(usage=_ns(server_tool_use=_ns(web_search_requests=3)))
        assert extract_anthropic_web_search_count(resp) == 3

    def test_zero_web_search(self):
        resp = _ns(usage=_ns(server_tool_use=_ns(web_search_requests=0)))
        assert extract_anthropic_web_search_count(resp) == 0

    def test_no_web_search_attr(self):
        resp = _ns(usage=_ns(server_tool_use=_ns()))
        assert extract_anthropic_web_search_count(resp) == 0


# ---------------------------------------------------------------------------
# extract_anthropic_stop_reason
# ---------------------------------------------------------------------------


class TestExtractAnthropicStopReason:
    def test_has_stop_reason(self):
        resp = _ns(stop_reason="end_turn")
        assert extract_anthropic_stop_reason(resp) == "end_turn"

    def test_no_stop_reason(self):
        assert extract_anthropic_stop_reason(_ns()) is None


# ---------------------------------------------------------------------------
# extract_anthropic_usage_from_response
# ---------------------------------------------------------------------------


class TestExtractAnthropicUsageFromResponse:
    def test_no_usage(self):
        resp = _ns()
        result = extract_anthropic_usage_from_response(resp)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_basic_usage(self):
        usage = _ns(input_tokens=100, output_tokens=50)
        usage.model_dump = lambda: {"input_tokens": 100, "output_tokens": 50}
        resp = _ns(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert "raw_usage" in result

    def test_with_cache_tokens(self):
        usage = _ns(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=10,
        )
        usage.model_dump = lambda: {}
        resp = _ns(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["cache_read_input_tokens"] == 20
        assert result["cache_creation_input_tokens"] == 10

    def test_zero_cache_not_added(self):
        usage = _ns(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        usage.model_dump = lambda: {}
        resp = _ns(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert "cache_read_input_tokens" not in result
        assert "cache_creation_input_tokens" not in result

    def test_with_web_search_count(self):
        usage = _ns(
            input_tokens=100,
            output_tokens=50,
            server_tool_use=_ns(web_search_requests=2),
        )
        usage.model_dump = lambda: {}
        resp = _ns(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["web_search_count"] == 2


# ---------------------------------------------------------------------------
# extract_anthropic_usage_from_event
# ---------------------------------------------------------------------------


class TestExtractAnthropicUsageFromEvent:
    def test_message_start_event(self):
        msg_usage = _ns(
            input_tokens=150,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=10,
        )
        msg_usage.model_dump = lambda: {"input_tokens": 150}
        event = _ns(
            type="message_start",
            message=_ns(usage=msg_usage),
        )
        result = extract_anthropic_usage_from_event(event)
        assert result["input_tokens"] == 150
        assert result["cache_creation_input_tokens"] == 5
        assert result["cache_read_input_tokens"] == 10
        assert "raw_usage" in result

    def test_message_delta_event(self):
        event_usage = _ns(output_tokens=42)
        event_usage.model_dump = lambda: {"output_tokens": 42}
        event = _ns(type="message_delta", usage=event_usage)
        result = extract_anthropic_usage_from_event(event)
        assert result["output_tokens"] == 42

    def test_message_delta_with_web_search(self):
        event_usage = _ns(
            output_tokens=10,
            server_tool_use=_ns(web_search_requests=3),
        )
        event_usage.model_dump = lambda: {}
        event = _ns(type="message_delta", usage=event_usage)
        result = extract_anthropic_usage_from_event(event)
        assert result["web_search_count"] == 3

    def test_unrelated_event(self):
        event = _ns(type="content_block_start")
        result = extract_anthropic_usage_from_event(event)
        assert result == {}

    def test_message_start_no_message(self):
        event = _ns(type="message_start")
        result = extract_anthropic_usage_from_event(event)
        assert result == {}


# ---------------------------------------------------------------------------
# handle_anthropic_content_block_start
# ---------------------------------------------------------------------------


class TestHandleAnthropicContentBlockStart:
    def test_text_block(self):
        event = _ns(type="content_block_start", content_block=_ns(type="text"))
        block, tool = handle_anthropic_content_block_start(event)
        assert block == {"type": "text", "text": ""}
        assert tool is None

    def test_tool_use_block(self):
        event = _ns(
            type="content_block_start",
            content_block=_ns(type="tool_use", id="t1", name="search"),
        )
        block, tool = handle_anthropic_content_block_start(event)
        assert block["type"] == "function"
        assert block["id"] == "t1"
        assert block["function"]["name"] == "search"
        assert tool is not None
        assert tool["input_string"] == ""

    def test_wrong_event_type(self):
        event = _ns(type="content_block_delta")
        block, tool = handle_anthropic_content_block_start(event)
        assert block is None
        assert tool is None

    def test_no_content_block(self):
        event = _ns(type="content_block_start")
        block, tool = handle_anthropic_content_block_start(event)
        assert block is None
        assert tool is None

    def test_unknown_block_type(self):
        event = _ns(type="content_block_start", content_block=_ns(type="image"))
        block, tool = handle_anthropic_content_block_start(event)
        assert block is None
        assert tool is None

    def test_no_block_type(self):
        event = _ns(type="content_block_start", content_block=_ns())
        block, tool = handle_anthropic_content_block_start(event)
        assert block is None
        assert tool is None


# ---------------------------------------------------------------------------
# handle_anthropic_text_delta
# ---------------------------------------------------------------------------


class TestHandleAnthropicTextDelta:
    def test_accumulates_text(self):
        current_block = {"type": "text", "text": "hello"}
        event = _ns(delta=_ns(text=" world"))
        result = handle_anthropic_text_delta(event, current_block)
        assert result == " world"
        assert current_block["text"] == "hello world"

    def test_none_text_in_block(self):
        current_block = {"type": "text", "text": None}
        event = _ns(delta=_ns(text="start"))
        handle_anthropic_text_delta(event, current_block)
        assert current_block["text"] == "start"

    def test_no_delta(self):
        result = handle_anthropic_text_delta(_ns(), None)
        assert result is None

    def test_none_block(self):
        event = _ns(delta=_ns(text="text"))
        result = handle_anthropic_text_delta(event, None)
        assert result == "text"

    def test_empty_delta_text(self):
        current_block = {"type": "text", "text": "hi"}
        event = _ns(delta=_ns(text=None))
        result = handle_anthropic_text_delta(event, current_block)
        assert result == ""

    def test_function_block_not_modified(self):
        current_block = {"type": "function", "id": "t1"}
        event = _ns(delta=_ns(text="data"))
        result = handle_anthropic_text_delta(event, current_block)
        assert result == "data"
        # Block not modified because it's not type text
        assert "text" not in current_block


# ---------------------------------------------------------------------------
# handle_anthropic_tool_delta
# ---------------------------------------------------------------------------


class TestHandleAnthropicToolDelta:
    def test_accumulates_json(self):
        block = {"type": "function", "id": "t1", "function": {"name": "fn"}}
        tool_in_progress = {"block": block, "input_string": '{"key":'}
        content_blocks = [block]
        tools_in_progress = {"t1": tool_in_progress}
        event = _ns(
            type="content_block_delta",
            delta=_ns(type="input_json_delta", partial_json='"value"}'),
            index=0,
        )
        handle_anthropic_tool_delta(event, content_blocks, tools_in_progress)
        assert tools_in_progress["t1"]["input_string"] == '{"key":"value"}'

    def test_wrong_event_type(self):
        tools_in_progress = {}
        handle_anthropic_tool_delta(
            _ns(type="content_block_start"), [], tools_in_progress
        )
        assert tools_in_progress == {}

    def test_wrong_delta_type(self):
        block = {"type": "function", "id": "t1"}
        event = _ns(
            type="content_block_delta",
            delta=_ns(type="text_delta", text="hi"),
            index=0,
        )
        handle_anthropic_tool_delta(event, [block], {})

    def test_index_out_of_range(self):
        event = _ns(
            type="content_block_delta",
            delta=_ns(type="input_json_delta", partial_json="x"),
            index=5,
        )
        handle_anthropic_tool_delta(event, [], {})


# ---------------------------------------------------------------------------
# finalize_anthropic_tool_input
# ---------------------------------------------------------------------------


class TestFinalizeAnthropicToolInput:
    def test_parses_json(self):
        block = {
            "type": "function",
            "id": "t1",
            "function": {"name": "fn", "arguments": {}},
        }
        tool_in_progress = {"block": block, "input_string": '{"city":"NY"}'}
        content_blocks = [block]
        tools_in_progress = {"t1": tool_in_progress}
        event = _ns(type="content_block_stop", index=0)
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        assert block["function"]["arguments"] == {"city": "NY"}
        assert "t1" not in tools_in_progress

    def test_invalid_json_keeps_empty(self):
        block = {
            "type": "function",
            "id": "t1",
            "function": {"name": "fn", "arguments": {}},
        }
        tool_in_progress = {"block": block, "input_string": "not json"}
        content_blocks = [block]
        tools_in_progress = {"t1": tool_in_progress}
        event = _ns(type="content_block_stop", index=0)
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        assert block["function"]["arguments"] == {}

    def test_wrong_event_type(self):
        event = _ns(type="content_block_start", index=0)
        finalize_anthropic_tool_input(event, [], {})

    def test_index_out_of_range(self):
        event = _ns(type="content_block_stop", index=5)
        finalize_anthropic_tool_input(event, [], {})


# ---------------------------------------------------------------------------
# format_anthropic_streaming_input
# ---------------------------------------------------------------------------


class TestFormatAnthropicStreamingInput:
    def test_basic(self):
        kwargs = {"messages": [{"role": "user", "content": "hi"}], "model": "claude-3"}
        result = format_anthropic_streaming_input(kwargs)
        assert result is not None


# ---------------------------------------------------------------------------
# format_anthropic_streaming_output_complete
# ---------------------------------------------------------------------------


class TestFormatAnthropicStreamingOutputComplete:
    def test_with_content_blocks(self):
        blocks = [{"type": "text", "text": "answer"}]
        result = format_anthropic_streaming_output_complete(blocks, "")
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["text"] == "answer"

    def test_fallback_to_accumulated_content(self):
        result = format_anthropic_streaming_output_complete([], "fallback text")
        assert result[0]["content"][0]["text"] == "fallback text"

    def test_empty_both(self):
        result = format_anthropic_streaming_output_complete([], "")
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["text"] == ""
