"""Unit tests for posthog.ai.anthropic.anthropic_converter."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Obj:
    """Simple namespace for mock objects."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ===========================================================================
# format_anthropic_response
# ===========================================================================


class TestFormatAnthropicResponse:
    def test_none_response(self):
        assert format_anthropic_response(None) == []

    def test_text_content(self):
        block = _Obj(type="text", text="Hello world")
        resp = _Obj(content=[block])
        result = format_anthropic_response(resp)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hello world"}]

    def test_empty_text_skipped(self):
        block = _Obj(type="text", text="")
        resp = _Obj(content=[block])
        result = format_anthropic_response(resp)
        assert result == []

    def test_tool_use_content(self):
        block = _Obj(
            type="tool_use", name="get_weather", id="tu_1", input={"city": "NYC"}
        )
        resp = _Obj(content=[block])
        result = format_anthropic_response(resp)
        items = result[0]["content"]
        assert items[0]["type"] == "function"
        assert items[0]["id"] == "tu_1"
        assert items[0]["function"]["name"] == "get_weather"
        assert items[0]["function"]["arguments"] == {"city": "NYC"}

    def test_tool_use_no_input(self):
        block = _Obj(type="tool_use", name="fn", id="tu_2")
        resp = _Obj(content=[block])
        result = format_anthropic_response(resp)
        assert result[0]["content"][0]["function"]["arguments"] == {}

    def test_mixed_text_and_tool(self):
        text_block = _Obj(type="text", text="Here's the result:")
        tool_block = _Obj(type="tool_use", name="calc", id="tu_3", input={"x": 1})
        resp = _Obj(content=[text_block, tool_block])
        result = format_anthropic_response(resp)
        items = result[0]["content"]
        assert len(items) == 2
        assert items[0]["type"] == "text"
        assert items[1]["type"] == "function"

    def test_unknown_content_type_skipped(self):
        block = _Obj(type="image", data="base64stuff")
        resp = _Obj(content=[block])
        result = format_anthropic_response(resp)
        assert result == []

    def test_no_content_attribute(self):
        resp = _Obj()
        result = format_anthropic_response(resp)
        assert result == []


# ===========================================================================
# format_anthropic_input
# ===========================================================================


class TestFormatAnthropicInput:
    def test_messages_only(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = format_anthropic_input(msgs)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_with_system_prompt(self):
        msgs = [{"role": "user", "content": "Hi"}]
        result = format_anthropic_input(msgs, system="Be helpful")
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "Be helpful"}
        assert result[1] == {"role": "user", "content": "Hi"}

    def test_empty_messages(self):
        result = format_anthropic_input([])
        assert result == []

    def test_none_messages(self):
        result = format_anthropic_input(None)
        assert result == []

    def test_missing_role(self):
        result = format_anthropic_input([{"content": "x"}])
        assert result[0]["role"] == "user"

    def test_missing_content(self):
        result = format_anthropic_input([{"role": "user"}])
        assert result[0]["content"] == ""


# ===========================================================================
# extract_anthropic_tools
# ===========================================================================


class TestExtractAnthropicTools:
    def test_with_tools(self):
        assert extract_anthropic_tools({"tools": [{"name": "fn"}]}) == [{"name": "fn"}]

    def test_no_tools(self):
        assert extract_anthropic_tools({}) is None


# ===========================================================================
# format_anthropic_streaming_content
# ===========================================================================


class TestFormatAnthropicStreamingContent:
    def test_text_blocks(self):
        blocks = [{"type": "text", "text": "Hello"}]
        result = format_anthropic_streaming_content(blocks)
        assert result == [{"type": "text", "text": "Hello"}]

    def test_function_blocks(self):
        blocks = [
            {
                "type": "function",
                "id": "tu_1",
                "function": {"name": "fn", "arguments": {}},
            }
        ]
        result = format_anthropic_streaming_content(blocks)
        assert result[0]["type"] == "function"
        assert result[0]["id"] == "tu_1"

    def test_empty_text(self):
        blocks = [{"type": "text", "text": ""}]
        result = format_anthropic_streaming_content(blocks)
        assert result[0]["text"] == ""

    def test_mixed_blocks(self):
        blocks = [
            {"type": "text", "text": "hi"},
            {"type": "function", "id": "c1", "function": {"name": "fn"}},
        ]
        result = format_anthropic_streaming_content(blocks)
        assert len(result) == 2

    def test_unknown_type_skipped(self):
        blocks = [{"type": "unknown", "data": "x"}]
        result = format_anthropic_streaming_content(blocks)
        assert result == []


# ===========================================================================
# extract_anthropic_web_search_count
# ===========================================================================


class TestExtractAnthropicWebSearchCount:
    def test_no_usage(self):
        resp = _Obj()
        assert extract_anthropic_web_search_count(resp) == 0

    def test_no_server_tool_use(self):
        resp = _Obj(usage=_Obj())
        assert extract_anthropic_web_search_count(resp) == 0

    def test_with_web_search_requests(self):
        resp = _Obj(usage=_Obj(server_tool_use=_Obj(web_search_requests=3)))
        assert extract_anthropic_web_search_count(resp) == 3

    def test_zero_web_search_requests(self):
        resp = _Obj(usage=_Obj(server_tool_use=_Obj(web_search_requests=0)))
        assert extract_anthropic_web_search_count(resp) == 0

    def test_no_web_search_requests_attr(self):
        resp = _Obj(usage=_Obj(server_tool_use=_Obj()))
        assert extract_anthropic_web_search_count(resp) == 0


# ===========================================================================
# extract_anthropic_stop_reason
# ===========================================================================


class TestExtractAnthropicStopReason:
    def test_with_stop_reason(self):
        resp = _Obj(stop_reason="end_turn")
        assert extract_anthropic_stop_reason(resp) == "end_turn"

    def test_no_stop_reason(self):
        assert extract_anthropic_stop_reason(_Obj()) is None


# ===========================================================================
# extract_anthropic_usage_from_response
# ===========================================================================


class TestExtractAnthropicUsageFromResponse:
    def test_no_usage(self):
        resp = _Obj()
        result = extract_anthropic_usage_from_response(resp)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0

    def test_basic_usage(self):
        usage = _Obj(input_tokens=100, output_tokens=50)
        resp = _Obj(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50

    def test_cache_tokens(self):
        usage = _Obj(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=10,
        )
        resp = _Obj(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["cache_read_input_tokens"] == 30
        assert result["cache_creation_input_tokens"] == 10

    def test_zero_cache_not_included(self):
        usage = _Obj(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        resp = _Obj(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert "cache_read_input_tokens" not in result
        assert "cache_creation_input_tokens" not in result

    def test_web_search_count_included(self):
        usage = _Obj(
            input_tokens=50,
            output_tokens=25,
            server_tool_use=_Obj(web_search_requests=2),
        )
        resp = _Obj(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["web_search_count"] == 2

    def test_raw_usage_serialized(self):
        usage = _Obj(
            input_tokens=10,
            output_tokens=5,
            model_dump=lambda: {"input_tokens": 10, "output_tokens": 5},
        )
        resp = _Obj(usage=usage)
        result = extract_anthropic_usage_from_response(resp)
        assert result["raw_usage"] == {"input_tokens": 10, "output_tokens": 5}


# ===========================================================================
# extract_anthropic_usage_from_event
# ===========================================================================


class TestExtractAnthropicUsageFromEvent:
    def test_message_start_event(self):
        msg_usage = _Obj(
            input_tokens=50,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=10,
        )
        event = _Obj(type="message_start", message=_Obj(usage=msg_usage))
        result = extract_anthropic_usage_from_event(event)
        assert result["input_tokens"] == 50
        assert result["cache_creation_input_tokens"] == 5
        assert result["cache_read_input_tokens"] == 10

    def test_message_delta_event(self):
        event = _Obj(type="message_delta", usage=_Obj(output_tokens=30))
        result = extract_anthropic_usage_from_event(event)
        assert result["output_tokens"] == 30

    def test_message_delta_with_web_search(self):
        event = _Obj(
            type="message_delta",
            usage=_Obj(
                output_tokens=30,
                server_tool_use=_Obj(web_search_requests=1),
            ),
        )
        result = extract_anthropic_usage_from_event(event)
        assert result["web_search_count"] == 1

    def test_no_usage_event(self):
        event = _Obj(type="content_block_start")
        result = extract_anthropic_usage_from_event(event)
        assert result.get("input_tokens") is None

    def test_message_delta_zero_web_search(self):
        event = _Obj(
            type="message_delta",
            usage=_Obj(
                output_tokens=10,
                server_tool_use=_Obj(web_search_requests=0),
            ),
        )
        result = extract_anthropic_usage_from_event(event)
        assert "web_search_count" not in result

    def test_raw_usage_in_message_start(self):
        msg_usage = _Obj(
            input_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            model_dump=lambda: {"input_tokens": 50},
        )
        event = _Obj(type="message_start", message=_Obj(usage=msg_usage))
        result = extract_anthropic_usage_from_event(event)
        assert result["raw_usage"] == {"input_tokens": 50}


# ===========================================================================
# handle_anthropic_content_block_start
# ===========================================================================


class TestHandleAnthropicContentBlockStart:
    def test_text_block(self):
        block = _Obj(type="text")
        event = _Obj(type="content_block_start", content_block=block)
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block == {"type": "text", "text": ""}
        assert tool is None

    def test_tool_use_block(self):
        block = _Obj(type="tool_use", id="tu_1", name="fn")
        event = _Obj(type="content_block_start", content_block=block)
        content_block, tool = handle_anthropic_content_block_start(event)
        assert content_block["type"] == "function"
        assert content_block["id"] == "tu_1"
        assert content_block["function"]["name"] == "fn"
        assert tool is not None
        assert tool["input_string"] == ""

    def test_wrong_event_type(self):
        event = _Obj(type="content_block_delta")
        cb, tool = handle_anthropic_content_block_start(event)
        assert cb is None
        assert tool is None

    def test_no_content_block(self):
        event = _Obj(type="content_block_start")
        cb, tool = handle_anthropic_content_block_start(event)
        assert cb is None
        assert tool is None

    def test_block_no_type(self):
        block = _Obj()
        event = _Obj(type="content_block_start", content_block=block)
        cb, tool = handle_anthropic_content_block_start(event)
        assert cb is None
        assert tool is None

    def test_unknown_block_type(self):
        block = _Obj(type="unknown")
        event = _Obj(type="content_block_start", content_block=block)
        cb, tool = handle_anthropic_content_block_start(event)
        assert cb is None
        assert tool is None


# ===========================================================================
# handle_anthropic_text_delta
# ===========================================================================


class TestHandleAnthropicTextDelta:
    def test_appends_text(self):
        block = {"type": "text", "text": "Hello"}
        event = _Obj(delta=_Obj(text=" world"))
        result = handle_anthropic_text_delta(event, block)
        assert result == " world"
        assert block["text"] == "Hello world"

    def test_initializes_from_none(self):
        block = {"type": "text", "text": None}
        event = _Obj(delta=_Obj(text="start"))
        handle_anthropic_text_delta(event, block)
        assert block["text"] == "start"

    def test_no_delta(self):
        block = {"type": "text", "text": "unchanged"}
        event = _Obj()
        assert handle_anthropic_text_delta(event, block) is None
        assert block["text"] == "unchanged"

    def test_delta_no_text(self):
        block = {"type": "text", "text": "x"}
        event = _Obj(delta=_Obj())
        assert handle_anthropic_text_delta(event, block) is None

    def test_none_block(self):
        event = _Obj(delta=_Obj(text="hi"))
        result = handle_anthropic_text_delta(event, None)
        assert result == "hi"

    def test_non_text_block(self):
        block = {"type": "function", "id": "c1"}
        event = _Obj(delta=_Obj(text="x"))
        result = handle_anthropic_text_delta(event, block)
        assert result == "x"
        # block unchanged since type != text
        assert "text" not in block


# ===========================================================================
# handle_anthropic_tool_delta
# ===========================================================================


class TestHandleAnthropicToolDelta:
    def test_accumulates_json(self):
        content_blocks = [
            {
                "type": "function",
                "id": "tu_1",
                "function": {"name": "fn", "arguments": {}},
            },
        ]
        tools_in_progress = {
            "tu_1": {"block": content_blocks[0], "input_string": '{"ke'}
        }
        event = _Obj(
            type="content_block_delta",
            delta=_Obj(type="input_json_delta", partial_json='y": "val"}'),
            index=0,
        )
        handle_anthropic_tool_delta(event, content_blocks, tools_in_progress)
        assert tools_in_progress["tu_1"]["input_string"] == '{"key": "val"}'

    def test_wrong_event_type(self):
        content_blocks = []
        tools = {}
        event = _Obj(type="content_block_start")
        handle_anthropic_tool_delta(event, content_blocks, tools)

    def test_wrong_delta_type(self):
        content_blocks = [{"type": "function", "id": "tu_1", "function": {}}]
        tools = {"tu_1": {"block": content_blocks[0], "input_string": ""}}
        event = _Obj(type="content_block_delta", delta=_Obj(type="text_delta"))
        handle_anthropic_tool_delta(event, content_blocks, tools)
        assert tools["tu_1"]["input_string"] == ""

    def test_index_out_of_range(self):
        content_blocks = []
        tools = {}
        event = _Obj(
            type="content_block_delta",
            delta=_Obj(type="input_json_delta", partial_json="x"),
            index=5,
        )
        handle_anthropic_tool_delta(event, content_blocks, tools)


# ===========================================================================
# finalize_anthropic_tool_input
# ===========================================================================


class TestFinalizeAnthropicToolInput:
    def test_parses_json(self):
        block = {
            "type": "function",
            "id": "tu_1",
            "function": {"name": "fn", "arguments": {}},
        }
        content_blocks = [block]
        tools_in_progress = {
            "tu_1": {"block": block, "input_string": '{"key": "value"}'}
        }
        event = _Obj(type="content_block_stop", index=0)
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        assert block["function"]["arguments"] == {"key": "value"}
        assert "tu_1" not in tools_in_progress

    def test_invalid_json_keeps_empty_dict(self):
        block = {
            "type": "function",
            "id": "tu_2",
            "function": {"name": "fn", "arguments": {}},
        }
        content_blocks = [block]
        tools_in_progress = {"tu_2": {"block": block, "input_string": "not json"}}
        event = _Obj(type="content_block_stop", index=0)
        finalize_anthropic_tool_input(event, content_blocks, tools_in_progress)
        assert block["function"]["arguments"] == {}

    def test_wrong_event_type(self):
        event = _Obj(type="content_block_delta", index=0)
        finalize_anthropic_tool_input(event, [], {})

    def test_index_out_of_range(self):
        event = _Obj(type="content_block_stop", index=5)
        finalize_anthropic_tool_input(event, [], {})


# ===========================================================================
# format_anthropic_streaming_input
# ===========================================================================


class TestFormatAnthropicStreamingInput:
    def test_basic(self):
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        result = format_anthropic_streaming_input(kwargs)
        assert result is not None


# ===========================================================================
# format_anthropic_streaming_output_complete
# ===========================================================================


class TestFormatAnthropicStreamingOutputComplete:
    def test_with_content_blocks(self):
        blocks = [{"type": "text", "text": "Hi there"}]
        result = format_anthropic_streaming_output_complete(blocks, "fallback")
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there"}]

    def test_fallback_to_accumulated(self):
        result = format_anthropic_streaming_output_complete([], "fallback content")
        assert result[0]["content"] == [{"type": "text", "text": "fallback content"}]

    def test_empty_both(self):
        result = format_anthropic_streaming_output_complete([], "")
        assert result[0]["content"] == [{"type": "text", "text": ""}]
