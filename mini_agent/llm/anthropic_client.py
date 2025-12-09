import logging
from typing import Any

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from .base import LLMClientBase


logger = logging.getLogger(__name__)

class AnthropicClient(LLMClientBase):

     def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax Anthropic endpoint)
            model: Model name to use (default: MiniMax-M2)
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)

        # Initialize Anthropic async client
        self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
        )


    async def _make_api_request(
        self,
        system_message : str | None = None,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> anthropic.types.Message:

    params = {
          "model": self.model,
          "max_tokens": 16384,
          "messages": api_messages,
    }
    if system_message:
         params["system"] = system_message

    if tools:
         params["tools"] = self._convert_tools(tools)

    response = await self.client.messages.create(**params)
    return response



   def convert_tools(self,tools : list[Any]) -> list[dict[str, Any]]:
    """Convert tools to Anthropic format.

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

    """
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        
        elif hasattr(tool, "to_schema"):
            result.append(tool.to_schema())

        else:
            raise TypeError(f"Unsupported tool type: {type(tool)}")


    return result


    def convert_messages(self,messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:

        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue

            if msg.role in ["user", "assistant"]:
                 if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                    # Build content blocks for assistant with thinking and/or tool calls
                    content_blocks = []

                    if msg.thinking:
                        content_blocks.append({
                            "type": "thinking", "thinking": msg.thinking
                        })

                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})


                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "input": tool_call.function.arguments,
                                }
                        )

                    api_messages.append({
                        "role": "assistant", "content": content_blocks
                    })

                 else:
                     api_messages.append({"role": msg.role, "content": msg.content})


                elif msg.role == "tool":
                    api_messages.append({
                        {
                            "role" : "tool",
                            "content" : [
                                {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                                }
                            ],
                        }
                    }
                )

                return system_message, api_messages


        

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:

        """Prepare the request for Anthropic API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """

        system_messages,api_messages = self._convert_messages(messages)

        return {
            "system_message": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }


    def _parse_response(self,response : anthropic.types.Message) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Args:
            response: Anthropic Message response

        Returns:
            LLMResponse object
        """

        text_content = ""
        thinking_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text

            elif block.type == "thinking":
                thinking_content = block.thinking

            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                     id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=block.input,
                        ),
                )
            )



        usage = None
        if hasattr(response,"usage") and response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                total_tokens=(response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            )


        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )



    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from Anthropic API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            LLMResponse object
        """

        request_params = self._prepare_request(messages, tools)

        if self.retry_config_enabled:
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call =  retry_decorator(self._make_api_request)
            response = await  api_call(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
            )

        else:
             response = await self._make_api_request(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
            )



        return self._parse_response(response)

