import logging

from ..retry import RetryConfig
from ..schema import LLMProvider, LLMResponse, Message
from .anthropic_client import AnthropicClient
from .base import LLMClientBase
from .openai_client import OpenAIClient


logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client wrapper supporting multiple providers.

    This class provides a unified interface for different LLM providers.
    It automatically instantiates the correct underlying client based on
    the provider parameter and appends the appropriate API endpoint suffix.

    Supported providers:
    - anthropic: Appends /anthropic to api_base
    - openai: Appends /v1 to api_base
    """


    def __init__(
        self,
        api_key: str,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        api_base: str = "https://api.minimaxi.com",
        model: str = "MiniMax-M2",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize LLM client with specified provider.

        Args:
            api_key: API key for authentication
            provider: LLM provider (anthropic or openai)
            api_base: Base URL for the API (default: https://api.minimaxi.com)
                     Will be automatically suffixed with /anthropic or /v1 based on provider
            model: Model name to use
            retry_config: Optional retry configuration
        """


        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.retry_config = retry_config or RetryConfig()

        api_base = api_base.replace("/anthropic", "")

        if provider == LLMProvider.ANTHROPIC:
             full_api_base = f"{api_base.rstrip('/')}/anthropic"

        if provider == LLMProvider.OPENAI:
            full_api_base = f"{api_base.rstrip('/')}/v1"

        self.api_base = full_api_base


        self._client :LLMClientBase

        if provider == LLMProvider.ANTHROPIC:
            self._client = AnthropicClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )

        elif provider == LLMProvider.OPENAI:
             self._client = OpenAIClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")



        logger.info(f"Initialized LLM client for provider: {provider}")



    
    @property
    def retry_callback(self):
         """Get retry callback."""
        return self._client.retry_callback


    @retry_callback.setter
    def retry_callback(self,value):
        """Set retry callback."""
        self._client.retry_callback = value


    
    async def generate(
        self,
        messages : list[Message],
        tools : list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            LLMResponse object
        """
        return await self._client.generate(messages, tools)

