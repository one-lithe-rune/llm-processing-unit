import requests

from llmpu.formatters import BaseSessionFormatter
from llmpu.history import HistoryTurn
from .base import BaseSession


class OAICompatibleChatSession(BaseSession):
    """
    A session using a OpenAI Chat completions compatible endpoint
    """

    def __init__(
        self,
        host: str,
        path: str = "/v1/chat/completions",
        initial_processors: list[BaseSessionFormatter] = None,
        token_limit: int = 1024,
        extra_props: dict = None,
        api_key: str = None,
    ):
        super().__init__(host, path, initial_processors, token_limit, extra_props)
        if api_key is not None:
            self._session.headers = {"Authorization", f"Bearer {api_key}"}
        self._api_key: str = api_key

    def close(self):
        self._session.close()

    def get_response(
        self, context: list[HistoryTurn], token_limit=None
    ) -> dict[str, str]:
        # do any preprocessing of what we're going to send
        for processor in self._processors:
            request_context = processor.apply(context)

        # Send the final request to AI chat server
        response: requests.Response = self._session.post(
            self._endpoint,
            json=self._extra_props
            | {
                "messages": request_context,
                "max_tokens": self._token_limit if token_limit is None else token_limit,
            },
        )

        self._last_response = response.json()
        return self._last_response["choices"][0]["message"]
