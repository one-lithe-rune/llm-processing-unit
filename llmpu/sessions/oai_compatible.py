import requests
from urllib.parse import urljoin

from llmpu.formatters import BaseSessionFormatter
from llmpu.history import HistoryTurn
from .base import BaseSession, Jsonable


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
        history: list = None,
        api_key: str = None,
    ):
        self._session: requests.Session = requests.Session()
        if api_key is not None:
            self._session.headers = {"Authorization", f"Bearer {api_key}"}
        self._endpoint: str = urljoin(host, path)
        self._token_limit: int = token_limit
        self._extra_props: dict[str, str] = (
            extra_props if extra_props is not None else dict()
        )
        self._api_key: str = api_key
        self._last_response: Jsonable = None

        self.history = history if history else []
        self.processors = (
            initial_processors if initial_processors is not None else list()
        )

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
