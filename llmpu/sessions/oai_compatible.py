import requests

from llmpu.formatters import BaseSessionFormatter
from llmpu.history import HistoryTurn
from .base import BaseSession, SessionError


class OAISessionError(SessionError):
    pass


class OAICompatibleChatSession(BaseSession):
    """
    A session using an OpenAI Chat completions compatible endpoint
    """

    def __init__(
        self,
        host: str,
        path: str = "/v1/chat/completions",
        initial_processors: list[BaseSessionFormatter] = None,
        token_limit: int = 1024,
        extra_props: dict = None,
        model: str = None,
        api_key: str = None,
        api_org: str = None,
        api_proj: str = None,
    ):
        super().__init__(host, path, initial_processors, token_limit, extra_props)

        self._session_headers: dict[str, str] = {}
        self._api_key: str = api_key
        self._api_org: str = api_org
        self._api_proj: str = api_proj
        self._model: str = model

        if api_key is not None:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        if api_org is not None:
            self._session.headers["OpenAI-Organization"] = api_org
        if api_proj is not None:
            self._session.headers["OpenAI-Project"] = api_proj
        if model is not None:
            self._extra_props["model"] = model

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
        try:
            response.raise_for_status()
            return self._last_response["choices"][0]["message"]
        except requests.exceptions.HTTPError:
            raise OAISessionError(self._last_response)
