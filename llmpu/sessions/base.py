import requests

from abc import ABC, abstractmethod
from urllib.parse import urljoin

from llmpu.history import HistoryTurn
from llmpu.formatters import BaseSessionFormatter

Jsonable = dict[str, "Jsonable"] | list["Jsonable"] | str | int | float | bool | None


class BaseSession(ABC):
    """
    Base class for retrieving responses from AI provider endpoint
    """

    def __init__(
        self,
        host: str,
        path: str,
        initial_processors: list[BaseSessionFormatter] = None,
        token_limit: int = 1024,
        extra_props: dict = None,
    ):
        self._session: requests.Session = requests.Session()
        self._endpoint = urljoin(host, path)
        self._token_limit = token_limit
        self._extra_props = extra_props if extra_props is not None else dict()
        self._last_response: Jsonable = None

        self.processors = (
            initial_processors if initial_processors is not None else list()
        )

    @property
    def processors(self) -> list[BaseSessionFormatter]:
        return self._processors

    @processors.setter
    def processors(
        self, values: list[BaseSessionFormatter | type[BaseSessionFormatter]]
    ):
        # allow both instance and class forms to be passed in as a convenience
        self._processors = [
            formatter() if isinstance(formatter, type) else formatter
            for formatter in values
        ]

    @property
    def token_limit(self) -> int:
        return self._token_limit

    @token_limit.setter
    def token_limit(self, value: int):
        self._token_limit = value

    @property
    def last_response(self) -> Jsonable:
        return self._last_response

    @abstractmethod
    def get_response(self, context: str | list[HistoryTurn]) -> dict[str, str]:
        pass
