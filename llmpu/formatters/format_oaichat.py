from .base import BaseSessionFormatter
from llmpu.history import HistoryTurn

# See https://github.com/tatsu-lab/stanford_alpaca


class OAIChatSessionFormatter(BaseSessionFormatter):
    """
    Class to rewrite a history list to be in OpenAI chat format
    """

    def apply(self, turns: list[HistoryTurn]):
        return [{"role": turn.role, "content": turn.content} for turn in turns]
