from .base import BaseSessionFormatter
from llmpu.history import HistoryTurn

# See https://github.com/tatsu-lab/stanford_alpaca


class AlpacaSessionFormatter(BaseSessionFormatter):
    """
    Class to rewrite a history list to be in Alpaca format
    """

    @property
    def stop_words():
        return "### Instruction:\n"

    def apply(self, turns: list[HistoryTurn]):
        # Alpaca-ify session
        result = []

        for turn in turns:
            if turn.role == "user":
                result.append(
                    {
                        "role": "user",
                        "content": f"### Instruction:\n{turn.content}\n\n",
                    }
                )
            elif turn.role == "input":
                result.append(
                    {
                        "role": "input",
                        "content": f"### Input:\n{turn.content}\n\n",
                    }
                )
            else:
                result.append(
                    {
                        "role": "assistant",
                        "content": f"### Response:\n{turn.content}\n\n",
                    }
                )

        # if the last message is a user message then we also need it to include a
        # '### Response' line to trigger the response correctly
        if result[-1]["role"] in ["user", "input"]:
            result.append(
                {
                    "role": "assistant",
                    "content": "### Response:\n",
                }
            )

        return result
