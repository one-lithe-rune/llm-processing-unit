from .base import BaseSessionFormatter
from llmpu.history import HistoryTurn

# See https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/


class Llama3SessionFormatter(BaseSessionFormatter):
    """
    Class to rewrite a history list to be in Llama3 Base format
    """

    @property
    def stop_words():
        return ["<|end_of_text|>"]

    def apply(self, history: list[HistoryTurn]):
        content = "".join(
            [turn.content for round in history.included for turn in round.turns]
        )

        return [{"role": "user", "content": f"<|begin_of_text|>{content}"}]


class Llama3InstructSessionFormatter(BaseSessionFormatter):
    def __init__(self):
        self._role_formats = {
            "system": "system",
            "user": "user",
            "input": "user",
            "assistant": "assistant",
        }

    """
    Class to rewrite a session to be in Llama3 chat format
    """

    @property
    def stop_words():
        return ["<|eot_id|>", "<|end_of_text|>"]

    def apply(self, turns: list[HistoryTurn]):
        # Llama3-ify the current history for chat completion
        result = [
            {
                "role": turn.role,
                "content": (
                    f"<|start_header_id|>{self._role_formats[turn.role]}<|end_header_id|>\n"
                    f"\n{turn.content}<|eot_id|>"
                ),
            }
            for turn in turns
        ]

        # always adds an open assistant entry at the end to set up a response
        result.append(
            {
                "role": "assistant",
                "content": f"<|start_header_id|>{self._role_formats['assistant']}<|end_header_id|>",
            }
        )

        # always put BOS before the first content
        result[0]["content"] = f"<|begin_of_text|>{result[0]['content']}"

        return result


class Llama3ChatSessionFormatter(Llama3InstructSessionFormatter):
    def __init__(self):
        self._role_formats = {
            "system": "system",
            "user": "user({0})",
            "input": "user({0})",
            "assistant": "assistant({0})",
        }

    @property
    def uses_characters(self):
        return True

    def set_characters(self, instruct_char: str = "Alice", assist_char: str = "Bob"):
        self._role_formats["user"] = self._role_formats["user"].format(instruct_char)
        self._role_formats["assistant"] = self._role_formats["assistant"].format(
            assist_char
        )
