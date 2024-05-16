from llmpu.history import HistoryTurn


class BaseSessionFormatter:
    """
    Base class for session processors, implements an active nothing processor
    whose apply method simply returns the input session unaltered.
    """

    @property
    def uses_characters(self):
        return False

    def apply(self, history: list[HistoryTurn]) -> list:
        return history
