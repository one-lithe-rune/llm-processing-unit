import json

from dataclasses import dataclass


@dataclass
class HistoryTurn:
    role: str
    content: str

    def clone(self):
        return HistoryTurn(self.role, self.content)


class HistoryJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, HistoryTurn):
            return obj.__dict__
        return super().default(obj)


class HistoryJSONDecoder(json.JSONDecoder):
    def __init__(self):
        super().__init__(object_hook=self.object_hook)

    def object_hook(self, obj):
        if "role" in obj and "content" in obj:
            return HistoryTurn(**obj)
        return obj
