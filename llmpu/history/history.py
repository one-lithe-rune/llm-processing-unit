import json

from pathlib import Path
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


def load_history(file_path: Path) -> list[HistoryTurn]:
    """
    loads data from a JSON file into the history. Answers
    a empty list if the requested file does not exist
    """
    result: list[HistoryTurn] = list()

    if Path(file_path).exists():
        with open(file_path) as file:
            result = json.load(file, cls=HistoryJSONDecoder)

        print(f"loaded: {file_path}")
    else:
        print(f"not found: {file_path}")

    return result


def save_history(file_path, turns: list[HistoryTurn]):
    """
    saves the history data to a JSON file, overwriting the file
    if it already exists.
    """

    with open(file_path, mode="w+") as file:
        json.dump(turns, file, indent=4, cls=HistoryJSONEncoder)
