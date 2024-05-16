import json

from pathlib import Path
from typing import Self

from llmpu.sessions import BaseSession
from llmpu.history import HistoryTurn, HistoryJSONEncoder, HistoryJSONDecoder


class LlmProcessingUnit:
    """
    A class that treats an instruct trained LLM as if it were somewhat
    analogous to the Arthimetic and Logic Unit of a very simple 8-Bit style
    processor of yore.

    'Registers' are provided for the System Prompt, Instruction, last Result,
    and a configurable number of Contexts (defaults to 3).

    'Memory' locations are represented as paths through a dictionary, with
    leaf nodes storing LIFO stacks of 'turns' (strings sent or recieved
    from the llm, with a role attached) that can be pushed to, or popped from.

    The Context registers can be directly loaded from a memory location
    so also contain a stack of turns. Other registers usually only contain
    one turn and can be pushed to or popped from memory locations but
    unlike the Context registers cannot directly load a whole memory
    location stack.

    The entire memory dictionary can be saved and loaded from a file with
    'save_mem' and 'load_mem' methods.

    Finally 'evaluate' is used to send the turns in the selected registers
    to the LLM. The LLM's response is then placed as turn in the Result
    register.
    """

    def __init__(
        self,
        session: BaseSession,
        memory: dict[str, HistoryTurn | list[HistoryTurn]] = None,
        context_registers: int = 3,
    ):
        self._session: BaseSession = session
        self._context_registers = context_registers
        self._registers: dict[str, str | list[str]] = {
            "system": [],
            "instruction": [],
            "result": [],
        }
        for idx in range(context_registers):
            self._registers[f"context{idx}"] = None

        # TODO typing is not correct here needs to be recursive. See
        # Jsonable in session.base for an example
        self._memory: dict[str, str | list[str]] = dict() if memory is None else memory

    def _get_mem_location(self, path: list[str, int]):
        current = self._memory
        for key in path:
            if key in current:
                current = current[key]
            else:
                raise KeyError(f"Invalid memory path: '{path}'")

        return current

    def load_sys(self, value):
        """
        Load the passed value into the system prompt register as a turn.
        """
        self._registers["system"] = [HistoryTurn(role="system", content=value)]
        return self

    def load_ins(self, value):
        """
        Load the passed value into the instruction register as a turn.
        """
        self._registers["instruction"] = [HistoryTurn(role="user", content=value)]
        return self

    def load_context(self, reg_idx: int, mem_path: list[str | int]):
        """
        Load the list of turns at a the passed memory dictionary location
        into the context register at passed index.
        """
        if reg_idx not in range(self._context_registers):
            raise ValueError(f"Unknown register 'context{reg_idx}'")

        self._registers[f"context{reg_idx}"] = self._get_mem_location(mem_path)
        return self

    def read_sys(self) -> HistoryTurn:
        """
        Answer the current value of the system register
        """
        return self._registers["system"][0] if self._registers["system"] else None

    def read_ins(self) -> HistoryTurn:
        """
        Answer the current value of the instruction register
        """
        return (
            self._registers["instruction"][0]
            if self._registers["instruction"]
            else None
        )

    def read_context(self, reg_idx: int) -> list[HistoryTurn]:
        """
        Answer the current value of the Context register at idx
        """
        if reg_idx not in range(self._context_registers):
            raise ValueError(f"Unknown register 'context{reg_idx}'")

        return self._registers[f"context{0}"]

    def read_result(self) -> HistoryTurn:
        """
        Answer the current contents of result register
        """
        return self._registers["result"][0] if self._registers["result"] else None

    def push(self, register: str, mem_path: list[str]):
        """
        Push any turns in a register onto the end of the list of turns
        at the passed memory dictionary location. If the location
        does not yet exist that location will be created.
        """

        if register not in self._registers:
            raise ValueError(f"Unknown register '{register}'")

        current = self._memory
        for key in mem_path[:-1]:
            if key is not None:
                if key not in current:
                    current[key] = dict()
                current = current[key]

        leaf_value = current.get(mem_path[-1], list())
        if isinstance(leaf_value, list):
            current[mem_path[-1]] = leaf_value + self._registers[register]
        else:
            raise ValueError(f"Invalid memory location for push: {mem_path}")

        return self

    def pop(self, mem_path: list[str], register: str):
        """
        Pop the last list turn at a memory dictionary location into the
        passed register, removing it from the list in the memory location.
        Any existing turns in the register will be replaced with the
        popped turn.
        """

        if register not in self._registers:
            raise ValueError(f"Unknown register {register}")

        leaf_key = mem_path[-1]
        mem_parent = self._get_mem_location(mem_path[:-1])
        mem_value = mem_parent[leaf_key]
        if not isinstance(mem_value, list):
            raise ValueError(f"Invalid memory location for pop: {mem_path}")

        self._registers[register] = mem_value.pop()

        return self

    def clear_reg(self, register: str):
        """
        Clear the contents of the passed register
        """
        if register not in self._registers:
            raise ValueError(f"Unknown register '{register}'")

        self._registers[register] = None
        return self

    def clear_mem(self, mem_path: list[str]):
        """
        Clear the memory dictionary location specified by the passed
        path list.
        """

        leaf_key = mem_path[-1]
        mem_parent = self._get_mem_location(mem_path[:-1])
        mem_parent.pop(leaf_key, None)
        return self

    def evaluate(
        self, registers: list[str] = ["system", "context0", "instruction"]
    ) -> Self:
        """
        Sends a request to the LLM to evaluate a context built from the passed
        registers, placing the answer in the the 'result' register.
        """

        full_context: list[HistoryTurn] = [
            turn
            for register in registers
            if self._registers[register] is not None
            for turn in self._registers[register]
        ]

        self._registers["result"] = [
            HistoryTurn(
                **self._session.get_response(full_context),
            )
        ]
        return self

    def load_mem(self, file_path: Path | str) -> Self:
        """
        loads the memory from a JSON file
        """

        if Path(file_path).exists():
            with open(file_path) as file:
                self._memory = json.load(file, cls=HistoryJSONDecoder)

            print(f"loaded: {file_path}")
        else:
            print(f"not found: {file_path}")

        return self

    def save_mem(self, file_path: Path | str) -> Self:
        """
        saves the memory to a JSON file
        """

        with open(file_path, mode="w+") as file:
            json.dump(self._memory, file, indent=4, cls=HistoryJSONEncoder)

        return self
