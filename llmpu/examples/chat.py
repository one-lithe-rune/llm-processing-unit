"""
A quick and dirty example of a LLM chat client built using the
LlmProcessingUnit.

To run from the projec top level make sure you have activated your
.venv and then do `python -m llmpu.examples.chat`
"""

import readline  # noqa: F401
import sys

import llmpu.sessions as sessions
import llmpu.formatters as formatters

from llmpu import LlmProcessingUnit

# some basic terminal control codes
TERM_GREEN_UNDERLINE = "\033[32;1;4m"
TERM_BLUE_UNDERLINE = "\034[32;1;4m"
TERM_DEFAULT = "\033[0m"
TERM_ERASE_LINE = "\33[2K\r"
TERM_CLEAR = "\033[2J\033[1;1H"


def process_slashcommand(llm: LlmProcessingUnit, input: str):
    """
    Check for various slash commands in the input and print out
    relevant results if they are
    """

    if input == "/exit":
        sys.exit(0)

    if input == "/clear":
        sys.stdout.write(TERM_CLEAR)
        sys.stdout.flush()

    if input == "/system":
        prompt = llm.read_sys().content if llm.read_sys() else "None"
        print(f"\nSystem prompt is: {prompt}\n")
    elif input.startswith("/system "):
        llm.load_sys(input.split(" ", 1)[1])
        prompt = llm.read_sys().content if llm.read_sys() else "None"
        print(f"\nSystem prompt is now: {prompt}\n")

    if input == "/dump memory":
        print(f"\n{llm._memory}\n")

    if input == "/dump registers":
        print(f"\n{llm._registers}\n")


def chat(endpoint: sessions.BaseSession):

    llm = LlmProcessingUnit(endpoint)
    TRANSCRIPT_MEMSLOT = ["ChatTranscript0"]

    print("LLMpu example: Simple chat client\n")
    print("Type /exit to close\n")

    while True:
        # get the user's input to be sent to the LLM
        user_input: str = input(f"{TERM_GREEN_UNDERLINE}User:{TERM_DEFAULT} ")

        # if they entered slash command then do that instead
        if user_input.startswith("/"):
            process_slashcommand(llm, user_input)
            continue

        # load up the instruction we want to evaluate
        llm.load_ins(user_input)
        print("...", end="", flush=True)

        # The registers used to build the context to send are the
        # defaults, I'm just explicitly setting them here to make
        # things more obvious
        llm.evaluate(["system", "context0", "instruction"])
        print(f"{TERM_ERASE_LINE}", end="")
        print(
            f"{TERM_GREEN_UNDERLINE}{llm.read_result().role}:{TERM_DEFAULT} {llm.read_result().content}\n"
        )

        # update the memory slot we are storing the chat in
        llm.push("instruction", TRANSCRIPT_MEMSLOT)
        llm.push("result", TRANSCRIPT_MEMSLOT)

        # load that back into the context ready for the next round
        llm.load_context(0, TRANSCRIPT_MEMSLOT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    sessions.add_args(parser)
    formatters.add_args(parser)
    args = parser.parse_args()

    session = sessions.from_args(args)
    session.processors = [formatters.from_args(args)]

    chat(session)
