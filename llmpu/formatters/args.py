from argparse import ArgumentParser, Namespace

from .base import BaseSessionFormatter
from .format_alpaca import AlpacaSessionFormatter
from .format_llama3 import (
    Llama3InstructSessionFormater,
)
from .format_oaichat import OAIChatSessionFormatter


def add_args(parser: ArgumentParser, default_prompt="alpaca"):
    """
    Add command line arguments to an argument parser for selection
    of a SessionFormatter
    """
    parser.add_argument(
        "-p",
        "--ai-prompt-format",
        choices=["alpaca", "llama3", "oaichat"],
        type=str,
        default=default_prompt,
        help="prompt format to use when communicating with the ai server",
    )


def from_args(args: Namespace) -> BaseSessionFormatter:
    """
    Answer a SessionFormatter class determined by appropriate
    attributes in the passed argsparse Namespace
    """
    return {
        "alpaca": AlpacaSessionFormatter,
        "llama3": Llama3InstructSessionFormater,
        "oaichat": OAIChatSessionFormatter,
    }[args.ai_prompt_format]
