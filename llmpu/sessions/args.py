from argparse import ArgumentParser, Namespace

from .base import BaseSession
from .oai_compatible import OAICompatibleChatSession


def add_args(
    parser: ArgumentParser,
    default_host="http://localhost:5001",
    default_type="openai_compatible",
    default_model=None,
    default_api_key=None,
    default_api_org=None,
    default_api_proj=None,
):
    """
    Add command line arguments to an argument parser, allowing a
    session connecting to an AI server to be configured
    """
    parser.add_argument(
        "--ai-host",
        default=default_host,
        help="host name or ip address of the AI server, include port if applicable",
    )
    parser.add_argument(
        "--ai-session-type",
        choices=["openai_compatible"],
        default=default_type,
        help="Which session type to use use when connecting to the ai server",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default=default_model,
        help="api key for the AI server, if applicable",
    )
    parser.add_argument(
        "--ai-api-key",
        type=str,
        default=default_api_key,
        help="api key for the AI server, if applicable",
    )
    parser.add_argument(
        "--ai-api-org",
        type=str,
        default=default_api_org,
        help="organization to pass to the AI server api, if applicable",
    )
    parser.add_argument(
        "--ai-api-project",
        type=str,
        default=default_api_proj,
        help="project to pass to the AI server api, if applicable",
    )


def from_args(args: Namespace) -> BaseSession:
    """
    Answer a Session class based on the passed arguments
    """
    return {"openai_compatible": OAICompatibleChatSession}[args.ai_session_type](
        host=args.ai_host,
        api_key=args.ai_api_key,
        api_org=args.ai_api_org,
        api_proj=args.ai_api_project,
        model=args.ai_model,
    )
