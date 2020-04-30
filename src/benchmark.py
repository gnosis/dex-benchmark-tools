import argparse
import logging

from .compare import setup_arg_parser as setup_compare_arg_parser
from .extract import setup_arg_parser as setup_extract_arg_parser
from .update_oracle import setup_arg_parser as setup_update_oracle_arg_parser
from .download import setup_arg_parser as setup_download_arg_parser

logger = logging.getLogger("benchmark")


if __name__ == "__main__":
    """
    Main function.
    """

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Benchmarking toolbox."
    )
    subparsers = parser.add_subparsers(
        title="subcommand",
        description="valid subcommands",
        help="run `subcommand --help` for help on a subcommand"
    )

    setup_download_arg_parser(subparsers)

    setup_extract_arg_parser(subparsers)

    setup_compare_arg_parser(subparsers)

    setup_update_oracle_arg_parser(subparsers)

    args = parser.parse_args()

    args.exec_subcommand(args)
