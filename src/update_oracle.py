import argparse
from .util import Results, Oracle
from pathlib import Path


def main(args):
    results = Results.from_file(args.results_path)
    if Path(args.oracle_path).exists():
        oracle = Oracle.from_file(args.oracle_path)
    else:
        oracle = Oracle()
    for benchmark in results.benchmark_results:
        oracle.update(benchmark)
    oracle.to_file(args.oracle_path)


def setup_arg_parser(subparsers):
    parser = subparsers.add_parser('update-oracle', help="Updates oracle from result files.")

    parser.add_argument(
        'results-path',
        type=Path,
        help='File containing the results file to update the oracle with.'
    )
    parser.add_argument(
        'oracle-path',
        type=Path,
        help='Oracle file to update (will be created if it does not exist)'
    )
    parser.set_defaults(exec_subcommand=main)
