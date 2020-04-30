import argparse
import os
from pathlib import Path

from .util import BenchmarkResults, Results, instances, load_json


def extract_instance_results(
    instance_name: str,
    problem_path: Path,
    solution_path: Path
) -> BenchmarkResults:
    problem = load_json(problem_path)
    solution = load_json(solution_path)

    if "metadata" in solution.keys() and\
       "description" in solution["metadata"].keys():
        description = solution["metadata"]["description"]
    else:
        description = None

    fee = solution["fee"]

    u = int(solution["objVals"]["utility"])
    v = int(solution["objVals"]["volume"])
    fees = int(solution["objVals"]["fees"])
    dua = int(solution["objVals"]["utility_disreg"])
    dut = int(solution["objVals"]["utility_disreg_touched"])

    obj_val = u - dua + fees // 2
    true_obj_val = u - dut + fees // 2
    runtime = solution["solver"]["runtime"]
    nr_exec_orders = len(solution["orders"])
    nr_orders = len(problem["orders"])

    # FIXME: this should not be necessary once we have a proper exit status
    # as part of the solution json.
    time_limit = 180

    results = BenchmarkResults(
        runtime=runtime,
        obj_val=obj_val,
        nr_exec_orders=nr_exec_orders,
        nr_orders=nr_orders,
        u=u,
        v=v,
        dua=dua,
        dut=dut,
        fee=fee,
        true_obj_val=true_obj_val,
        solution_path=str(solution_path),
        instance_name=instance_name,
        description=description,
        time_limit=time_limit
    )
    return results


def extract(problems_path: Path, solutions_path: Path) -> Results:
    instance_name_to_problem_path = {
        instance_name: problem_path
        for instance_name, problem_path in instances(problems_path)
    }
    return Results(benchmark_results=[
        extract_instance_results(
            instance_name,
            instance_name_to_problem_path[instance_name],
            solution_path
        )
        for instance_name, solution_path in instances(solutions_path)
    ])


def main(args):
    results = extract(args.problems_path, args.solutions_path)
    results.to_file(args.results_path)
    if args.oracle_path is not None:
        from .update_oracle import main as update_oracle
        update_oracle(args)


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid directory")


def setup_arg_parser(subparsers):
    parser = subparsers.add_parser(
        'extract',
        help="Extracts relevant results and metrics from a set of solution files."
    )

    parser.add_argument(
        'problems-path',
        type=dir_path,
        help='Directory containing the problem files.'
    )
    parser.add_argument(
        'solutions-path',
        type=dir_path,
        help='Directory containing the solution files.'
    )
    parser.add_argument(
        'results-path',
        type=Path,
        help='File to dump results to.'
    )
    parser.add_argument(
        '--oracle-path',
        type=Path,
        help='If given then also update the given oracle file with the extracted results.'
    )
    parser.set_defaults(exec_subcommand=main)
