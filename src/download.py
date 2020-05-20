import filecmp
import json
import logging
import os
import random
import re
import tempfile
from collections import namedtuple
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional, Set

import boto3
from tqdm import tqdm

from .util import Oracle, Results, instances, load_json

bucket = 'gnosis-dev-dfusion'
page_size = 1000


def get_instance_list(
    client,
    prefix: str
):
    """Return set of filename is S3 bucket."""
    l = set()
    page = client.list_objects_v2(
        Bucket=bucket, Prefix=prefix, MaxKeys=page_size
    )
    l |= {r['Key'] for r in page['Contents'] if r['Key'][-4:] == 'json'}

    while page['IsTruncated']:
        page = client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=page_size,
            ContinuationToken=page['NextContinuationToken']
        )
        l |= {r['Key'] for r in page['Contents'] if r['Key'][-4:] == 'json'}
    return l


def get_instance_date(instance_path: str):
    return re.search(r'\/([0-9\-]+)', instance_path).group(1)


def get_instance_batch(instance_path: str):
    return re.search(r'instance_([0-9]+)', instance_path).group(1)


def filter_instances_by_date(instance_paths, from_date):
    instance_paths = {
        instance_path
        for instance_path in instance_paths
        if get_instance_date(instance_path) >= from_date
    }
    return instance_paths


def filter_instances_by_batch(instance_paths, batches):
    instance_paths = {
        instance_path
        for instance_path in instance_paths
        if get_instance_batch(instance_path) in batches
    }
    return instance_paths


def get_instance_name(instance_path: str):
    return Path(instance_path).name


def get_result_name(result_path: str):
    return Path(result_path).parent.name


def download_instance(
    client,
    path: str,
    output_dir: Path,
    name_extractor: Callable[[str], str]
):
    instance_name = name_extractor(path)
    local_path = str(output_dir / instance_name)
    client.download_file(Bucket=bucket, Key=path, Filename=local_path)


def download_instances(
    client,
    instance_paths: Set[str],
    output_dir: Path,
    name_extractor: Callable[[str], str]
):
    for instance_path in tqdm(instance_paths):
        download_instance(client, instance_path, output_dir, name_extractor)


def select_batches(
    batches: Set[str],
    n: int,
    selection_criteria: str
):
    if selection_criteria == 'most_recent':
        batches = sorted(list(batches), key=int, reverse=True)
        batches = set(batches[:n])
    elif selection_criteria == 'random':
        batches = set(random.sample(batches, n))
    else:
        assert False
    return batches


def select_feasible_batches(
    max_nr_instances: int,
    max_fraction_infeasible: float,
    selection_criteria: str,
    result_paths: Path
):
    # select all feasible instances
    feasible = set()
    for name, path in instances(result_paths):
        try:
            instance = load_json(path)
        except json.JSONDecodeError:
            continue  # ignore empty files and other unparsable garbage
        if len(instance['orders']) > 0:
            feasible.add(name)

    feasible = {get_instance_batch(instance) for instance in feasible}

    # select subset of feasible instances
    nr_feasible = min(len(feasible), int(max_nr_instances * (1 - max_fraction_infeasible)))

    return select_batches(feasible, nr_feasible, selection_criteria)


def filter_unique(instances_path: Path):
    to_remove = set()
    for (name1, path1), (name2, path2) \
            in tqdm(combinations(instances(instances_path), 2)):
        if path1 in to_remove or path2 in to_remove:
            continue
        if filecmp.cmp(path1, path2):
            to_remove.add(path2)

    for path in to_remove:
        os.remove(path)


def get_batch_ids_from_file(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


def filter_paths(paths, args):
    if args.from_date is not None:
        paths = filter_instances_by_date(paths, args.from_date)
    if args.with_batch_ids_from_file is not None:
        paths = filter_instances_by_batch(
            paths,
            get_batch_ids_from_file(args.with_batch_ids_from_file)
        )
    if args.with_batch_ids is not None:
        paths = filter_instances_by_batch(
            paths,
            args.with_batch_ids
        )
    return paths


def main(args):
    if args.max_fraction_infeasible != 1 and (
        args.with_batch_ids is not None
        or args.with_batch_ids_from_file is not None
    ):
        print("WARNING: '--max_fraction_infeasible' ignored when '--with_batch_ids' is given.")
        args.max_fraction_infeasible = 1

    instances_prefix = f'data/{args.network}/{args.solver}/instances/'
    results_prefix = f'data/{args.network}/{args.solver}/results/'

    client = boto3.client('s3')

    print("Downloading solution list.")
    result_paths = get_instance_list(client, prefix=results_prefix)
    result_paths = filter_paths(result_paths, args)

    batches = {get_instance_batch(result_path) for result_path in result_paths}

    result_dir = tempfile.TemporaryDirectory()

    print("Downloading solutions.")
    download_instances(client, result_paths, Path(result_dir.name), get_result_name)

    print("Selecting feasible instances.")
    feasible_batches = select_feasible_batches(
        args.max_nr_instances,
        args.max_fraction_infeasible,
        args.selection_criteria,
        Path(result_dir.name)
    )

    print("Downloading problem list.")
    instance_paths = get_instance_list(client, prefix=instances_prefix)
    instance_paths = filter_paths(instance_paths, args)

    print("Downloading feasible problems.")
    feasible_instance_paths = filter_instances_by_batch(instance_paths, feasible_batches)
    download_instances(
        client, feasible_instance_paths, Path(args.instances_path), get_instance_name
    )

    print("Selecting infeasible instances.")
    infeasible_batches = batches - feasible_batches
    if args.max_fraction_infeasible != 1:
        nr_infeasible = int(
            (len(feasible_batches) * args.max_fraction_infeasible)
            / (1 - args.max_fraction_infeasible)
        )
    else:
        nr_infeasible = len(infeasible_batches)

    infeasible_batches = select_batches(
        infeasible_batches,
        nr_infeasible,
        args.selection_criteria
    )

    print("Downloading infeasible problems.")
    infeasible_instance_paths = filter_instances_by_batch(
        instance_paths, infeasible_batches
    )
    download_instances(
        client, infeasible_instance_paths, Path(args.instances_path), get_instance_name
    )

    print("Removing duplicate problems.")
    filter_unique(Path(args.instances_path))


def setup_arg_parser(subparsers):
    parser = subparsers.add_parser('download', help="Downloads instances from S3 bucket.")

    parser.add_argument(
        'instances_path',
        type=Path,
        help='Directory where to copy instances to.'
    )
    parser.add_argument(
        '--from_date',
        type=str,
        help='Get instances newer than the given date (inclusive).'
    )
    parser.add_argument(
        '--max_nr_instances',
        type=int,
        default=100,
        help='Maximum number of instances to download.'
    )
    parser.add_argument(
        '--max_fraction_infeasible',
        type=float,
        default=1 / 3,
        help='Instances should have at most given fraction of infeasible instances.'
    )
    parser.add_argument(
        '--selection_criteria',
        type=str,
        choices=['most_recent', 'random'],
        default='random',
        help='How to select instances.'
    )
    parser.add_argument(
        '--with_batch_ids',
        type=str,
        nargs='+',
        help='Get instances with given batch ids (separated by spaces).'
    )
    parser.add_argument(
        '--with_batch_ids_from_file',
        type=str,
        help='Get instances with batch ids given in specified file (one per line).'
    )
    parser.add_argument(
        '--network',
        type=str,
        choices=['mainnet', 'mainnet-prod', 'rinkeby'],
        default='mainnet',
        help='Network where to pull instances from.'
    )
    parser.add_argument(
        '--solver',
        type=str,
        choices=['standard-solver', 'fallback-solver', 'open-solver'],
        default='standard-solver',
        help='Solver where to pull solution instances from.'
    )
    parser.set_defaults(exec_subcommand=main)
