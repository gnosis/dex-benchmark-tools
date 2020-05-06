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
instances_prefix = 'data/mainnet/standard-solver/instances/'
results_prefix = 'data/mainnet/standard-solver/results/'
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
    l |= {r['Key'] for r in page['Contents']}
    return l
    while page['IsTruncated']:
        page = client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=page_size,
            ContinuationToken=page['NextContinuationToken']
        )
        l |= {r['Key'] for r in page['Contents']}
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
    fraction_infeasible: float,
    selection_criteria: str,
    result_paths: Path
):
    # select all feasible instances
    feasible = set()
    for name, path in instances(result_paths):
        print(name, path)
        try:
            instance = load_json(path)
        except json.JSONDecodeError:
            continue  # ignore empty files and other unparsable garbage
        if len(instance['orders']) > 0:
            feasible.add(name)

    feasible = {get_instance_batch(instance) for instance in feasible}

    # select subset of feasible instances
    nr_feasible = min(len(feasible), max_nr_instances * (1 - fraction_infeasible))

    return select_batches(feasible, nr_feasible, selection_criteria)


def filter_unique(instances_path: Path):
    to_remove = set()
    for (name1, path1), (name2, path2) \
            in tqdm(combinations(instances(instances_path), 2)):
        if path1 in to_remove or path2 in to_remove:
            continue
        if filecmp.cmp(path1, path2):
            to_remove.add(path2)
    print("removing:", to_remove)
    for path in to_remove:
        os.remove(path)


def main(args):
    client = boto3.client('s3')

    print("Downloading solution list.")
    result_paths = get_instance_list(client, prefix=results_prefix)
    if args.from_date is not None:
        result_paths = filter_instances_by_date(result_paths, args.from_date)

    batches = {get_instance_batch(result_path) for result_path in result_paths}

    result_dir = tempfile.TemporaryDirectory()

    print("Downloading solutions.")
    download_instances(client, result_paths, Path(result_dir.name), get_result_name)

    print("Selecting feasible instances.")
    feasible_batches = select_feasible_batches(
        args.max_nr_instances,
        args.fraction_infeasible,
        args.selection_criteria,
        Path(result_dir.name)
    )

    print("Downloading problem list.")
    instance_paths = get_instance_list(client, prefix=instances_prefix)
    if args.from_date is not None:
        instance_paths = filter_instances_by_date(instance_paths, args.from_date)

    print("Downloading feasible problems.")
    feasible_instance_paths = filter_instances_by_batch(instance_paths, feasible_batches)
    download_instances(
        client, feasible_instance_paths, Path(args.instances_path), get_instance_name
    )

    print("Selecting infeasible instances.")
    infeasible_batches = batches - feasible_batches
    nr_infeasible = int(
        (len(feasible_batches) * args.fraction_infeasible)
        / (1 - args.fraction_infeasible)
    )
    print(len(feasible_batches))
    print("nr_infeasible", nr_infeasible)
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
        '--fraction_infeasible',
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
    parser.set_defaults(exec_subcommand=main)
