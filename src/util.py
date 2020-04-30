import json
from inflection import underscore, camelize
import math
import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger("benchmark")


def load_json(instance_name):
    with open(instance_name, "r") as f:
        return json.load(f)


def dump_json(data, instance_name, **kwargs):
    with open(instance_name, "w+") as f:
        json.dump(data, f, **kwargs)


def instances(path: str):
    for root, dirs, files in os.walk(path):
        for instance_name in files:
            instance_path = Path(root) / instance_name
            yield instance_name, instance_path


class JSONSerializable:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except AttributeError:
                pass  # pass attributes that are available as properties

    @classmethod
    def from_dict(cls, d):
        d = {
            underscore(k): v for k, v in d.items()
        }
        return cls(**d)

    def to_dict(self, additional_properties=[]):
        self_as_dict = vars(self)
        self_as_dict.update({
            pty: getattr(self, pty) for pty in additional_properties
        })
        self_as_dict = {
            camelize(k, uppercase_first_letter=False): v for k, v in self_as_dict.items()
        }
        return self_as_dict

    @classmethod
    def from_file(cls, instance_name):
        return cls.from_dict(load_json(instance_name))

    def to_file(self, instance_name, additional_properties=[]):
        dump_json(
            self.to_dict(additional_properties),
            instance_name,
            indent=4,
            default=lambda x: str(x)
        )


class BenchmarkResults(JSONSerializable):
    eps = 1e-8
    computed_properties = [
        "proved_optimality",
        "proved_infeasibility",
        "proved_feasibility",
        "decided_feasibility"
    ]

    @property
    def proved_optimality(self):
        return self.runtime <= self.time_limit

    @property
    def proved_infeasibility(self):
        return self.runtime <= self.time_limit and self.obj_val <= self.eps

    @property
    def proved_feasibility(self):
        return self.obj_val > self.eps

    @property
    def decided_feasibility(self):
        return self.proved_feasibility or self.proved_infeasibility

    def to_dict(self, include_properties: bool = False):
        return super().to_dict(self.computed_properties if include_properties else [])


class BenchmarkOracle(JSONSerializable):
    def update(self, results: BenchmarkResults):
        if results.proved_feasibility and results.obj_val > self.obj_val:
            if self.proven_infeasible:
                error_str = f"Benchmark {results.instance_name} now proven feasible had "\
                    f"been proven infeasible before."
                logger.error(error_str)
                raise RuntimeError(error_str)

            self.obj_val = results.obj_val
            self.u = results.u
            self.v = results.v
            self.runtime = results.runtime
            self.proven_feasible = True
            self.nr_exec_orders = results.nr_exec_orders

        elif results.proved_infeasibility:
            if self.proven_feasible:
                error_str = f"Benchmark {results.instance_name} now proven infeasible had"
                f" been previously proven feasible before."
                logger.error(error_str)
                raise RuntimeError(error_str)

            self.runtime = results.runtime
            self.proven_infeasible = True
            self.nr_exec_orders = 0

        self.optimum_found = self.optimum_found or results.proved_optimality


class Results:
    def __init__(self, benchmark_results: List[BenchmarkResults]):
        self.benchmark_results = benchmark_results

    @classmethod
    def from_file(cls, instance_path: Path):
        d = load_json(instance_path)
        benchmark_results = []
        for b in d:
            b = {underscore(k): v for k, v in b.items()}
            b = BenchmarkResults.from_dict(b)
            benchmark_results.append(b)
        return cls(benchmark_results=benchmark_results)

    def to_file(self, instance_path: Path):
        d = [b.to_dict() for b in self.benchmark_results]
        dump_json(d, instance_path, indent=4, default=lambda x: str(x))


class Oracle(JSONSerializable):
    def __init__(self, benchmarks=dict()):
        self.benchmarks = benchmarks

    def update(self, results: BenchmarkResults):
        instance_name = results.instance_name
        if instance_name not in self.benchmarks.keys():
            self.benchmarks[instance_name] = BenchmarkOracle(
                proven_feasible=False,
                proven_infeasible=False,
                optimum_found=False,
                obj_val=0.0,
                v=0.0,
                runtime=math.inf
            )
        self.benchmarks[instance_name].update(results)

    @classmethod
    def from_file(cls, instance_name):
        d = load_json(instance_name)
        benchmarks = {
            k: BenchmarkOracle.from_dict(v) for k, v in d.items()
        }
        return cls(benchmarks)

    def to_file(self, instance_name):
        benchmarks_as_dict = {
            k: v.to_dict() for k, v in self.benchmarks.items()
        }
        dump_json(benchmarks_as_dict, instance_name, indent=4)
