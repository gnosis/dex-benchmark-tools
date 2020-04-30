import itertools
import math
import tempfile
import webbrowser
from functools import reduce
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from .util import Oracle, Results


def compute_short_names(s1: str, s2: str):
    return Path(s1).stem, Path(s2).stem


def build_benchmark_dataframe(results_id: str, results: Results):
    df = pd.DataFrame.from_records([
        b.to_dict(include_properties=True) for b in results.benchmark_results
    ])
    df["results_id"] = results_id
    return df


def build_oracle_dataframe(oracle: Oracle):
    df = pd.DataFrame.from_dict(
        {
            k: v.to_dict() for k, v in oracle.benchmarks.items()
        },
        orient="index"
    )
    return df


def format_float(f: float):
    return f"{f:.2f}"


def collect_benchmark_summary_html(df: pd.DataFrame):
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.groupby("group").agg({
        "objVal": "count",
        "nrOrders": ["mean", "max"],
        "runtimeOracle": ["mean", "max"],
        "provenFeasible": "mean",
        "provenInfeasible": "mean",
        "optimumFound": "mean"
    })

    df = df.rename(columns={
        "objVal": "instances",
        "runtimeOracle": "runtime"
    })
    return df.to_html(
        classes="table",
        float_format=format_float
    )


def collect_freq_table(df: pd.DataFrame):
    tdf = df.groupby("group")
    p = tdf.sum() / tdf.count()
    nr_results = df.shape[1]

    joint_ps = []
    for t in itertools.product(*[[False, True]] * nr_results):
        test = reduce(
            lambda a, i: a & (df[df.columns[i]] == t[i]),
            range(nr_results),
            True
        )
        joint_p = df[test].groupby("group").count() / tdf.count()
        joint_p = joint_p[df.columns[0]].rename(", ".join([
            ("+" if t[i] else "-") + df.columns[i] for i in range(nr_results)
        ]))
        joint_ps.append(joint_p)

    df = pd.concat([p] + joint_ps, axis=1).transpose().fillna(0)
    return df


def collect_proving_feasibility_table(df: pd.DataFrame):
    df = df[df.provenFeasible].pivot_table(
        columns="results_id",
        index=["instanceName", "group"],
        values="provedFeasibility",
        fill_value=False
    )
    return collect_freq_table(df)


def collect_deciding_feasibility_table(df: pd.DataFrame):
    df = df.pivot_table(
        columns="results_id",
        index=["instanceName", "group"],
        values="decidedFeasibility",
        fill_value=False
    )
    return collect_freq_table(df)


def collect_solution_quality_table(df: pd.DataFrame):
    df = df[df.provenFeasible]
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.pivot_table(
        columns="results_id",
        index=["instanceName", "objValOracle", "group"],
        values="objVal"
    )

    columns = df.columns
    df.reset_index("objValOracle", inplace=True)

    df = np.log(df[columns].div(df.objValOracle, axis=0))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.groupby("group").mean()
    return np.exp(df)


def make_cdf(s: pd.Series, pop_size: Optional[int] = None):
    if pop_size is None:
        pop_size = len(s)
    s = s.sort_values()
    s = pd.Series(range(1, 1 + len(s)), index=s.values)
    s /= pop_size
    return s


def make_multiple_cdf_plot(
    df: pd.DataFrame,
    group: str,
    extra_test: bool = True,
    ax=plt.gca()
):
    for results_id in df.results_id.unique():
        pop_size = df[(df.results_id == results_id) & (df.group == group)].shape[0]
        if pop_size == 0:
            continue
        s = make_cdf(
            df[(df.results_id == results_id) & (df.group == group) & extra_test].runtime,
            pop_size
        )
        s.name = results_id
        s.plot(ax=ax, title=group, label=results_id)


def make_optimality_plots(df: pd.DataFrame, cur_path: Path):
    groups = sorted(list(df.group.unique()))
    nb_rows = int(math.ceil(len(groups) / 5))
    nb_cols = min(5, len(groups))
    fig, axs = plt.subplots(nb_rows, nb_cols, sharey=True, figsize=(10, 2))
    for i, group in enumerate(groups):
        make_multiple_cdf_plot(
            df,
            group,
            extra_test=df.provedOptimality,
            ax=axs[i]
        )

    fig.text(
        0.5,
        -0.1,
        'Fraction of instances solved to optimality (yy) per time in seconds (xx)',
        ha='center'
    )
    plt.legend(loc='lower right')
    output_path = cur_path / "optimality.png"
    plt.savefig(output_path, bbox_inches="tight")
    return output_path


def add_grouping(df: pd.DataFrame, group_by: str):
    if group_by == "problem_size_quantiles":
        quantiles = df.nrOrders.quantile([1 / 3, 2 / 3])
        df["group"] = df.nrOrders.apply(
            lambda x: "small" if x < quantiles.iloc[0] else
                      "medium" if x < quantiles.iloc[1] else "large"
        )
    elif group_by == "solution_size_quantiles":
        quantiles = df.nrExecOrdersOracle.quantile([1 / 3, 2 / 3])
        df["group"] = df.nrExecOrdersOracle.apply(
            lambda x: "small" if x < quantiles.iloc[0] else
                      "medium" if x < quantiles.iloc[1] else "large"
        )
    elif group_by == "oracle_runtime_quantiles":
        quantiles = df.runtimeOracle.quantile([1 / 3, 2 / 3])
        df["group"] = df.runtimeOracle.apply(
            lambda x: "easy" if x < quantiles.iloc[0] else
                      "medium" if x < quantiles.iloc[1] else "hard"
        )
    elif group_by == "feasibility":
        df["group"] = df.apply(
            lambda x: "feasible" if x.provenFeasible else
                      "infeasible" if x.provenInfeasible else "unknown",
            axis=1
        )
    else:
        assert False
    return df


def main(args):
    results_1 = Results.from_file(args.solver_results_path_1)
    results_2 = Results.from_file(args.solver_results_path_2)

    oracle = Oracle.from_file(args.oracle_path)

    results_id1, results_id2 = compute_short_names(
        args.solver_results_path_1,
        args.solver_results_path_2
    )

    df1 = build_benchmark_dataframe(results_id1, results_1)
    df2 = build_benchmark_dataframe(results_id2, results_2)
    df = pd.concat([df1, df2])

    df_oracle = build_oracle_dataframe(oracle)

    df = df.join(df_oracle, on="instanceName", how="left", rsuffix="Oracle")

    df = add_grouping(df, args.group_by)
    benchmark_summary = collect_benchmark_summary_html(df)

    proving_feasibility_table = collect_proving_feasibility_table(df)

    deciding_feasibility_table = collect_deciding_feasibility_table(df)

    solution_quality_table = collect_solution_quality_table(df)

    cur_path = Path(__file__).parent
    optimality_plot_path = make_optimality_plots(df, cur_path)

    env = Environment(loader=FileSystemLoader(str(cur_path)))

    template = env.get_template("compare.html")
    template_vars = {
        "results_id1": results_id1,
        "results_id2": results_id2,
        "benchmark_summary": benchmark_summary,
        "proving_feasibility_table": proving_feasibility_table
        .to_html(classes="table", float_format=format_float),
        "deciding_feasibility_table": deciding_feasibility_table
        .to_html(classes="table", float_format=format_float),
        "solution_quality_table": solution_quality_table
        .to_html(classes="table", float_format=format_float),
        "optimality_plot_path": optimality_plot_path
    }
    html_out = template.render(template_vars)

    with tempfile.NamedTemporaryFile('w', delete=True, suffix='.html') as f:
        url = 'file://' + f.name
        f.write(html_out)
        f.flush()
        webbrowser.open(url)


def setup_arg_parser(subparsers):
    parser = subparsers.add_parser('compare', help="Compares two results files.")

    parser.add_argument(
        'solver-results-path-1',
        type=Path,
        help='File containing the first results file to compare.'
    )
    parser.add_argument(
        'solver-results-path-2',
        type=Path,
        help='File containing the second results file to compare.'
    )
    parser.add_argument(
        'oracle-path',
        type=Path,
        help='Oracle file containing known results for the benchmarks being compared.'
    )
    parser.add_argument(
        '--group-by',
        type=str,
        choices=[
            'problem_size_quantiles',
            'solution_size_quantiles',
            'oracle_runtime_quantiles',
            'feasibility'
        ],
        default='problem_size_quantiles',
        help='Oracle file containing known results for the benchmarks being compared.'
    )
    parser.set_defaults(exec_subcommand=main)
