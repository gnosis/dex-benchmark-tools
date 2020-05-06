NOTE: Use of the code here to evaluate the standard solver is pending on adding the runtime to a solution (see https://github.com/gnosis/dex-solver/issues/207).

# Summary

This is a set of tools for benchmarking the batch auction solvers.

# Benchmark workflow

1. Get a set of problem instances in a folder. Optionally, this tool can be used to **download** some from the S3 bucket.
2. Run the solver on those problem instances, making sure a solution and corresponding problem file have the same file name. See `src/run_standard_solver.sh` for an example.
3. **Extract** "results" from the solution instances. That is, take all the solution instances and product a *results file*. A *results file* simply condenses the metrics we are interested in evaluating like runtime, solution size, etc. Solution (and problem) instances can be deleted after this step.
4. [Optional] **Update** the *oracle file* with the obtained results. The *oracle file* is a database of the best results obtained per instance, useful to know if an instance has already been proved feasible/infeasible, what is the best optimum found so far, etc. 
5. **Compare** 2 *results files*.   

# Installation

```bash
virtualenv --python=/usr/bin/python3 venv
. venv/bin/activate
pip install -r requirements.txt
```

# Usage

## Downloading problem instances

```bash
python -m src.benchmark download --help
```

Downloads a set of instances (i.e. problems) from the S3 bucket. The relevant access keys must be in place. Install aws client and see the section `Configure AWS` [here](https://gitlab.gnosisdev.com/devops/gnosis-staging). 


## Extracting metrics from solution instances

```bash
python -m src.benchmark extract --help
```

From a folder with a set of solution instances, extract meaningful metrics like runtime, solution size, etc. and stores them in a "results" file. The solution instance files can then be deleted.


## Update oracle

```bash
python -m src.benchmark update_oracle --help
```

This step can be done as part of "extract" (see 'extract' options). It updates the oracle DB with the results from a given "results" file.

## Compare results

```bash
python -m src.benchmark compare --help
```

Compares two "results" files. Should open the browser and display some statistics.

# Interpreting comparison

TODO
