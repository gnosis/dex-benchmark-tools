#!/bin/bash
# Run standard solver on a set of instances.
# parameters:
# $1 - path to standard solver root dir
# $2 - path to problems directory
# $3 - path to solutions directory

# Save current dir
cur_dir=$(pwd)

# Go to solver dir
cd $1

# Create tmp dir
tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)

# Run all instances
for i in $2/*.json; do
    python -m src.match --solution $tmp_dir/solution.json $i best-token-pair;
    mv $tmp_dir/solution.json $3/$(basename $i);
done

# Remove tmp dir
rm -rf $tmp_dir

# Go back to where we were
cd $cur_dir
