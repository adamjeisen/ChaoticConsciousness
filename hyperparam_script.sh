#!/bin/bash
# hyperparam_script.sh

# This script is used to run the hyperparameter tuning for the ChaoticConsciousness model.
# Define the list of start times
START_TIMES=(113.906 353.165 867.291 1111.808 1593.264 1869.73
        3126.033 3150.618 3193.709 3476.332 4519.733 4549.834 5036.493
        5072.027 5085.11 5567.418 5640.45 5830.915
        5944.084, 6144.672)

# Loop through the start times
for START_TIME in "${START_TIMES[@]}"
do
    # echo "${START_TIME}"
    # Run the hyperparameter testing script with the current start time
    python hyperparameter_testing.py -m hyperparameter_testing=grid_search "++hyperparameter_testing.params.start_time=$START_TIME" &
    wait %1
done
# python hyperparameter_testing.py -m hyperparameter_testing=grid_search '++hyperparameter_testing.params.start_time=113.906, 353.165'