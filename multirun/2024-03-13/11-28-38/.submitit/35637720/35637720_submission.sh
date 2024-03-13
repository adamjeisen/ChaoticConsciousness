#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=2
#SBATCH --error=/rdma/vast-rdma/vast/millerlab/eisenaj/code/ChaoticConsciousness/multirun/2024-03-13/11-28-38/.submitit/%j/%j_0_log.err
#SBATCH --job-name=run_delase
#SBATCH --mail-user=eisenaj@mit.edu
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/rdma/vast-rdma/vast/millerlab/eisenaj/code/ChaoticConsciousness/multirun/2024-03-13/11-28-38/.submitit/%j/%j_0_log.out
#SBATCH --signal=USR2@120
#SBATCH --time=15
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /rdma/vast-rdma/vast/millerlab/eisenaj/code/ChaoticConsciousness/multirun/2024-03-13/11-28-38/.submitit/%j/%j_%t_log.out --error /rdma/vast-rdma/vast/millerlab/eisenaj/code/ChaoticConsciousness/multirun/2024-03-13/11-28-38/.submitit/%j/%j_%t_log.err /om2/user/eisenaj/miniforge3/envs/communication-transformer/bin/python -u -m submitit.core._submit /rdma/vast-rdma/vast/millerlab/eisenaj/code/ChaoticConsciousness/multirun/2024-03-13/11-28-38/.submitit/%j
