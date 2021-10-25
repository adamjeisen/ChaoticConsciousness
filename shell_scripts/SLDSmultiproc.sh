#! /bin/bash
#
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12	
#SBATCH --time=10:00

source /om2/user/eisenaj/anaconda/etc/profile.d/conda.sh
conda activate chaotic-consciousness
python /om2/user/eisenaj/code/ChaoticConsciousness/_sandbox/slds_method/multi_process.py


