#!/bin/bash
# Definining resource we want to allocate.
#SBATCH --nodes=1
#SBATCH --ntasks=1

# 5 CPU cores per task to keep the parallel data feeding going. 
#SBATCH --cpus-per-task=2

# Allocate enough memory.
#SBATCH --mem=64G
#SBATCH -p small

# Time limit on Puhti's small partition is 3 days. 72:00:00
#SBATCH -t 00:15:00
#SBATCH -J gen-cont

# Puhti project number
#SBATCH --account=Project_2001426
#SBATCH -J context

# Log file locations, %j corresponds to slurm job id. symlinks didn't work. Will add hard links to directory instead. Now it saves in projappl dir.
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Clear all modules
module purge
module load python-data

function on_exit {
    rm -f predictions/$SLURM_JOBID
}
trap on_exit EXIT

echo "START $SLURM_JOBID: $(date)"

TYPE="$1"
i="$2"
if [[ ${TYPE} != "org" ]]; then
    python3 ./scripts/get_contexts.py \
        -t ${TYPE} \
        -w 100 \
        sorted-split/database_{documents,matches}-$i.tsv \
        > split-contexts/${TYPE}-contexts-w100-$i.tsv \
        2>delme/${TYPE}_contexts-$i.txt
else
    python3 ./scripts/get_contexts.py \
        -t ${TYPE} \
        -w 100 \
        sorted-split-org-only-first-species/database_{documents,matches}-$i.tsv \
        > split-contexts/${TYPE}-contexts-w100-$i.tsv \
        2>delme/${TYPE}_contexts-$i.txt
fi
