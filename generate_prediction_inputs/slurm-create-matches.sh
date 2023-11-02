#!/bin/bash
# Definining resource we want to allocate.
#SBATCH --nodes=1
#SBATCH --ntasks=40

# 5 CPU cores per task to keep the parallel data feeding going. 
#SBATCH --cpus-per-task=1

# Allocate enough memory.
#SBATCH --mem=162G
#SBATCH -p small
###SBATCH -p gputest

# Time limit on Puhti's small partition is 3 days. 72:00:00
#SBATCH -t 24:00:00
###SBATCH -t 00:30:00
#SBATCH -J matches

# Puhti project number
#SBATCH --account=Project_2001426

# Log file locations, %j corresponds to slurm job id. symlinks didn't work. Will add hard links to directory instead. Now it saves in projappl dir.
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Clear all modules
mkdir -p logs

module purge
module load gcc/11.3.0 
module load perl

ENTITIES="$1" #dictionary-files-tagger-STRINGv12/all_entities.tsv
INPUT_MATCHES="$2" #all_prediction_matches.tsv
OUTPUT_MATCHES="$3"

#generate matches in the correct format with identifiers
./scripts/create_matches.pl "$ENTITIES" "$INPUT_MATCHES" "$OUTPUT_MATCHES"


