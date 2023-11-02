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
#SBATCH -J docs

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

EXCLUDED_DOCS="$1"
INPUT_DOCS="$2" #'gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` |'
OUTPUT_DOCS="$3"

#generate documents in a tab delimited format
./scripts/create_documents.pl "$EXCLUDED_DOCS" "$INPUT_DOCS" "$OUTPUT_DOCS"

#sbatch slurm-create-docs.sh dictionary-files-tagger-STRINGv12/excluded_documents.txt 'gzip -cd `ls -1 /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/pmc/*.en.merged.filtered.tsv.gz` `ls -1r /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/pubmed/*.tsv.gz` |' database_documents.tsv