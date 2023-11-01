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
#SBATCH -J Jentag

# Puhti project number
#SBATCH --account=Project_2001426

# Log file locations, %j corresponds to slurm job id. symlinks didn't work. Will add hard links to directory instead. Now it saves in projappl dir.
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Clear all modules
mkdir -p logs

module purge
module load gcc/11.3.0 
module load openmpi/4.1.4
module load boost/1.79.0-mpi

DICT_DIR="${1:-dictionary-files-tagger-STRINGv12}"
INPUT_DIR="${2:-/scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs}"
BLOCK_GL="${3:-dictionary-files-tagger-STRINGv12/all_global.tsv}"
BLOCK_LC="${4:-dictionary-files-tagger-STRINGv12/all_local.tsv}"
OUT_DIR="${5:-results}"
TAGGER="${6:-/scratch/project_2001426/stringdata/tagger/tagcorpus}"

mkdir -p ${OUT_DIR}

gzip -cd `ls -1 ${INPUT_DIR}/pmc/*.en.merged.filtered.tsv.gz` `ls -1r ${INPUT_DIR}/pubmed/*.tsv.gz` | \
cat ${DICT_DIR}/excluded_documents.txt - | \
${TAGGER} \
--threads=40 \
--autodetect \
--types=${DICT_DIR}/curated_types.tsv \
--entities=${DICT_DIR}/all_entities.tsv \
--names=${DICT_DIR}/all_names_textmining.tsv \
--groups=${DICT_DIR}/all_groups.tsv \
--stopwords=${BLOCK_GL} \
--local-stopwords=${BLOCK_LC} \
--type-pairs=${DICT_DIR}/all_type_pairs.tsv \
--out-matches=${OUT_DIR}/all_matches.tsv \
--out-segments=${OUT_DIR}/all_segments.tsv \
--out-pairs=${OUT_DIR}/all_pairs.tsv

echo -n 'result written in '"$OUT_DIR"$'\n'
seff $SLURM_JOBID
