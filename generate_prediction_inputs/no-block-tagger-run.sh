#!/bin/bash

# Function to monitor job completion
monitor_job_completion() {
    local job_name=$1
    local USER=$2
    shift 2  # Shift the arguments to access the command
    command_to_execute="$@"  # The command to execute after removing the first two arguments
    job_id=$(squeue -h -o "%A" -u $USER -n $job_name)
    if [ -n "$job_id" ]; then
        echo "Found job ID for $job_name: $job_id"
        # Check if the job is still in the queue or running
        while squeue -u "$USER" | grep "$job_id" &> /dev/null; do
            sleep 60  # Adjust the sleep duration as needed
        done
        echo "Slurm job $job_name with ID $job_id has finished."
        # Execute the command
        eval "$command_to_execute"
    else
        echo "No active job found with the name $job_name."
    fi
}

sbatch slurm-tagger.sh \
    dictionary-files-tagger-STRINGv12 \
    /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs \
    dictionary-files-tagger-STRINGv12/empty_global.tsv \
    dictionary-files-tagger-STRINGv12/curated_local.tsv \
    no-block \
    /scratch/project_2001426/stringdata/tagger/tagcorpus


job_name="Jentag"  # Replace 'your_job_name' with the actual Slurm job name - check the slurm script for that
USER="katenast"

#generate matches in the correct format with identifiers
monitor_job_completion "$job_name" "$USER" sbatch slurm-create-matches.sh dictionary-files-tagger-STRINGv12/all_entities.tsv no-block/all_matches.tsv no-block/database_matches.tsv
if [ ! -f "/scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/results/database_documents.tsv" ]; then
    monitor_job_completion "$job_name" "$USER" sbatch slurm-create-docs.sh dictionary-files-tagger-STRINGv12/excluded_documents.txt 'gzip -cd `ls -1 /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/pmc/*.en.merged.filtered.tsv.gz` `ls -1r /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/pubmed/*.tsv.gz` |' /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/results/database_documents.tsv
fi

ln -s /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/results/database_documents.tsv no-block/database_documents.tsv

#no-block I only need the pair results - > I am not doing any prediction runs here. 