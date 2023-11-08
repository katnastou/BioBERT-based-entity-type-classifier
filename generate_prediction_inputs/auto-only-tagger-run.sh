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
    dictionary-files-tagger-STRINGv12/blacklist_terms_over_10M.txt \
    dictionary-files-tagger-STRINGv12/curated_local.tsv \
    auto-only \
    /scratch/project_2001426/stringdata/tagger/tagcorpus


job_name="Jentag"  # Replace 'your_job_name' with the actual Slurm job name - check the slurm script for that
USER="katenast"

#generate matches in the correct format with identifiers
monitor_job_completion "$job_name" "$USER" sbatch slurm-create-matches.sh dictionary-files-tagger-STRINGv12/all_entities.tsv auto-only/all_matches.tsv auto-only/database_matches.tsv
if [ ! -f "/scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/results/database_documents.tsv" ]; then
    monitor_job_completion "$job_name" "$USER" sbatch slurm-create-docs.sh dictionary-files-tagger-STRINGv12/excluded_documents.txt 'gzip -cd `ls -1 /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/pmc/*.en.merged.filtered.tsv.gz` `ls -1r /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/pubmed/*.tsv.gz` |' /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/results/database_documents.tsv
fi

ln -s /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs/results/database_documents.tsv auto-only/database_documents.tsv


job_name="matches"
# Initial sort and split job
monitor_job_completion "$job_name" "$USER" sbatch sort-split.sh auto-only/database_documents.tsv auto-only/database_matches.tsv

# Monitoring and executing subsequent jobs
job_name="sortsp"  # Replace 'your_job_name' with the actual Slurm job name - check the slurm script for that

monitor_job_completion "$job_name" "$USER" ./run-context-split.sh

job_name="context"

monitor_job_completion "$job_name" "$USER" ../run_predict_batch_auto_only_che.sh
monitor_job_completion "$job_name" "$USER" ../run_predict_batch_auto_only_dis.sh
monitor_job_completion "$job_name" "$USER" ../run_predict_batch_auto_only_ggp.sh
monitor_job_completion "$job_name" "$USER" ../run_predict_batch_auto_only_org.sh

job_name="predict"
monitor_job_completion "$job_name" "$USER" sbatch ../blocklist_generation/generate-blocklists.sh /scratch/project_2001426/stringdata/blocklists-paper/auto-only

job_name="genbloc"
monitor_job_completion "$job_name" "$USER" cat /scratch/project_2001426/stringdata/blocklists-paper/auto-only/auto_global.tsv dictionary-files-tagger-STRINGv12/blacklist_terms_over_10M.txt > dictionary-files-tagger-STRINGv12/blacklist_terms_over_10M+auto_only_list.txt
monitor_job_completion "$job_name" "$USER" cp /scratch/project_2001426/stringdata/blocklists-paper/auto-only/auto_local.tsv  dictionary-files-tagger-STRINGv12/auto_only_local.txt
monitor_job_completion "$job_name" "$USER" sbatch slurm-tagger.sh dictionary-files-tagger-STRINGv12 \
                                           /scratch/project_2001426/stringdata/stringdata-v12/tagger_input_docs \
                                           dictionary-files-tagger-STRINGv12/blacklist_terms_over_10M+auto_only_list.txt \
                                           dictionary-files-tagger-STRINGv12/auto_only_local.txt \
                                           auto-only/final-tagger-run \
                                           /scratch/project_2001426/stringdata/tagger/tagcorpus
#this will give me the tagger results with auto-only for the evaluations
