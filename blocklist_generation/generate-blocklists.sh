#!/bin/bash

TYPES="
ggp
che
org
dis
"

#generate local lists for everything
for type in $TYPES; do
    PRED_DIR="/scratch/project_2001426/stringdata/STRING-blocklists-v12/${type}-predictions"
    mkdir -p local_blacklists-01-${type}
    for f in ${PRED_DIR}/output_with_probabilities_*; do echo $f; time python3 make_local_blacklist.py -m 1 -t 1.01 -s ${type} -T ${type} $f > local_blacklists-01-${type}/$(basename $f); done


    if [ "$type" != "dis" ]; then
        python3 global_blacklist_candidates.py -m 2 -t 0.5 local_blacklists-01-${type}/output_with_probabilities_* | sort -rn > ${type}_global_min_mentions_1_min_docs_2_threshold_0.5.tsv
        awk -F"\t" '{printf("%s\tt\n",$4)}' ${type}_global_min_mentions_1_min_docs_2_threshold_0.5.tsv > global_blacklist_${type}_2col.tsv
    else
        #different threshold for diseases
        python3 global_blacklist_candidates.py -m 2 -t 0.85 local_blacklists-01-${type}/output_with_probabilities_* | sort -rn > ${type}_global_min_mentions_1_min_docs_2_threshold_0.85.tsv
        awk -F"\t" '{printf("%s\tt\n",$4)}' ${type}_global_min_mentions_1_min_docs_2_threshold_0.85.tsv > global_blacklist_${type}_2col.tsv
    fi

    if [[ "$type" == "che" || "$type" == "ggp" ]]; then
        python3 global_blacklist_candidates.py -m 2 -t 0.0 local_blacklists-01-${type}/*.tsv | sort -rn > global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv

        #local block/allowlist

        mkdir -p local-blacklists-ratio-1000-${type}
        for f in local_blacklists-01-${type}/*.tsv; do echo $f; python3 make_local_bw_lists.py --blacklist --ratio 1000 global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv $f | sort -rn > local-blacklists-ratio-1000-${type}/$(basename $f); done
        #allowlist/whitelist only for ggp
        if [ $type == "ggp" ] then
            mkdir -p local-whitelists-ratio-1e15-global-${type}-prob-0.3
            for f in local_blacklists-01-${type}/*.tsv; do echo $f; python3 make_local_bw_lists.py --ratio 1000000000000000 --global-ggp-prob 0.3 global-non-${type}-probabilities-min-mentions-1-min-docs-2.tsv $f | sort -rn > local-whitelists-ratio-1e15-global-${type}-prob-0.3/$(basename $f); done

            cat local-whitelists-ratio-1e15-global-${type}-prob-0.3/* > local-whitelist-${type}-ratio-1e15-global-${type}-prob-0.3.tsv

            awk -F"\t" '{printf("%s\t%s\tf\n",$2, $3)}' local-whitelist-${type}-ratio-1e15-global-${type}-prob-0.3.tsv > local_whitelist_${type}_3col.tsv
        fi

        cat local-blacklists-ratio-1000-${type}/* > local-blacklists-ratio-1000-${type}.tsv
        awk -F"\t" '{printf("%s\t%s\tt\n",$2, $3)}' local-blacklists-ratio-1000-${type}.tsv > local_blacklist_${type}_3col.tsv
    fi
done

#Concatenate all auto local lists
cat local_whitelist_ggp_3col.tsv local_blacklist_ggp_3col.tsv local_blacklist_che_3col.tsv | sort > auto_local.tsv

#Concatenate all automated global lists into one list
cat global_blacklist_che_2col.tsv global_blacklist_ggp_2col.tsv global_blacklist_dis_2col.tsv global_blacklist_org_2col.tsv | sort -u > auto_global.tsv
