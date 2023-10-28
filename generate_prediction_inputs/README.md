# Generate tagger output and processing for large scale prediction run

Input documents for large-scale execution: all [PubMed abstracts](https://a3s.fi/s1000/PubMed-input.tar.gz) (as of August 2022) and all full-texts available in the [PubmedCentral BioC](https://a3s.fi/s1000/PMC-OA-input.tar.gz) text mining collection (as of April 2022). 

Input dictionary files: all the files necessary to execute tagger are available in [dictionary-files-tagger-STRINGv12.tar.gz](https://zenodo.org/api/records/10008720/files/dictionary-files-tagger-STRINGv12.zip?download=1)

## What happens next

After the last step you should have the automatically blocklists for all entity types targeted. Then these are added in the tagger dictionary build (one can simply concatenate the curated and auto lists, taking into consideration that the curated lists are preferred when there are clashes) and you get to run with the new dictionary. To recreate the process to generate results as they are provided in STRING v12 one needs to run the tagger as follows: 

```
gzip -cd `ls -1 pmc/*.en.merged.filtered.tsv.gz` `ls -1r pubmed/*.tsv.gz` | cat dictionary-files-tagger-STRINGv12/excluded_documents.txt - | tagger/tagcorpus --threads=40 --autodetect --types=dictionary-files-tagger-STRINGv12/curated_types.tsv --entities=dictionary-files-tagger-STRINGv12/all_entities.tsv --names=dictionary-files-tagger-STRINGv12/all_names_textmining.tsv --groups=dictionary-files-tagger-STRINGv12/all_groups.tsv --stopwords=all_global.tsv --local-stopwords==all_local.tsv --out-matches=all_matches.tsv --out-segments=all_segments.tsv --out-pairs=all_pairs.tsv
```

Where the `all_pairs.tsv` contains the co-occurence-based text mining biomedical entity pairs with the updated blocklist. 