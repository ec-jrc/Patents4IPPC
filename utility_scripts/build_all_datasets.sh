echo "Building NLI..."
python utility_scripts/build_nli.py \
    -d assets/data/raw/NLI/AllNLI.tsv \
    -o assets/data/processed/all_nli.csv

echo "Building STSb..."
python utility_scripts/build_stsb.py \
    -d assets/data/raw/STSb/stsbenchmark.tsv \
    -o assets/data/processed/stsb.csv

echo "Building Helmers manual..."
python utility_scripts/build_helmers_manual.py \
    -d assets/data/raw/Helmers_manual \
    --preprocess \
    -o assets/data/processed/helmers_manual.csv
    # NOTE: It is better not to include titles in this case as some of them
    # have been truncated and thus end with three dots (...)
    #--use-titles

echo "Building TREC-Chem 2009 automatic..."
python utility_scripts/build_trec_chem_automatic.py \
    -c assets/data/intermediate/TREC-Chem/2009/corpus.csv \
    -j assets/data/intermediate/TREC-Chem/2009/PA_qrels.csv \
    --preprocess \
    --use-titles \
    --add-negative-examples \
    --negatives-to-positives-ratio 1.0 \
    --seed 101 \
    -o assets/data/processed/trec_chem_2009_automatic.csv

echo "Building TREC-Chem 2010 automatic..."
python utility_scripts/build_trec_chem_automatic.py \
    -c assets/data/intermediate/TREC-Chem/2010/corpus.csv \
    -j assets/data/intermediate/TREC-Chem/2010/PA_qrels.csv \
    --preprocess \
    --use-titles \
    --add-negative-examples \
    --negatives-to-positives-ratio 1.0 \
    --seed 101 \
    -o assets/data/processed/trec_chem_2010_automatic.csv

echo "Building TREC-Chem 2011 automatic..."
python utility_scripts/build_trec_chem_automatic.py \
    -c assets/data/intermediate/TREC-Chem/2010/corpus.csv \
    -j assets/data/intermediate/TREC-Chem/2011/PA_qrels.csv \
    --preprocess \
    --use-titles \
    --add-negative-examples \
    --negatives-to-positives-ratio 1.0 \
    --seed 101 \
    -o assets/data/processed/trec_chem_2011_automatic.csv

# echo "Building TREC-Chem 2009 manual..."
# python utility_scripts/build_trec_chem_manual.py \
#     -q assets/data/raw/TREC-Chem/2009/TS_topics.xml \
#     -j assets/data/raw/TREC-Chem/2009/TREC-Chem-2009-TS_relevances/ts_qrls_closed.txt \
#     -c assets/data/intermediate/TREC-Chem/2009/corpus.csv \
#     --year 2009 \
#     --no-negative-scores \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/trec_chem_2009_manual.csv
#     # NOTE: "--no-negative-scores" removes relevance judgments where the score
#     # is negative (-1 means "unjudged", -2 means "unsure")

# echo "Building TREC-Chem 2010 manual..."
# python utility_scripts/build_trec_chem_manual.py \
#     -q assets/data/raw/TREC-Chem/2010/TS_topics.xml \
#     -j assets/data/raw/TREC-Chem/2010/TREC-Chem-2010-TS-qrels/TS2010_senior_allREL.qrel \
#     -c assets/data/intermediate/TREC-Chem/2010/corpus.csv \
#     --year 2010 \
#     --no-negative-scores \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/trec_chem_2010_manual.csv
#     # NOTE: "--no-negative-scores" removes relevance judgments where the score
#     # is negative (-1 means "unjudged", -2 means "unsure")    

# echo "Building NTCIR-3 PATENT (Headline + Text)..."
# python utility_scripts/build_ntcir3.py \
#     -c assets/data/intermediate/NTCIR/paj_corpus.csv \
#     -q assets/data/raw/NTCIR/3/topics/en \
#     -j assets/data/raw/NTCIR/3/rels/frel.b \
#     -qp headline \
#     -qp text \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/ntcir3_headline_and_text.csv

# echo "Building NTCIR-3 PATENT (Description + Narrative)..."
# python utility_scripts/build_ntcir3.py \
#     -c assets/data/intermediate/NTCIR/paj_corpus.csv \
#     -q assets/data/raw/NTCIR/3/topics/en \
#     -j assets/data/raw/NTCIR/3/rels/frel.b \
#     -qp description \
#     -qp narrative \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/ntcir3_description_and_narrative.csv

# echo "Building NTCIR-4 PATENT..."
# python utility_scripts/build_ntcir4_or_5.py \
#     -c assets/data/intermediate/NTCIR/paj_corpus.csv \
#     -q assets/data/raw/NTCIR/4/topic008-041_en_utf8 \
#     -j assets/data/raw/NTCIR/4/rels/rel.b.main \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/ntcir4.csv
#     # NOTE: NTCIR-4 already has more negative examples than it has positive
#     # examples
#     #--add-negative-examples \
#     #--negatives-to-positives-ratio 1.0 \
#     #--seed 101

# echo "Building NTCIR-5 PATENT..."
# python utility_scripts/build_ntcir4_or_5.py \
#     -c assets/data/intermediate/NTCIR/paj_corpus.csv \
#     -q assets/data/raw/NTCIR/5/topics_eng \
#     -j assets/data/raw/NTCIR/5/rels/rel.b.ntc5 \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/ntcir5.csv
#     # NOTE: If you plan to merge NTCIR-5 with NTCIR-4, then you probably don't
#     # need to add synthetic negative examples as NTCIR-4 already has plenty
#     #--add-negative-examples \
#     #--negatives-to-positives-ratio 1.0 \
#     #--seed 101

# echo "Building GS1..."
# python utility_scripts/build_gs1.py \
#     -c assets/data/raw/JRC/exportAT.csv \
#     -b assets/data/raw/JRC/QUERIES_final \
#     -j assets/data/raw/JRC/GS1_complete.xlsx \
#     -a assets/data/intermediate/JRC/acronyms_maps \
#     --preprocess \
#     --use-titles \
#     -o assets/data/processed/gs1_complete.csv
