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

echo "Building Helmers manual (full-texts)..."
###############################################################################
echo ""
echo "Building the Helmers full-texts dataset requires scraping patent claims from Google Patents, which requires to download a webdriver."
echo ""

echo "Which browser are you running?: (chrome|firefox)"
read WD_TYPE
while [ $WD_TYPE != "chrome" ] && [ $WD_TYPE != "firefox" ]; do
    echo "Please enter \"chrome\" or \"firefox\":"
    read WD_TYPE
done

echo ""

if [ $WD_TYPE == "chrome" ]; then
    echo "Please download and unzip the Chrome webdriver from here (make sure to download the one that is compatible with your version of Chrome):"
    echo "https://chromedriver.chromium.org/downloads"
    echo ""
fi

if [ $WD_TYPE == "firefox" ]; then
    echo "Please download and unzip the Gecko webdriver from here (go to the latest release, look for \"Assets\" and download the webdriver based on what OS you have):"
    echo "https://github.com/mozilla/geckodriver/releases"
    echo ""
fi

echo "Now please enter the path to the unzipped webdriver you just downloaded:"
read WD_PATH
while [ ! -f "$WD_PATH" ]; do
    echo "File \"$WD_PATH\" does not exist. Please enter a valid path:"
    read WD_PATH
done

echo ""
echo "Back to building the dataset..."
###############################################################################
python utility_scripts/build_helmers_manual.py \
    -d assets/data/raw/Helmers_manual \
    --preprocess \
    --scrape-claims \
    --webdriver $WD_PATH \
    --webdriver-type $WD_TYPE \
    --separate-sections \
    -o assets/data/processed/helmers_manual_full_texts

echo "Building Helmers manual (full-texts, CSV)..."
python utility_scripts/build_helmers_manual.py \
    -d assets/data/raw/Helmers_manual \
    --preprocess \
    --scrape-claims \
    --webdriver $WD_PATH \
    --webdriver-type $WD_TYPE \
    -o assets/data/processed/helmers_manual_full_texts

echo "Building CLEF-IP 2013 (CSV)..."
python utility_scripts/build_clef_ip_2013.py \
    -q assets/data/intermediate/CLEF-IP/2013/qrels \
    -o assets/data/processed/CLEF-IP/2013/csv \
    --as-csv-files

echo "Building CLEF-IP 2013..."
python utility_scripts/build_clef_ip_2013.py \
    -q assets/data/intermediate/CLEF-IP/2013/qrels \
    -o assets/data/processed/CLEF-IP/2013/normal

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
