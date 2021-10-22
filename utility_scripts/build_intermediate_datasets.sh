echo "Building TREC-Chem 2009 corpus..."
python utility_scripts/build_trec_chem_corpus.py \
    -i assets/data/raw/TREC-Chem/2009/corpus \
    -y 2009 \
    -o assets/data/intermediate/TREC-Chem/2009/corpus.csv

echo "Building TREC-Chem 2010 corpus..."
python utility_scripts/build_trec_chem_corpus.py \
    -i assets/data/raw/TREC-Chem/2010/corpus \
    -y 2010 \
    -o assets/data/intermediate/TREC-Chem/2010/corpus.csv

echo "Building TREC-Chem 2009 PA qrels..."
python utility_scripts/build_trec_chem_2009_pa_qrels.py \
    -q assets/data/raw/TREC-Chem/2009/PA_topics.xml \
    -j assets/data/raw/TREC-Chem/2009/PA_qrels.txt \
    -o assets/data/intermediate/TREC-Chem/2009/PA_qrels.csv

echo "Building TREC-Chem 2010 PA qrels..."
python utility_scripts/build_trec_chem_201X_pa_qrels.py \
    -c assets/data/intermediate/TREC-Chem/2010/corpus.csv \
    -q assets/data/raw/TREC-Chem/2010/PA_topics \
    --english-only \
    -o assets/data/intermediate/TREC-Chem/2010/PA_qrels.csv

echo "Building TREC-Chem 2011 PA qrels..."
python utility_scripts/build_trec_chem_201X_pa_qrels.py \
    -c assets/data/intermediate/TREC-Chem/2010/corpus.csv \
    -q assets/data/raw/TREC-Chem/2011/PA_topics \
    --english-only \
    -o assets/data/intermediate/TREC-Chem/2011/PA_qrels.csv

# echo "Building PAJ corpus..."
# python utility_scripts/build_paj_corpus.py \
#     -i assets/data/raw/NTCIR/tcdata_patent_paj/paj \
#     -o assets/data/intermediate/NTCIR/paj_corpus.csv

# echo "Building acronyms maps for BREF documents..."
# python utility_scripts/build_acronyms_map.py \
#     assets/data/raw/JRC/BREF_docs/*.txt \
#     assets/data/intermediate/JRC/acronyms_maps
