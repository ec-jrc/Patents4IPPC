OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://sbert.net/datasets/AllNLI.tsv.gz
gunzip AllNLI.tsv.gz

mv AllNLI.tsv $OUTPUT_PATH
