OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://sbert.net/datasets/stsbenchmark.tsv.gz
gunzip stsbenchmark.tsv.gz

mv stsbenchmark.tsv $OUTPUT_PATH
