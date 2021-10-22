OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://ir.nist.gov/TREC2010-CHEM/trec/2009/qrels-pa2009.txt -O $OUTPUT_PATH
