OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://ir.nist.gov/TREC2010-CHEM/trec/2010/PATopics.tar.gz
tar -xzf PATopics.tar.gz
mv PATopics $OUTPUT_PATH

rm PATopics.tar.gz
