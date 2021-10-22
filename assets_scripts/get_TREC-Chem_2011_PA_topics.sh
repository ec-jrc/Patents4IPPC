OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://ir.nist.gov/TREC2010-CHEM/trec/2011/PATopics2011.tar.gz
tar -xzf PATopics2011.tar.gz
mv PATopics2011 $OUTPUT_PATH

rm PATopics2011.tar.gz
