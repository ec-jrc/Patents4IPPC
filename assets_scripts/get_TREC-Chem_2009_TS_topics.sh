OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://ir.nist.gov/TREC2010-CHEM/trec/2009/TS-all.xml
head -n 7 TS-all.xml > $OUTPUT_PATH
sed -i -E "s/(<\/narrative>)/\n\1/" $OUTPUT_PATH
echo "</topic>" >> $OUTPUT_PATH
tail -n +8 TS-all.xml >> $OUTPUT_PATH
rm TS-all.xml
