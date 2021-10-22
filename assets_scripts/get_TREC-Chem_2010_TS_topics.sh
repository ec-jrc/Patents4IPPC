OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

# Download and unzip TS topics
wget -q http://ir.nist.gov/TREC2010-CHEM/trec/2010/TSTopics2010.zip
unzip -qq TSTopics2010.zip -d TSTopics2010
rm TSTopics2010.zip
cd TSTopics2010/FINAL

sed -i -E "s/<\/narative>/<\/narrative>/" TS-18.txt
sed -i -E "s/<chemicals>(.+?)<chemicals>/<chemicals>\1<\/chemicals>/" TS-18.txt
echo "" >> TS-18.txt

sed -i -E "s/<title>(.+?)<\/narrative>/<title>\1<\/title>/" TS-21.txt

sed -i -E "s/(<condition>)/<\/details>\n\1/" TS-22.txt

sed -i -E "s/<chemicals>(.+?)<chemicals>/<chemicals>\1<\/chemicals>/" TS-44.txt

echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" > $OUTPUT_PATH
echo "<topics>" >> $OUTPUT_PATH
cat *.txt >> $OUTPUT_PATH
echo "</topics>" >> $OUTPUT_PATH

cd ../..
rm -fr TSTopics2010
