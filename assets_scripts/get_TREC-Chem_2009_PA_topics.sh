OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q http://ir.nist.gov/TREC2010-CHEM/trec/2009/PA-topics_withApplications.zip
unzip -qq PA-topics_withApplications.zip -d PA-topics_withApplications

mv PA-topics_withApplications/PA-topics_applications/PA-all.xml $OUTPUT_PATH

rm PA-topics_withApplications.zip
rm -r PA-topics_withApplications
