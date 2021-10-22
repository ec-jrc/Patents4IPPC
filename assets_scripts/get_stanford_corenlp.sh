OUTPUT_PATH=$1

wget -q http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip 
unzip -qq stanford-corenlp-full-2018-02-27.zip
rm stanford-corenlp-full-2018-02-27.zip

mv stanford-corenlp-full-2018-02-27 $OUTPUT_PATH
