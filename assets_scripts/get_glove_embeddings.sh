OUTPUT_PATH=$1

mkdir gensim_tmp
GENSIM_DATA_DIR=gensim_tmp python -m gensim.downloader --download glove-wiki-gigaword-300 
mv gensim_tmp/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz .
rm -r gensim_tmp

mv glove-wiki-gigaword-300.gz $OUTPUT_PATH
