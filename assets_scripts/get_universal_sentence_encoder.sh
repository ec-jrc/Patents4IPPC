OUTPUT_PATH=$1

wget -q https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed
mkdir universal-sentence-encoder-large_5
tar -xzf universal-sentence-encoder-large_5.tar.gz -C universal-sentence-encoder-large_5
rm universal-sentence-encoder-large_5.tar.gz

mv universal-sentence-encoder-large_5 $OUTPUT_PATH
