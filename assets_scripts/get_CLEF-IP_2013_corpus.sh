OUTPUT_DIR=$1

THIS_SCRIPT_PATH=$(readlink -f "$0")
THIS_SCRIPT_DIR=$(dirname "$THIS_SCRIPT_PATH")

# Download the various archives (total size: ~13 GB).
# Note that the corpus for CLEF-IP 2013 is the same as for the previous year (2012)
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR
cat $THIS_SCRIPT_DIR/file_list_CLEF-IP_2013_corpus.txt | xargs -n 1 -P 0 wget -q 

# Extract the downloaded archives (total size after decompressing the archives: 105.5 GB)
ls -1 *.7z.001 | xargs -n 1 -P 0 7za x

# Remove the downloaded archives
rm -fr *.7z.*
cd $THIS_SCRIPT_DIR
