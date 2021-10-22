OUTPUT_DIR=$1

THIS_SCRIPT_PATH=$(readlink -f "$0")
THIS_SCRIPT_DIR=$(dirname "$THIS_SCRIPT_PATH")

mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Download the various archives (total size: 21.5 GB)
cat $THIS_SCRIPT_DIR/file_list_TREC-Chem_2010_corpus.txt | xargs -n 1 -P 0 wget -q 

# Extract the downloaded archives (total size after decompressing the archives: 101.6 GB)
ls -1 *.tar.gz | xargs -n 1 -P 0 tar -xzf

# Flatten the directory structure
find xml -name "*.xml" | xargs mv -t xml
rm -fr xml/EP xml/US xml/WO

# Remove the downloaded archives
rm *.tar.gz
