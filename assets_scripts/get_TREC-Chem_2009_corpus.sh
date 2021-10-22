OUTPUT_DIR=$1

THIS_SCRIPT_PATH=$(readlink -f "$0")
THIS_SCRIPT_DIR=$(dirname "$THIS_SCRIPT_PATH")

# Download the various archives (total size: 17.8 GB)
mkdir tc2009_tmp
cd tc2009_tmp
cat $THIS_SCRIPT_DIR/file_list_TREC-Chem_2009_corpus.txt | xargs -n 1 -P 0 wget -q 

# Extract the downloaded archives (total size after decompressing the archives: 88.3 GB)
ls -1 *.tar.gz | xargs -n 1 -P 0 tar -xzf
ls -1 *.zip | xargs -n 1 -P 0 unzip -q

# Flatten the directory structure
rm -fr __MACOSX
cd ..
mkdir -p $OUTPUT_DIR/xml
find tc2009_tmp -name "*.xml" | sudo xargs mv -t $OUTPUT_DIR/xml
rm -fr tc2009_tmp
