OUTPUT_DIR=$1

THIS_SCRIPT_PATH=$(readlink -f "$0")
THIS_SCRIPT_DIR=$(dirname "$THIS_SCRIPT_PATH")

mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Download the various archives
wget -q --no-check-certificate https://www.ifs.tuwien.ac.at/~clef-ip/download/2013/topics/clef-ip-2013-clms-psg-training.zip
wget -q --no-check-certificate https://www.ifs.tuwien.ac.at/~clef-ip/download/2013/topics/clef-ip-2013-clms-psg-TEST.tgz 
wget -q --no-check-certificate https://www.ifs.tuwien.ac.at/~clef-ip/download/2013/qrels/2013-clef-ip-clsm-to-psg-qrels.zip

# Extract the downloaded archives
unzip -q clef-ip-2013-clms-psg-training.zip
tar -xzf clef-ip-2013-clms-psg-TEST.tgz
unzip -q 2013-clef-ip-clsm-to-psg-qrels.zip

# Remove the downloaded archives
rm -fr *.tgz *.zip
cd $THIS_SCRIPT_DIR
