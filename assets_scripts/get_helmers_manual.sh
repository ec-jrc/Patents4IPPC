OUTPUT_PATH=$1

mkdir -p $(dirname $OUTPUT_PATH)

wget -q https://ndownloader.figshare.com/files/13383059 -O human_eval.zip

unzip -qq human_eval.zip -d helmers_manual_tmp
mv helmers_manual_tmp/human_eval $OUTPUT_PATH
rm -r helmers_manual_tmp

rm human_eval.zip
