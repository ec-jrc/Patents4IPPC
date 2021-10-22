OUTPUT_DIR=$1

mkdir -p $OUTPUT_DIR

python ../download_from_gdrive.py 1zcG9jwpVIzwsIBnJKdMZVA_QIMEWRVIZ $OUTPUT_DIR/tfidf_patstat_model.pkl
