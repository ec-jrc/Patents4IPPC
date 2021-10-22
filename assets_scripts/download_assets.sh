echo "Downloading AllNLI dataset..."
sh get_AllNLI.sh assets/data/raw/NLI/AllNLI.tsv

echo "Downloading STS Benchmark dataset..."
sh get_STSb.sh assets/data/raw/STSb/stsbenchmark.tsv

echo "Downloading TREC-Chem 2009 corpus (may take several hours)..."
sh get_TREC-Chem_2009_corpus.sh assets/data/raw/TREC-Chem/2009/corpus

echo "Downloading TREC-Chem 2010 corpus (may take several hours)..."
sh get_TREC-Chem_2010_corpus.sh assets/data/raw/TREC-Chem/2010/corpus

echo "Downloading TREC-Chem 2009 PA topics..."
sh get_TREC-Chem_2009_PA_topics.sh assets/data/raw/TREC-Chem/2009/PA_topics.xml

echo "Downloading TREC-Chem 2009 PA qrels..."
sh get_TREC-Chem_2009_PA_qrels.sh assets/data/raw/TREC-Chem/2009/PA_qrels.txt

echo "Downloading TREC-Chem 2009 TS topics..."
sh get_TREC-Chem_2009_TS_topics.sh assets/data/raw/TREC-Chem/2009/TS_topics.xml

echo "Downloading TREC-Chem 2010 PA topics..."
sh get_TREC-Chem_2010_PA_topics.sh assets/data/raw/TREC-Chem/2010/PA_topics

echo "Downloading TREC-Chem 2010 TS topics..."
sh get_TREC-Chem_2010_TS_topics.sh assets/data/raw/TREC-Chem/2010/TS_topics.xml

echo "Downloading TREC-Chem 2011 PA topics..."
sh get_TREC-Chem_2011_PA_topics.sh assets/data/raw/TREC-Chem/2011/PA_topics

echo "Downloading Helmers manual..."
sh get_helmers_manual.sh assets/data/raw/Helmers_manual

echo "Downloading pre-trained TF-IDF model..."
sh get_pretrained_tfidf_model.sh assets/models

echo "Downloading GloVe embeddings..."
sh get_glove_embeddings.sh assets/models/glove/glove-wiki-gigaword-300.gz

echo "Downloading Stanford CoreNLP..."
sh get_stanford_corenlp.sh assets/models/glove/stanford-corenlp-full-2018-02-27

echo "Downloading Universal Sentence Encoder..."
sh get_universal_sentence_encoder.sh assets/models/universal-sentence-encoder-large_5
