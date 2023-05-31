from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('data/dsl_skipgram_2020_m5_f500_epoch2_w5.model.w2v.bin', binary=True)
model.save_word2vec_format('danish_dsl_and_reddit_word2vec_word_embeddings.txt', binary=False)