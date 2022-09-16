from gensim.models import Word2Vec
import gensim.downloader

glove_vectors = gensim.downloader.load('word2vec-google-news-300')



print("\n")

model = Word2Vec(sentences=[["hello","world","earth","sunshine","law"]],vector_size=5,window=5,min_count=1,workers=4)
print(model.wv["hello"])



...