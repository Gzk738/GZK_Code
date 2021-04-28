"""raw_corpus = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stoplist = set('for a of the and to in'.split(' '))
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in raw_corpus]

from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

precessed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
print(precessed_corpus)

from gensim import corpora
dictionary = corpora.Dictionary(precessed_corpus)
print(dictionary)

print("------------------------------------------------------")
print(dictionary.token2id)

print("------------------------------------------------------")
new_doc = "human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

print("------------------------------------------------------")

bow_corpus = [dictionary.doc2bow(text) for text in precessed_corpus]
print(bow_corpus)


from gensim import models
tfidf = models.TfidfModel(bow_corpus)
string = "system minors"
string_bow = dictionary.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_bow)
print(string_tfidf)"""

