
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import gensim.models
from gensim.models import CoherenceModel
mallet_path = '/Users/guozikun/Desktop/computer/GitHub/GZK_Code/HELP_GUNGUN/mallet-2.0.8/bin/mallet'
train = []

fp = open('/Users/guozikun/Desktop/computer/GitHub/GZK_Code/HELP_GUNGUN/jieba_data_short.txt','r',encoding='utf-8')
for line in fp:
    if line != '':
        line = line.split()
        train.append([w for w in line])
temp_lst = []
for i in train[0]:
    temp_lst.append(i.split(","))
dictionary = corpora.Dictionary(temp_lst)
corpus = [dictionary.doc2bow(text) for text in temp_lst]

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=100)
# num_topics：主题数目
# passes：训练伦次
# num_words：每个主题下输出的term的数目

for topic in lda.print_topics(num_words = 20):
    termNumber = topic[0]
    print(topic[0], ':', sep='')
    listOfTerms = topic[1].split('+')
    for term in listOfTerms:
        listItems = term.split('*')
        print('  ', listItems[1], '(', listItems[0], ')', sep='')

import pyLDAvis.gensim

'''插入之前的代码片段'''

d=pyLDAvis.gensim.prepare(lda, corpus, dictionary)

'''
lda: 计算好的话题模型

corpus: 文档词频矩阵

dictionary: 词语空间
'''

#pyLDAvis.show(d)		#展示在浏览器
# pyLDAvis.displace(d) #展示在notebook的output cell中
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
                                                        start=2, limit=40, step=6)
# Show graph
limit = 40;
start = 2;
step = 6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()




