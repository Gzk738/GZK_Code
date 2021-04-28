from gensim import corpora
from gensim.models import LdaModel
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.corpora import WikiCorpus
from gensim.models.word2vec import  LineSentence
import codecs
import re
import os
import os.path
import multiprocessing
import logging
import json
import sys


train = []

def txt_json():
    # 文件路径
    path = "jieba_data.txt"
    # 读取文件
    with open(path, 'r', encoding="utf-8") as file:
        # 定义一个用于切割字符串的正则
        seq = re.compile(":")
        result = []
        # 逐行读取
        for line in file:
            lst = seq.split(line.strip())
            item = {
                str(line): line
            }
            result.append(item)
        print(type(result))
    # 关闭文件
    with open('jieba_data.json', 'w') as dump_f:
        json.dump(result, dump_f)

def data_cut():
    f = open("data_output.txt", "r", encoding="utf-8")
    s = f.read()

    words = psg.cut(s)
    print("cut successed")

    for word, flag in words:
        out_f = open("jieba_data.txt", "a", encoding="utf-8")
        out_f.write(word + " ")
        out_f.close()

    print("cut finished!!!")

def train_model(file):
    fp = codecs.open(file ,'r',encoding='utf8')
    print("reading json success")
    for line in fp:
        line = line.split()
        train.append([w for w in line])
        print("add data")

    dictionary = corpora.Dictionary(train)

    corpus = [dictionary.doc2bow(text) for text in train]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes=60)
    print("train success")
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

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    import pyLDAvis.gensim

    try:
        d=pyLDAvis.gensim.prepare(lda,corpus, dictionary, mds='mmds')
    except:
        print("model training falsed")
    #pyLDAvis.show(d, open_browser=True)
    pyLDAvis.save_html(d, 'lda_topic20.html')



def train_VEC():
    from gensim.test.utils import common_texts, get_tmpfile
    from gensim.models import Word2Vec
    fp = codecs.open("nosingle.txt" ,'r',encoding='utf8')
    print("reading json success")
    for line in fp:
        line = line.split()
        train.append([w for w in line])
        print("add data")

    dictionary = corpora.Dictionary(train)

    corpus = [dictionary.doc2bow(text) for text in train]
    f = open("jieba_data.txt", "r", encoding="utf-8")
    common_texts = f.read()
    mymodel = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20, passes=60)

    mymodel.save("word2vec.model")
    import pyLDAvis.gensim
    d = pyLDAvis.gensim.prepare(mymodel, corpus, dictionary, mds='mmds')
    pyLDAvis.save_html(d, 'lda_topic20.html')


if __name__ == '__main__':
    #data_cut()
    #txt_json()
    train_model("nosingle.json")
    #train_VEC()