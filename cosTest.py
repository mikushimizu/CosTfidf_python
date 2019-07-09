# -*- coding: utf-8 -*-
import numpy as np

f1 = open('ginga1.txt')
data1 = f1.read()
f1.close()

f2 = open('ginga2.txt')
data2 = f2.read()
f2.close()

f3 = open('ginga3.txt')
data3 = f3.read()
f3.close()

f4 = open('ginga4.txt')
data4 = f4.read()
f4.close()

f5 = open('ginga5.txt')
data5 = f5.read()
f5.close()

f6 = open('ginga6.txt')
data6 = f6.read()
f6.close()

f7 = open('ginga7.txt')
data7 = f7.read()
f7.close()

f8 = open('ginga8.txt')
data8 = f8.read()
f8.close()

f9 = open('ginga9.txt')
data9 = f9.read()
f9.close()

#わかち書き関数
def wakachi(text):
    from janome.tokenizer import Tokenizer
    t = Tokenizer()
    tokens = t.tokenize(text)
    docs=[]
    for token in tokens:
        docs.append(token.surface)
    return docs

#文書ベクトル化関数
def vecs_array(documents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(analyzer=wakachi,binary=False,use_idf=True,token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()
    

if __name__ == '__main__':
    from sklearn.metrics.pairwise import cosine_similarity
    docs = [
    data1,      # 文書１
    data2,      # 文書２
    data3,   # 文書３
    data4,
    data5,
    data6,
    data7,
    data8,
    data9
    ]

#類似度行列作成
    cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),4)
    print(cs_array)
    print(vecs_array(docs))