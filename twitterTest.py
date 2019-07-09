import csv
import codecs
import numpy as np
import heapq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(threshold=np.inf)
# データの用意
corpus = codecs.open('togetter_2017ep1.tsv', 'r', 'utf-8').read().splitlines()
tsv = csv.reader(corpus, delimiter = '\t')


#時間と分を使ってDictionaryのkeyを作る keyはそれでvalueに追加していく

#分かち書き関数
def wakachi(text):
    from janome.tokenizer import Tokenizer
    t = Tokenizer()
    tokens = t.tokenize(text)
    docs=[]
    for token in tokens:
        #3文字以上を対象に考える
        if len(token.surface) >= 3:
            docs.append(token.surface)
    return docs


for row in tsv:
        #print(row[1] + " " + row[2].replace('#EP演習', '').replace('#ep演習', '').replace('＃EP演習', ''))
        #row2=[]
        row2 = row[2].replace('#EP演習', '').replace('#ep演習', '').replace('＃EP演習', '').replace('w', '').replace('W', '')
        print(row[1] + " " +row2)

#文書ベクトルの作成
vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
transformer = TfidfTransformer()
tf = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(tf)

# tfidfが計算された結果の表示
print("TF-IDF値", "文書番号")
#for i in range(0, tfidf.size):
    #print(tfidf.data[i], tfidf.indices[i])

#上位3つのTF-IDF値を出す
lst_sort = np.sort(tfidf.data)
print(heapq.nlargest(3, lst_sort))

#番号と単語の対応
#for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
 #   print(str(k) + ": " + str(v))