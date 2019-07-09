import codecs
import numpy as np
import heapq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(threshold=np.inf)
# 各章のデータ読み込み
num = "1" #9章あるので、1~9の数字を入れる
corpus = codecs.open("ginga" +num+ ".txt", "r", "utf-8").read().splitlines()

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

#文書ベクトルの作成
vectorizer = CountVectorizer(analyzer=wakachi,binary=False,token_pattern=u"(?u)\\b\\w+\\b")
transformer = TfidfTransformer()
tf = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(tf)

# tfidfが計算された結果の表示
print("TF-IDF値", "文書番号")
for i in range(0, tfidf.size):
    print(tfidf.data[i], tfidf.indices[i])


#上位3つのTF-IDF値を出す
lst_sort = np.sort(tfidf.data)
print(heapq.nlargest(3, lst_sort))

#番号と単語の対応
for k,v in sorted(vectorizer.vocabulary_.items(), key=lambda x:x[1]):
    print(str(k) + ": " + str(v))

