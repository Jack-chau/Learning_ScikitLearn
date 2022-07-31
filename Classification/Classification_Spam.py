import pandas as pd
import numpy as np
# print(spam.head(10))

''' Preparation
        1. CountVectorizer
            - scikit-larn process numerical values (must convert text to numberical values)'''

from sklearn.feature_extraction.text import CountVectorizer

vectorizer_try = CountVectorizer(stop_words='english',lowercase=True)
# stop_words like conjunction,

# Let's try some string
messageList = ['Hello, this is an apple', 'I have an apple. Apple is good for your health.']
dataTry = vectorizer_try.fit_transform(messageList)
# print(type(dataTry))
# print(dataTry)
df = pd.DataFrame(dataTry.toarray(),columns=vectorizer_try.get_feature_names())
# print(df)

# How about chinese?
import jieba
jieba.set_dictionary('dict.txt')
#print('|'.join(jieba.cut('機器學習是人工智能的分支', cut_all=False)))  # 分詞
messageList = ['Hello, this is an apple', 'I have an apple. Apple is good for your health.','機器學習是人工智能的分支']

message_tokenized = [" ".join(jieba.cut(message)) for message in messageList]
dataTry = vectorizer_try.fit_transform(message_tokenized)
# print(dataTry)

df = pd.DataFrame(dataTry.toarray(), columns=vectorizer_try.get_feature_names())
# print(df)

message_list = ['He is special', 'I win a money and a prize', 'win money', 'You win special prize'] # first one is normal message, second one is spam
vec_explain = CountVectorizer(stop_words=['I', 'you', 'he', 'is', 'yes', 'and'], lowercase=True)

df_explain = pd.DataFrame(vec_explain.fit_transform(message_list).toarray(), columns=vec_explain.get_feature_names(),
                          index=['normal','normal','spam','spam'])
print(df_explain)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(df_explain,df_explain.index)
predict = clf.predict(vec_explain.transform(['You win special money']))
print(predict)














