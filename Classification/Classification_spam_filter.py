import pandas as pd
dataset = pd.read_csv('spam_or_not_spam.csv')
print(dataset.head(10))
#print(dataset.head(10))
#print(dataset.label.value_counts())
#print(dataset.info()) # one-null values

# Remove NaN values
# print(dataset[dataset.email.isnull()])
dataset.email.fillna('', inplace=True)
# print(dataset.info())

# Split Training set & Testing set
X, y = dataset['email'], dataset['label']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

# Text column => Term count columns Using CountVectorizer
    # 1. fit() - Learn all the terms in training dataset
    # 2. trainsform() - Count the occurances of each term in each ROW in training dataset

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words='english',lowercase=True)
vec.fit(X_train)
# print(X_train.shape)
# print(vec.get_feature_names())
# print(len(vec.get_feature_names()))
#print(vec.vocabulary_) #not a counter
X_train_dt = vec.transform(X_train)
# X_train_dt = X_train_dt.toarray()
#print(X_train_dt)

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# model = MultinomialNB() # for discribe frequency (counter) features
# model = BernoulliNB # for binary / boolean features (pronunciation ber-nu-lee)
# model = GaussianNB # for continuous features (e.g height, wieght)
model = MultinomialNB()
model.fit(X_train_dt, y_train)
X_test_dt = vec.transform(X_test)
pred = model.predict(X_test_dt.toarray()) # could be slow for some models
prob = model.predict_proba(X_test_dt.toarray()) # could be slow for some models
# print(prob)

# Scoring
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,pred)
# print(f'Accuracy scour: {accuracy:.6f}')

# precision_score, recall_score, fl_score on test set
from sklearn.metrics import precision_score,recall_score,f1_score
precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
f1 = f1_score(y_test,pred)
# print(f'precision : {precision:.6f}')
# print(f'Recall : {recall:.6f}')
# print(f'f1_score : {f1:.6f}')
# or
score = model.score(X_test_dt.toarray(),y_test)
# print(score)

'''Compare the accuracy with baseline accuracy
        - if we always predict the most frequent class, we will get this accuracy
        - baseline accuracy for the whole dataset
'''

# print(dataset.label.value_counts()[0])
# print(dataset.label.shape)
# labeled harm(normal)/ harm + spam
baseline_accuracy_overall = dataset.label.value_counts()[0]/dataset.shape

'''baseline accuracy for the testing set'''
import numpy as np
# print(y_test)
unique, counts = np.unique(y_test,return_counts=True)
# print(f'unique {unique[0]}: {counts[0]}')
# print(f'unique {unique[1]}: {counts[1]}')
baseline_accuracy_test = counts[0]/len(y_test)  #harm/overall trainset count
# print(baseline_accuracy_test)

'''sklearn method, same as baseline_accuracy'''
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
dummy_clf.fit(X_train_dt.toarray(), y_train)
acc_sore = dummy_clf.score(X_test_dt.toarray(), y_test)
# print(acc_sore)


'''Confusion Matrix find true positive, true negative etc...'''
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,pred)
# print(confusion)
tn, fp, fn, tp = confusion.ravel()
# print(f'True_negative: {tn}')
# print(f'False_Positive: {fp}')
# print(f'False_negative: {fn}')
# print(f'True_positive: {tp}')

'''ROC Curve'''
# import matplotlib.pyplot as plot
# from sklearn.metrics import plot_roc_curve
# plot_roc_curve(model, X_test_dt.toarray(),y_test)
# x_point = [0,1]
# y_point = [0,1]
# plot.plot(x_point,y_point,'--')
# plot.xlim([0,1])
# plot.ylim([0,1])
# plot.show()

'''Cross-validation'''
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plot
y_prob_cv = cross_val_predict(model,X_train_dt.toarray(), y_train,cv=3,method='predict_proba')
# print(y_prob_cv)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train,y_prob_cv[:,1])
# plot.plot(false_positive_rate,true_positive_rate)
# plot.plot([0,1],[1,0],'--')
# plot.title('ROC')
# plot.ylabel('True Positive Rate')
# plot.xlabel('False Positive Rate')
# plot.xlim([0,1])
# plot.ylim([0,1])
# plot.show()

'''AUC (Area Under the ROC Curve)'''
# from sklearn.metrics import roc_auc_score
# auc_cv= roc_auc_score(y_train,y_prob_cv[:,1])
# print(f'AUC: {auc_cv:.2f}')

'''Save Model   save vectorizer and safe model'''
import joblib
count_vectorizer_filename = 'vec.sav'
classifier_filename = 'spam_classifier.sav'
joblib.dump(vec,count_vectorizer_filename)
joblib.dump(model, classifier_filename)

'''Load the model'''
count_vectorizer_filename = 'vec.sav'
classifier_filename = 'spam_classifier.sav'
vec = joblib.load(count_vectorizer_filename)
model = joblib.load(classifier_filename)

'''Apply it on unseen data  
        - Convert & Predict using loaded vectorizer & model
'''
unknow_dt = vec.transform(['Hi, I will send you the documents later. Regards,Jack',
                           'Call 1234-5678 on this awesome opportunity',
                           'Final chance to win a iphone!!!',
                           'Hi, Please come in today. Best regards, Steven'])
print(model.predict(unknow_dt.toarray()))