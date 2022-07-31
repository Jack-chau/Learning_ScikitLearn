# Classification example of (Iris)
    # k-nearest neighbors algorithm
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
# load data set
iris = sns.load_dataset('iris')
print(iris.head())
print(iris.columns)
print(iris.species.value_counts())
# three type of labels: setosa, versicolor, virginica

# sns.pairplot(iris,hue='species',height=1.5)
# plt.show()

X_iris = iris.drop('species',axis=1) # move label from dataset axis = 1 is column
y_iris = iris['species'] # id output
print(X_iris)

from sklearn.neighbors import KNeighborsClassifier
# learn from data AND label
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_iris,y_iris)

# predict on unseen data
X_unseen = [[4.95,3.1,1.4,0.3]]
pred = model.predict(X_unseen)
print(pred)


