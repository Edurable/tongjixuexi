import numpy as np
import numpy.linalg as linalg  # 用于计算范数的工具包
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn
import sklearn.model_selection
import collections
from KNearestNeighbors.knn import KNN
iris = sklearn.datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
# print(y,"y数组的长度是:",len(y))
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)


clf = KNN(X_train, y_train)
clf.score(X_test, y_test)
