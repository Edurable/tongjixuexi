import sklearn
import sklearn.linear_model as model
import sklearn.datasets as skData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
load_iris = skData.load_iris()

# print(load_iris['data'])
# print(dir(load_iris))
# print(load_iris.feature_names)
# print(load_iris['feature_names'])
# print(load_iris['target'])
df = pd.DataFrame(load_iris['data'], columns=load_iris['feature_names'])
df['label'] = load_iris.target
df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
# 加载数据
data = np.array(df.iloc[:100, [0, 1, 2]])  # iloc是使用索引进行取值
# print(data)
X, y = data[:, 0:2], data[:, 2]
y = np.array([1 if i == 1 else -1 for i in y])

clf = perceptron = model.Perceptron(fit_intercept=True, max_iter=1000,shuffle=True)
clf.fit(X,y)
# Weights assigned to the features.
print(clf.coef_)
# 截距 Constants in decision function.
print(clf.intercept_)

# 画布大小
plt.figure(figsize=(10,10))

# 中文标题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(data[50:100, 0], data[50:100, 1], c='orange', label='Iris-versicolor')

# 画感知机的线
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
