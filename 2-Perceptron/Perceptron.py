# 感知机
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skData

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

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')

# 加载数据
data = np.array(df.iloc[:100, [0, 1, 2]])  # iloc是使用索引进行取值
# print(data)
X, y = data[:, 0:2], data[:, 2]
y = np.array([1 if i == 1 else -1 for i in y])


# print(y)


class Model:
    # 对构建的模型进行初始化
    def __init__(self):
        self.w = np.zeros(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1
        # self.da    ta = data

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降法
    '''对错误的分类进行修改，知道错误的分类的sign函数值全部都>0时这个算法的循环才会停止'''

    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train) - 1):
                X = X_train[d]
                # print(X)
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'


perceptron = Model()
print(perceptron.w, "++", perceptron.b)
perceptron.fit(X, y)
print(perceptron.w, "++", perceptron.b)

plt.plot(np.linspace(4, 7, 10), -(np.poly1d(perceptron.w)(np.linspace(4, 7, 10)) / perceptron.w[1]), label='liner')
plt.show()
