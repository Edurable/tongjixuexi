import numpy.linalg as LA  # 使用计算工具库进行范数的计算
import collections


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def prediction(self, X):  # 这个函数的作用就是用来计算给的X参数和训练用的变量之间的2范数或者是其他范数
        knn_list = []
        for i in range(self.n):
            dist = LA.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
        # 对计算到的结果再次进行确认，看是否是最近的邻近点了！如果不是就会像下面的代码一样进行更新
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = LA.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:   #将最小的2范数，欧式距离列入！！
                knn_list[max_index] = (dist, self.y_train[i])

        knn = [k[-1] for k in knn_list]
        count_pairs = collections.Counter(knn)
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label =self.prediction(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)
