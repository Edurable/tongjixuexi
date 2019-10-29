import numpy as np

# 这就是索引,可以使用索引来获取元素,在Python还可以用负数索引(python将从右及最后一个元素)开始往左数影刺'-1'是最后一个元素的位置以此递增
print(np.linspace(1, 10, 10))
print(np.array(np.linspace(1, 10, 10))[-1])
print(np.array(np.linspace(1, 10, 10))[-2])
print(np.array(np.linspace(1, 10, 10))[-10])
#-1为什么是所引到了  序列中的最后一个元素，这是因为，python支持从右向左的遍历，此时，索引的下标就是从-1开始，依次遍历....-3 -2 -1