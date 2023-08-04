import numpy as np
from collections import Counter
import serial

# ser = serial.Serial('/tty/USB0', 115200)  # 根据具体串口号和波特率设置
# if not ser.isOpen():
#     ser.open()


# 计算欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# K最近邻算法
def knn(X_train, y_train, X_new, k):
    distances = [euclidean_distance(X_new, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]


train_features = [[2, 3], [4, 5], [1, 2], [6, 7], [2, 4], [3, 6]]
train_labels = ["A01", "A45", "A56", "A67", "A24", "A36"]
test_features = [[3, 4]]
# 训练数据集
X_train = np.array([[feature1, feature2] for feature1, feature2 in train_features])
y_train = np.array(train_labels)
# 测试数据集
X_test = np.array([[feature1, feature2] for feature1, feature2 in test_features])

# 设定K值
K = 5
# 预测测试数据集的位置
predictions = [knn(X_train, y_train, x_new, K) for x_new in X_test]
print(predictions)
