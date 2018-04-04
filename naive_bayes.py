import numpy as np
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

print(X)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

pred = clf.predict([[-0.8, -1]])
print(pred)
