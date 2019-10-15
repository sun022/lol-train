import numpy
from sklearn.linear_model import LogisticRegression

file = numpy.loadtxt('input_full.txt', delimiter=',')

X = file[:,1:]
y = file[:,0]

# print(X)
# print(y)

print('Begin Train')
clf = LogisticRegression(random_state=3, solver='sag', max_iter=10000).fit(X, y)
print('End Train')

# print(X[:2])
# print(y[:2])
print(clf.predict(X[:10, :]))
print(clf.predict_proba(X[:10, :]))
print(clf.score(X, y))

y_pred = clf.predict(X)

correct = 0
total = 0
for i in range(0, y.size):
    total+=1
    if y[i] < 0.5 and y_pred[i] < 0.5:
        correct+=1
    elif y[i] > 0.5 and y_pred[i] > 0.5:
        correct+=1

print('Correct:', correct)
print('Total:', total)
