import numpy
from sklearn import linear_model

file = numpy.loadtxt('input_med.txt', delimiter=',')

X = file[:,1:]
y = file[:,0]

# print(X)
# print(y)

print('Begin Train')
reg = linear_model.LinearRegression(fit_intercept = True)
reg.fit(X,y)
print(reg)
print('End Train')

print(reg.coef_)

# print(X[:2])
# print(y[:2])
print(reg.predict(X[:2, :]))
# print(reg.predict_proba(X[:2, :]))
print(reg.score(X, y))
