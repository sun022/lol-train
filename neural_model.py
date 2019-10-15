import numpy
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

file = numpy.loadtxt('input_large.txt', delimiter=',')

X = file[:,1:]
y = file[:,0] * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X)
# print(y)

print('Begin Train')
m = MLPRegressor(verbose = True, activation = 'logistic', solver = 'adam', early_stopping = False, hidden_layer_sizes = (50))
m.fit(X_train,y_train)
print(m)
print('End Train')

# print(y[:2])
print(m.predict(X_train[:5,]))
print(y_train[:5])
print()
print(m.predict(X_test[:5]))
print(y_test[:5])

print(m.score(X_train, y_train))
print(m.score(X_test, y_test))


y_pred = m.predict(X_test)

correct = 0
total = 0
for i in range(0, y_test.size):
    total+=1
    if y_test[i] < 0 and y_pred[i] < 0:
        correct+=1
    elif y_test[i] > 0 and y_pred[i] > 0:
        correct+=1

print('Correct:', correct)
print('Total:', total)
