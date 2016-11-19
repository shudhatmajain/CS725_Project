from __future__ import print_function
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm, ensemble

data=np.load("train_data.npy")    
testdata=np.load("data.npy")
labels = list(range(2,35,1)) + list(range(36, data.shape[1]-1, 1))
data_X=data[:,labels]
scaler = preprocessing.StandardScaler()
data_X = scaler.fit_transform(data_X)
data_y=data[:,35]
test_data_y=testdata[:,35]
testdata_X = testdata[:,labels]
testdata_X = scaler.transform(testdata_X)

# training the model
# regr = svm.SVR(C=1.0,epsilon=0.1)
print("lol")
regr = ensemble.AdaBoostRegressor(n_estimators=50, learning_rate=0.5)
regr.fit(data_X,data_y)
print("lol")
# predictions

testdata_y = regr.predict(testdata_X)

np.set_printoptions(precision=3)
np.savetxt('output.txt',testdata_y,delimiter='\n')

def f(a, b, t):
    count = 0.0
    for i in range(len(a)):
        label = 0
        if b[i]>0:
            label = 1
        predicted = 0
        if a[i] >= t:
            predicted = 1
        if predicted==label:
            count += 1
    return count/len(a)

print(f(testdata_y, test_data_y, 0.001))


