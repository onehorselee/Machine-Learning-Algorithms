###################################################################
##################### The Basic Model##############################
from sklearn import datasets
from sklearn import svm

# load Iris dataset
iris = datasets.load_iris()
x,y = iris.data, iris.target

# train the model
clf = svm.SVC(gamma=0.001, C = 100)
clf.fit(x,y)
predict = clf.predict(iris.data)

# caculate accuracy 
print("Accuracy:", sum(predict == iris.target)/len(iris.data))
