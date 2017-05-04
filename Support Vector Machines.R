# support vector machines (SVM)
# it's a supervised marching leanrnig methods
# it can be used to solve regression and clustering problems
# the goal is to find the optimal hyperplane and the margins is optimized
# svm can be also effiencient to non-linear problems

#load the SVM package
library(e1071)
plot(iris)
plot(iris$Sepal.Length,iris$Sepal.Width, col = iris$Species)
plot(iris$Petal.Length,iris$Petal.Width, col = iris$Species)

# build the training and test dataset
s <- sample(150, 100) #create a ramdon dataset with sample size 100 within the range (150 = sample size)
col <- c("Petal.Length", "Petal.Width", "Species")
iris_train <- iris[s,col]
iris_test <- iris[-s,col]

# train the SVM model
svmfit <- svm(Species ~., 
              data = iris_train, 
              kernel = "linear", 
              cost = 10, 
              scale = F)
print(svmfit)
plot(svmfit, iris_train[,col])

# use tuned function to identity what's the optimal parameter to use
tuned <- tune(svm, Species~., data = iris_train,
              kernal = "linear", 
              ranges = list(cost=c(0.001, 0.01, 0.1, 1, 10, 100)))
summary(tuned)
# based the suggested cost parameter, modify the model and re-run it.


# predict on the test dataset
p <- predict(svmfit, iris_test[,col], type = "class")
plot(p)

# calculate the accuracy
out <- table(p, iris_test[,3])
out
sum(diag(out))/50
