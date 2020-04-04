#The Boston Housing data in the mlbench R package with 506 data points 
#is being analysed uing Linear regression, Ridge, Elastic and LASSO


#Loading the packages
library(glmnet)
library(mlbench)
library(dplyr)
library(KernelKnn)
library(DAAG)
library(GGally)

#Loading the Boston Housing Data using the mlbench package
data("BostonHousing")
bdata <- BostonHousing 

View(bdata)
str(bdata)

#Visualizing the data
ggpairs(bdata) #It is clear the data is not a normal distribution & Multicolinear


#Spliting the dataset
set.seed(263)
trainIndex <- sample(1:nrow(bdata), 0.8*nrow(bdata))
train <- bdata[trainIndex,]
test <- bdata[-trainIndex,]


lambda_seq <- 10^seq(2, -2, by=-.1)

set.seed(263)

train3 = bdata2 %>%
  sample_frac(0.8)

test3 = bdata2 %>%
  setdiff(train3)

x_train3 = model.matrix(medv~., train3)[,-14]
x_test3 = model.matrix(medv~., test3)[,-14]

y_train3 = train3 %>%
  select(medv) %>%
  unlist() %>%
  as.numeric()

y_test3 = test3 %>%
  select(medv) %>%
  unlist() %>%
  as.numeric()

#Building the Models
lin1 <- lm(medv~., data = train) #Modeling using Linear model= 73.26%
linpred <- predict(lin1, test) #Predicting using Linear model
AIC(lin1) #2418.523

#Predicting using Ridge
cv_output1 <- cv.glmnet(x_train3, y_train3, 
                       alpha = 0, lambda=lambda_seq)

plot(cv_output1) #Best Lambda
best_lam1 <- cv_output1$lambda.min #0.0126

ridge_best <- glmnet(x_train3, y_train3, alpha = 0, lambda = best_lam1)
ridge_pred <- predict(ridge_best, s=best_lam1, newx = x_test3)
coef(ridge_best)
AIC(ridge_best)

#Predicting using Elastic net
cv_output2 <- cv.glmnet(x_train3, y_train3, 
                        alpha = 0.5, lambda=lambda_seq)

plot(cv_output2) #Best Lambda
best_lam2 <- cv_output2$lambda.min #0.02512

elastic_best <- glmnet(x_train3, y_train3, alpha = 0.5, lambda = best_lam2)
elastic_pred <- predict(elastic_best, s=best_lam2, newx = x_test3)
coef(elastic_best)

#Predicting using LAsso
cv_output <- cv.glmnet(x_train3, y_train3, 
                       alpha = 1, lambda=lambda_seq)

plot(cv_output) #Best Lambda
best_lam <- cv_output$lambda.min #0.0126


lasso_best <- glmnet(x_train3, y_train3, alpha = 1, lambda = best_lam)
lasso_pred <- predict(lasso_best, s=best_lam, newx = x_test3)
coef(lasso_best)

#Prediction test
actpred1 <- data.frame(cbind(actuals=test$medv, predicted=linpred))
cor_ac <- cor(actpred1) #85.6%

actpred2 <- data.frame(cbind(y_test3, lasso_pred))
cor_ac2 <- cor(actpred2) #66.39%

actpred3 <- data.frame(cbind(y_test3, ridge_pred))
cor_ac3 <- cor(actpred3) #66.27%

actpred4 <- data.frame(cbind(y_test3, elastic_pred))
cor_ac4 <- cor(actpred4) #66.36%


#K-Fold Cross Validation
#Linear Regression
cvResults1 <- suppressWarnings(CVlm(data.frame(bdata), form.lm = medv~., m=5,
                                    dots = FALSE, seed=263, legend.pos = "topleft",
                                    printit = FALSE))

attr(cvResults1, 'ms') #835.61

#Ridge Regression
