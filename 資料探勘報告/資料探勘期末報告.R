stroke=read.table("E:/資料探勘/healthcare-dataset-stroke-data.csv",header=T,sep=",")
View(stroke)
dim(stroke)
str(stroke)

#處理遺失值
stroke$bmi[stroke$bmi=="N/A"]<-NA
stroke$age<-as.integer(as.numeric(stroke$age))
stroke$age[stroke$age==0]<-NA
stroke$smoking_status[stroke$smoking_status=="Unknown"]<-NA
stroke$gender[stroke$gender=="Other"]<-NA
#檢查是否有na值
sum(is.na(stroke))
#畫圖找出遺失值位置
require(Amelia)
missmap(stroke)

mean.1<-mean(as.numeric(as.character(stroke$bmi)),na.rm=T)
na.rows<-is.na(stroke$bmi)
stroke[na.rows,10]<-mean.1

complete.cases(stroke)
strokeok <- stroke[complete.cases(stroke), ]
sum(is.na(strokeok))
missmap(strokeok)

#資料型態轉變
strokeok$gender<-factor(strokeok$gender,
                        levels=c("Male","Female"),labels=c("1","0"))
strokeok$hypertension<-factor(strokeok$hypertension)
strokeok$heart_disease<-factor(strokeok$heart_disease)
strokeok$ever_married<-factor(strokeok$ever_married,
                              levels=c("Yes","No"),labels=c("1","0"))
strokeok$work_type<-factor(strokeok$work_type,
                           levels=c("children","Govt_job","Never_worked",
                                    "Private","Self-employed"),
                           labels=c("1","2","3","4","5"))
strokeok$Residence_type<-factor(strokeok$Residence_type,
                                levels=c("Urban","Rural"),labels=c("1","0"))
strokeok$bmi<-as.numeric(strokeok$bmi)
strokeok$smoking_status<-factor(strokeok$smoking_status,
                                levels=c("formerly smoked","never smoked",
                                         "smokes"),labels=c("3","0","1"))
strokeok$stroke<-factor(strokeok$stroke)
str(strokeok)

#刪除id欄位再進行分析
strokeok=strokeok[,-1]
View(strokeok)
dim(strokeok)
summary(strokeok)


#將資料隨機分成訓練集(80%)與測試集(20%)
set.seed(123)
c <- nrow(strokeok);c
np <- ceiling(0.2*c);np
test.index = sample(1:c,np)
testdata = strokeok[test.index,]
dim(testdata)
traindata = strokeok[-test.index,]
dim(traindata)


#資料平衡
install.packages(c("zoo","xts","quantmod","abind","rpart","class","ROCR"))
install.packages("C:/Users/user/Downloads/DMwR_0.4.1.tar.gz", repos=NULL, type="source")
library(DMwR)
traindata <- SMOTE(stroke~., traindata, perc.over=700, 
                   perc.under=100,k=5,learner=NULL)
table(traindata$stroke)

#正規化
for (i in 8:9){
  traindata[,i] = (traindata[,i]-mean(traindata[,i]))/(sd(traindata[,i]))
  testdata[,i] = (testdata[,i]-mean(testdata[,i]))/(sd(testdata[,i]))
}


#Rule-Based Classifier
library(tidyverse)
library(ggplot2)
library(mlbench)
library(caret)
library(lattice)
library(tibble)
data1 <- as_tibble(traindata)
data1
rulesFit1 <- traindata %>% 
  train(stroke~.,method = "PART",
        data = .,tuneLength = 5,
        trControl = trainControl(method = "cv"))
rulesFit1

#knn
library(class)
library(ggplot2)
error.rate = NULL
for(j in 1:20){
  pred1=knn(traindata[,-11],testdata[,-11],traindata[,11],k=j)
  aj=table(pred1,testdata[,11])
  error.rate[j]=sum(diag(aj))/sum(aj)
}
cat("KNN準確率",error.rate,sep="\n")
k<-1:20
error.dfj <- data.frame(error.rate,k)
pl <- ggplot(error.dfj,aes(x=k,y=error.rate))+ geom_point()
pl+ylim(0.65,0.75) + geom_line(lty="dotted",color='red')
#測試
predj=knn(traindata[,-11],testdata[,-11],traindata[,11],k=17)
require(caret)
j1=confusionMatrix(predj,testdata$stroke);j1
#訓練
predj1=knn(traindata[,-11],traindata[,-11],traindata[,11],k=17)
j11=confusionMatrix(predj1,traindata$stroke);j11


#naive Bayes
library(e1071)
library(tidyverse)
library(dplyr)
library(caret)

#測試
naiveBayesModel1 <- naiveBayes(stroke ~ .,data = traindata)
naiveBayesModel1
survivedPred1 <- predict(naiveBayesModel1,testdata[-11])
survivedPred1
a1=confusionMatrix(survivedPred1,testdata$stroke);a1
#訓練
naiveBayesModel11 <- naiveBayes(stroke ~ .,data = traindata)
naiveBayesModel11
survivedPred11 <- predict(naiveBayesModel11,traindata[-11])
survivedPred11
a11=confusionMatrix(survivedPred11,traindata$stroke);a11


#logistic regression
#測試
glm1 <- glm(formula = stroke ~ ., family = "binomial", data = traindata)
predl1 <- predict(glm1, newdata =testdata, type = "response")
m1 <- ifelse(predl1 >0.5, 1,0)
m2<-confusionMatrix(as.factor(m1),testdata$stroke);m2
#訓練
glm11 <- glm(formula = stroke ~ ., family = "binomial", data = traindata)
predl11 <- predict(glm11, newdata =traindata, type = "response")
m11 <- ifelse(predl11 >0.5, 1,0)
m22<-confusionMatrix(as.factor(m11),traindata$stroke);m22


#svm
#測試
require(e1071)
svm.model1 <- svm( factor(stroke)~ .,data = traindata, cost = 100, gamma = 1)
summary(svm.model1)
svm.pred1 <- predict(svm.model1, testdata[,-11])
a<-confusionMatrix(svm.pred1,testdata$stroke);a
#訓練
require(e1071)
svm.model11 <- svm( factor(stroke)~ .,data = traindata, cost = 100, gamma = 1)
summary(svm.model11)
svm.pred11 <- predict(svm.model11, traindata[,-11])
abc<-confusionMatrix(svm.pred11,traindata$stroke);abc


#XGboost
library(tidyverse)
library(ggplot2)
library(caret)
library(RWeka)
train_index <- createFolds(traindata$stroke, k = 10)
traindata11 <- as_tibble(traindata)
xgboostFit <- traindata11 %>% train(stroke ~ ., method = "xgbTree", data = .,
                                    tuneLength = 5,
                                    trControl = trainControl(method = "cv" , indexOut = train_index),
                                    tuneGrid = expand.grid(
                                      nrounds = 20,
                                      max_depth = 3,
                                      colsample_bytree = 0.6,
                                      eta = 0.1,
                                      gamma=0,
                                      min_child_weight = 1,
                                      subsample =0.5
                                    )) 
xgboostFit

#-------------------------------------------------------------------------------------------------#
#利用lasso篩選變數
library(glmnet)
lasso = glmnet(x = as.matrix(traindata[, -11]) 
               ,y = traindata[, 11] ,alpha = 1,family = "binomial")
plot(lasso, xvar='lambda', main="Lasso")
cv.fit=cv.glmnet(x = data.matrix(traindata[, -11]),
                 y = traindata[, 11],family='binomial',type.measure="deviance")
best.lambda = cv.fit$lambda.min
best.lambda
plot(lasso, xvar='lambda', main="Lasso")
abline(v=log(best.lambda), col="blue", lty=5.5 )
a<-coef(cv.fit, s = cv.fit$lambda.min);a
select.ind<-which(a!=0)
select.ind = select.ind[-1]-1
select.ind
select.varialbes = colnames(traindata)[select.ind]
select.varialbes
#篩選出來的變數為全部模型


#-------------------------------------------------------------------------------------------------#
none <- glm(formula = stroke ~ 1, family = "binomial", data = traindata);none
full <- glm(formula = stroke ~ ., family = "binomial", data = traindata);full
step(none,scope = list(upper=full,lower=none),direction = 'both')
#篩選出來的變數為gender、age 、 hypertension 、heart_disease、ever_married、 avg_glucose_level、smoking_status

#Rule-Based Classifier
data3 <- as_tibble(traindata[,c(1,2,3,4,5,8,10,11)])
data3
rulesFit3 <- data3 %>% 
  train(stroke~., method = "PART",data = .,
        tuneLength = 5,trControl = trainControl(method = "cv"))
rulesFit3

#knn
error.rate = NULL
for(f in 1:20){
  pred3=knn(traindata[,c(1,2,3,4,5,8,10)],
            testdata[,c(1,2,3,4,5,8,10)],traindata[,c(11)],k=f)
  af=table(pred3,testdata[,c("stroke")])
  error.rate[f]=sum(diag(af))/sum(af)
}
cat("KNN準確率",error.rate,sep="\n")
k<-1:20
error.dff <- data.frame(error.rate,k)
pl <- ggplot(error.dff,aes(x=k,y=error.rate))+ geom_point()
pl+ylim(0.70,0.75) + geom_line(lty="dotted",color='red')
#測試
predf=knn(traindata[,c(1,2,3,4,5,8,10)],testdata[,c(1,2,3,4,5,8,10)],traindata[,c(11)],k=5)
f1=confusionMatrix(predf,testdata$stroke);f1
#訓練
predff=knn(traindata[,c(1,2,3,4,5,8,10)],traindata[,c(1,2,3,4,5,8,10)],traindata[,c(11)],k=5)
f11=confusionMatrix(predff,traindata$stroke);f11



#naive Bayes
#測試
naiveBayesModel3 <- naiveBayes(stroke ~ .,
                               data = traindata[,c(1,2,3,4,5,8,10,11)])
survivedPred3 <- predict(naiveBayesModel3,testdata[,c(1,2,3,4,5,10)])
c1=confusionMatrix(survivedPred3,testdata$stroke);c1
#訓練
naiveBayesModel33<- naiveBayes(stroke ~ .,
                                data = traindata[,c(1,2,3,4,5,8,10,11)])
survivedPred33 <- predict(naiveBayesModel33,traindata[,c(1,2,3,4,5,10)])
p11=confusionMatrix(survivedPred33,traindata$stroke);p11



#logistic regression
#測試
glm3 <- glm(formula = stroke ~ ., 
            family = "binomial", data = traindata[,c(1,2,3,4,5,8,10,11)])
predl3 <- predict(glm3, 
                  newdata =testdata[,c(1,2,3,4,5,8,10)], 
                  type = "response")
m111 <- ifelse(predl3> 0.5, 1,0)
m222 <- confusionMatrix(as.factor(m111),testdata$stroke);m222
#訓練
glm44 <- glm(formula = stroke ~ ., 
             family = "binomial", data = traindata[,c(1,2,3,4,5,8,10,11)])
predl44 <- predict(glm44, 
                   newdata =traindata[,c(1,2,3,4,5,8,10)], 
                   type = "response")
m44 <- ifelse(predl44 >0.5, 1,0)
m444<-confusionMatrix(as.factor(m44),traindata$stroke);m444


##svm
#測試
svm.model3 <- svm( factor(stroke)~ ., 
                   data = traindata[,c(1,2,3,4,5,8,10,11)], 
                   cost = 100, gamma = 1)
summary(svm.model3)
svm.pred3 <- predict(svm.model3, testdata[,c(1,2,3,4,5,8,10)])
c=confusionMatrix(svm.pred3,testdata$stroke);c
#訓練
svm.model55 <- svm( factor(stroke)~ .,
                    data = traindata[,c(1,2,3,4,5,8,10,11)], 
                    cost = 100, gamma = 1)
summary(svm.model55)
svm.pred55 <- predict(svm.model55, traindata[,c(1,2,3,4,5,8,10)])
abcd<-confusionMatrix(svm.pred55,traindata$stroke);abcd

#XGboost
train_index <- createFolds(traindata$stroke, k = 10)
traindata33 <- as_tibble(traindata[,c(1,2,3,4,5,8,10,11)])
xgboostFit2 <- traindata33 %>% train(stroke ~ ., method = "xgbTree", data = .,
                                     tuneLength = 5,
                                     trControl = trainControl(method = "cv" , indexOut = train_index),
                                     tuneGrid = expand.grid(
                                       nrounds = 20,
                                       max_depth = 3,
                                       colsample_bytree = 0.6,
                                       eta = 0.1,
                                       gamma=0,
                                       min_child_weight = 1,
                                       subsample =0.5
                                     )) 
xgboostFit2

#-------------------------------------------------------------------------------------------------#
#XG篩選變數
#法二View feature importance/influence from the learned model查看學習模型的特徵重要性/影響
require(Matrix)
require(data.table)
require(xgboost)
train <- sparse.model.matrix(stroke~.-1, data = traindata)
output_vector = traindata[,c("stroke")] == "1"
bst <- xgboost(data = train, label = output_vector, max.depth = 2,
               eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)


#-------------------------------------------------------------------------------------------------#
##利用XGboost篩選出來的變數(為age、 heart_disease、avg_glucose_level)下去分析並看準確率是否有沒有提升

#Rule-Based Classifier
data4 <- as_tibble(traindata[,c(2,4,8,11)])
data4
rulesFit4 <- data4 %>% 
  train(stroke~., method = "PART",data = .,
        tuneLength = 5,trControl = trainControl(method = "cv"))
rulesFit4


#knn
error.rate = NULL
for(g in 1:20){
  pred4=knn(traindata[,c(2,4,8)],testdata[,c(2,4,8)],traindata[,c(11)],k=g)
  ag=table(pred4,testdata[,c("stroke")])
  error.rate[g]=sum(diag(ag))/sum(ag)
}
cat("KNN準確率",error.rate,sep="\n")
k<-1:20
error.dfg <- data.frame(error.rate,k)
pl <- ggplot(error.dfg,aes(x=k,y=error.rate))+ geom_point()
pl+ylim(0.69,0.80) + geom_line(lty="dotted",color='red')
#測試  
predg=knn(traindata[,c(2,4,8)],testdata[,c(2,4,8)],traindata[,c(11)],k=3)
g1=confusionMatrix(predg,testdata$stroke);g1
#訓練
predgg=knn(traindata[,c(2,4,8)],traindata[,c(2,4,8)],traindata[,c(11)],k=3)
g11=confusionMatrix(predgg,traindata$stroke);g11



#naive Bayes
#測試
naiveBayesModel4 <- naiveBayes(stroke ~ .,
                               data = traindata[,c(2,4,8,11)])
survivedPred4 <- predict(naiveBayesModel4,testdata[,c(2,4,8)])
d1=confusionMatrix(survivedPred4,testdata$stroke);d1
#訓練
naiveBayesModel77<- naiveBayes(stroke ~ .,
                               data = traindata[,c(2,4,8,11)])
survivedPred77 <- predict(naiveBayesModel77,traindata[,c(2,4,8)])
p111=confusionMatrix(survivedPred77,traindata$stroke);p111



#logistic regression
#測試
glm4 <- glm(formula = stroke ~ ., 
            family = "binomial", data = traindata[,c(2,4,8,11)])
predl4 <- predict(glm4, 
                  newdata =testdata[,c(2,4,8)], 
                  type = "response")
m1111 <- ifelse(predl4> 0.5, 1,0)
m2222 <- confusionMatrix(as.factor(m1111),testdata$stroke);m2222
#訓練
glm77 <- glm(formula = stroke ~ ., 
             family = "binomial", data = traindata[,c(2,4,8,11)])
predl77 <- predict(glm77, 
                   newdata =traindata[,c(2,4,8)], 
                   type = "response")
m77 <- ifelse(predl77 >0.5, 1,0)
m777<-confusionMatrix(as.factor(m77),traindata$stroke);m777


##svm
#測試
svm.model4 <- svm( factor(stroke)~ ., 
                   data = traindata[,c(2,4,8,11)], 
                   cost = 100, gamma = 1)
summary(svm.model4)
svm.pred4 <- predict(svm.model4, testdata[,c(2,4,8)])
d=confusionMatrix(svm.pred4,testdata$stroke);d
#訓練
svm.model551 <- svm( factor(stroke)~ .,
                    data = traindata[,c(2,4,8,11)], 
                    cost = 100, gamma = 1)
summary(svm.model551)
svm.pred551 <- predict(svm.model551, traindata[,c(2,4,8)])
abcd1<-confusionMatrix(svm.pred551,traindata$stroke);abcd1


#XGboost
train_index <- createFolds(traindata$stroke, k = 10)
traindata44 <- as_tibble(traindata[,c(2,4,8,11)])
xgboostFit4 <- traindata44 %>% train(stroke ~ ., method = "xgbTree", data = .,
                                     tuneLength = 5,
                                     trControl = trainControl(method = "cv" , indexOut = train_index),
                                     tuneGrid = expand.grid(
                                       nrounds = 20,
                                       max_depth = 3,
                                       colsample_bytree = 0.6,
                                       eta = 0.1,
                                       gamma=0,
                                       min_child_weight = 1,
                                       subsample =0.5
                                     ))
xgboostFit4

