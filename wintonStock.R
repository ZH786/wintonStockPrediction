#------Required Packages-----#
library(dplyr)
library(ggplot2)
library(purrr)
library(Matrix)
library(xgboost)
library(magrittr)
library(mice)
library(caret)
library(keras)
library(ModelMetrics)

#------Functions-----#
interp.med <- function(x) {
  x[is.na(x)] <- median(na.omit(x)); x
}

generator <- function(data, start = 1, end = 120, delay=0){
  #data[,(ncol(data)-lookback+1):ncol(data)]
  xIndicies <-  start + end - 2
  xDat <- data[,start:xIndicies]
  yIndicies <- delay + end 
  yDat <- data[,yIndicies]
  list(inputs = xDat, targets = yDat)
}


#------Import Dataset-----#
rawTrain <- read.csv('train.csv')
rawTest <- read.csv('test_2.csv')
submissionSamp <- read.csv('sample_submission_2.csv')
#-------------------------#
trnFeat <- rawTrain[,2:26]
tstFeat <- rawTest[,2:26]
iisTrn <- rawTrain[,27:147]
iisTrnDaily <- iisTrn[,c(1,2)]
iisTrnMins <- iisTrn[,c(-1,-2)]
oosTrn <- rawTrain[,148:209]
oosTrnDaily <- oosTrn[,c(61,62)]
oosTrnMins <- oosTrn[,c(-61,-62)]
iisTst <- rawTest[,27:147]
wtsTrn <- rawTrain[,c('Weight_Intraday', 'Weight_Daily')]
daily <- data.frame(iisTrnDaily, oosTrnDaily)
intraDatTrn <- cbind(iisTrnMins, oosTrnMins)


#------PreProcessing-----# Daily
summary(trnFeat)
str(trnFeat)
table(trnFeat$Feature_1)
table(trnFeat$Feature_10)
table(trnFeat$Feature_20)

trnFeat %<>% mutate(Feature_1 = ifelse(is.na(Feature_1), 0, Feature_1),
                   Feature_10 = ifelse(is.na(Feature_10), 0, Feature_10),
                   Feature_20 = ifelse(is.na(Feature_20), 1, Feature_20),
                   Feature_5 = ifelse(is.na(Feature_5), 0, Feature_5),
                   Feature_9 = ifelse(is.na(Feature_9), 0, Feature_9),
                   Feature_13 = ifelse(is.na(Feature_13), 0, Feature_13),
                   Feature_16 = ifelse(is.na(Feature_16), 0, Feature_16))

tstFeat %<>% mutate(Feature_1 = ifelse(is.na(Feature_1), 0, Feature_1),
                    Feature_10 = ifelse(is.na(Feature_10), 0, Feature_10),
                    Feature_20 = ifelse(is.na(Feature_20), 1, Feature_20),
                    Feature_5 = ifelse(is.na(Feature_5), 0, Feature_5),
                    Feature_9 = ifelse(is.na(Feature_9), 0, Feature_9),
                    Feature_13 = ifelse(is.na(Feature_13), 0, Feature_13),
                    Feature_16 = ifelse(is.na(Feature_16), 0, Feature_16))

trnFeat[,c(1,5,9,10,13,16,20)] <- lapply(trnFeat[,c(1,5,9,10,13,16,20)],as.factor)
tstFeat[,c(1,5,9,10,13,16,20)] <- lapply(tstFeat[,c(1,5,9,10,13,16,20)],as.factor)
#missDat <- function(x) {
#  sum(is.na(x))/length(x)*100
#}

#apply(trnFeat, 2,missDat) # percentage of missing data

impute <- mice(trnFeat[,c(-1,-5,-9,-10,-13,-16,-20)], m = 3)
imputeTst <- mice(tstFeat[,c(-1,-5,-9,-10,-13,-16,-20)], m = 3)

trnFeatNew <- complete(impute,2)
tstFeatNew <- complete(imputeTst,2)

iisTrnMinsCln <- map(iisTrnMins,interp.med) %>% as.data.frame()
iisTstMinsCln <- map(iisTst[,c(-1,-2)],interp.med) %>% as.data.frame()

#stripplot(impute, pch= 20, cex = 1.2)

dailyTrnDat <- data.frame(trnFeatNew, trnFeat[,c(1,5,9,10,13,16,20)],iisTrnMinsCln, daily,wtsTrn)
tstDat <- data.frame(tstFeatNew, tstFeat[,c(1,5,9,10,13,16,20)], iisTstMinsCln,iisTst[,c(1,2)])
#dailyTrnNumFS <- scale(dailyTrnDat[,c(-19:-25)])
#dailyTrnNumFS.mean <- attr(dailyTrnNumFS, 'scaled:center')
#dailyTrnNumFS.std <- attr(dailyTrnNumFS, 'scaled:scale')
#dailyTrnDat <- data.frame(trnFeat[,c(1,5,9,10,13,16,20)], dailyTrnNumFS)

dailyRet1 <- dailyTrnDat[,c(-148)]
dailyRet2 <- dailyTrnDat[,c(-147)]

trainInd <- createDataPartition(dailyRet1$Ret_PlusOne, p = 0.75, times = 1)
#trnFeat <- trnFeat %>% select_if(~sum(!is.na(.)) > 10000) #Use if not using tree based model
trainRet1 <- dailyRet1[trainInd[[1]],]
testRet1 <- dailyRet1[-trainInd[[1]],]

#-------------Modelling-------------#
sprseMat <- sparse.model.matrix(Ret_PlusOne ~ ., data = trainRet1[,c(-148,-149)])
xgbMat <- xgb.DMatrix(sprseMat, label = trainRet1$Ret_PlusOne)
model1 <- xgboost(xgbMat,
                  booster = 'gbtree',
                  eta = 0.1,
                  max_depth = 7,
                  subsample = 0.7,
                  nrounds = 250,
                  weight = wtsTrn$Weight_Daily)

predSprseMat <- sparse.model.matrix(Ret_PlusOne ~ ., data = testRet1[,c(-148,-149)])
predRet1 <- predict(model1, predSprseMat)
plot(testRet1$Ret_PlusOne, predRet1)

errRet1 <- testRet1$Weight_Daily*abs(testRet1$Ret_PlusOne-predRet1)


train2Ind <- createDataPartition(dailyRet2$Ret_PlusTwo, p=0.75,times=1)
trainRet2 <- dailyRet2[train2Ind[[1]],]
testRet2 <- dailyRet2[-train2Ind[[1]],]
sprseMat2 <- sparse.model.matrix(Ret_PlusTwo ~ ., data = trainRet2[,c(-148,-149)])
xgbMat2 <- xgb.DMatrix(sprseMat2, label = trainRet2$Ret_PlusTwo)
model2 <- xgboost(xgbMat2,
                  booster = 'gbtree',
                  eta = 0.1,
                  max_depth = 7,
                  subsample = 0.7,
                  nrounds = 250,
                  weight = wtsTrn$Weight_Daily)

predSprseMat2 <- sparse.model.matrix(Ret_PlusTwo ~ ., data = testRet2[,c(-148,-149)])
predRet2 <- predict(model2, predSprseMat2)
plot(testRet1$Ret_PlusOne, predRet1)

errRet2 <- testRet2$Weight_Daily*abs(testRet2$Ret_PlusTwo-predRet2)
dailyTrnErr <- data.frame(errRet1, errRet2)

#----------BASELINE Model----------# - intraday model

#baseMedModel <- map_dbl(oosTrnMins, median, na.rm = FALSE)

predIntMed <- rep(0, nrow(testRet1))
errIntRet <- testRet1$Weight_Intraday*abs(oosTrnMins[-trainInd[[1]],] - predIntMed)

WMAE <- mean(c(as.matrix(dailyTrnErr),as.matrix(errIntRet))) #1726.154 Xgboost (daily) + baseline model (intra) 


#----------Neural Networks----------# - intraday model

table(!complete.cases(intraDatTrn))
#impute1 <- mice(intraDatTrn, m = 3)


#Preprocessing for IntraDay data
intraDatTrnClnBU <- map(intraDatTrn,interp.med) %>% as.data.frame() #Median imputation of missing data

intraDatTrnCln <- data.matrix(intraDatTrnClnBU)
intraDatTrnCln <- scale(intraDatTrnCln) #Feature scaling
intraDatTrnCln.means <- attr(intraDatTrnCln, 'scaled:center')
intraDatTrnCln.std <- attr(intraDatTrnCln, 'scaled:scale')

trainIntraInd <- createDataPartition(intraDatTrnCln[,c('Ret_121')], p=0.75,times=1)
trainIntra <- intraDatTrnCln[trainIntraInd[[1]],]
testIntra <- intraDatTrnCln[-trainIntraInd[[1]],]

#trainDat <- generator(trainIntra)

# intraModel <- keras_model_sequential() %>%
#   layer_lstm(units = 32, 
#              return_sequences = TRUE,
#              dropout = 0.1,
#              input_shape = list(NULL,dim(trainDat$inputs)[2])) %>% 
#   layer_lstm(units = 32,
#              dropout = 0.1) %>% 
#   layer_dense(units = 1)
# 
# intraModel %>% 
#   compile(optimizer = optimizer_adam(),
#                        loss = 'mse') %>% 
#   fit(x = trainDat$inputs,
#       y = trainDat$targets,
#       batch_size = 32,
#       epochs = 40)
  
intraModel <- keras_model_sequential() %>% 
  layer_dense(units = 50, activation = 'relu', input_shape = c(dim(trainIntra[,1:119])[2])) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 30, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 16, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)

intraModel %>% compile(optimizer = optimizer_adam(),
                       loss = 'mse') 

annOutput <- matrix(,nrow = nrow(testIntra), ncol = ncol(testIntra))
for (i in 120:179){
  intraModel %>% fit(x = trainIntra[,1:119],
                     y = trainIntra[,i],
                     batch_size = 32,
                     epochs = 40)
  
  actual <- testIntra[,i]
  pred <- intraModel %>% predict(testIntra[,c(1:119)])
  annOutput[,i] <- pred*intraDatTrnCln.std[i]+intraDatTrnCln.means[i]
}

annOutputBU <- annOutput
annOutput <- annOutput[,120:179]
testIntrawts <- wtsTrn$Weight_Intraday[-trainIntraInd[[1]]]
annErrTrn <- testIntrawts*abs(testIntra[,120:179]- annOutput)

WMAEann <- mean(c(as.matrix(dailyTrnErr),as.matrix(annErrTrn))) #748978.6??? Something gone wrong!

#------------FINAL MODEL TRAINING on complete Train data-----------#

dailyTrnDat
dailyRet1Fnl <- dailyTrnDat[,c(-148)]
dailyRet2Fnl <- dailyTrnDat[,c(-147)]
sprseMat1 <- sparse.model.matrix(Ret_PlusOne ~ ., data = dailyRet1Fnl[,c(-148,-149)])
xgbMat1 <- xgb.DMatrix(sprseMat1, label = dailyRet1Fnl$Ret_PlusOne)
modelDly1 <- xgboost(xgbMat1,
                  booster = 'gbtree',
                  eta = 0.1,
                  max_depth = 7,
                  subsample = 0.7,
                  nrounds = 250,
                  weight = dailyRet1Fnl$Weight_Daily)

sprseMat2 <- sparse.model.matrix(Ret_PlusTwo ~ ., data = dailyRet2Fnl[,c(-148,-149)])
xgbMat2 <- xgb.DMatrix(sprseMat2, label = dailyRet2Fnl$Ret_PlusTwo)
modelDly2 <- xgboost(xgbMat2,
                  booster = 'gbtree',
                  eta = 0.1,
                  max_depth = 7,
                  subsample = 0.7,
                  nrounds = 250,
                  weight = dailyRet2Fnl$Weight_Daily)

intraMedModel <- map_dbl(intraDatTrnClnBU[,c(120:179)],median,na.rm=TRUE)
intraTstMeds <- t(matrix(rep(intraMedModel,nrow(rawTest)), nrow = length(intraMedModel), ncol = nrow(rawTest)))
tstDat$Ret_PlusOne <- 1 #Sparse matrix requires this variable to exist in newdata
tstDat$Ret_PlusTwo <- 1 #Sparse matrix requires this variable to exist in newdata


sprseMatDly1Tst <- sparse.model.matrix(Ret_PlusOne ~ ., data = tstDat[,-148])
sprseMatDly2Tst <- sparse.model.matrix(Ret_PlusTwo ~ ., data = tstDat[,-147])

Ret_PlusOnePred <- predict(modelDly1, sprseMatDly1Tst)
Ret_PlusTwoPred <- predict(modelDly2, sprseMatDly2Tst)

subFnl <- cbind(intraTstMeds, Ret_PlusOnePred, Ret_PlusTwoPred)
FnlPred <- as.vector(t(subFnl))
submissionSamp$Predicted <- FnlPred
write.csv(submissionSamp, file = 'WintonSubmission.csv', row.names = FALSE)
