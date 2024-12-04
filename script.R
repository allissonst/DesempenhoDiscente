options(scipen = 9999)
library(readxl)
library(caTools)
library(caret)
library(pROC)
library(e1071)
library(ipred)
library(ROCR)
library(MASS)
library(xgboost)
library(gbm)   

dados <- read_excel("dados.xlsx", sheet = "dados")

set.seed(123)


#### MODELO NÃO SUPERVISIONADO K-MEANS #### 
grupos_aval <- kmeans(dados$aval,centers = 2)

grupos_aval$cluster[grupos_aval$cluster==2]=0 #baixo desempenho passa a receber valor de 0

dados$avalcluster = grupos_aval$cluster #Inputando uma nova coluna no df dados

#### MODELO LINEAR NORMAL #### 

linear_normal <- lm(aval ~ eng + ime + des, data=dados)
summary(linear_normal)

#### REGRESSÃO LOGÍSTICA #### 

logit <- glm(avalcluster ~ eng + ime + des, data=dados, family = binomial)
summary(logit)


#### APRENDIZADO DE MÁQUINA SUPERVISIONADO #### 

#Fazendo a divisão (treino: 0.70; teste: 0.30)

set.seed(123)

divisao=sample.split(dados$avalcluster,SplitRatio = 0.7)
treinamento=subset(dados,divisao==T)
teste=subset(dados,divisao==F)

treinamento$avalcluster=as.factor(treinamento$avalcluster)
teste$avalcluster=as.factor(teste$avalcluster)


#### ANÁLISE DISCRIMINANTE LINEAR ####
adl = lda(avalcluster ~ eng + ime + des, 
          data=treinamento)

#TREINAMENTO
adlclass=predict(adl, 
                newdata=treinamento)$class

confusionMatrix(adlclass,
                treinamento$avalcluster,
                positive = "1")

adlclassprob=predict(adl, 
                    newdata=treinamento, 
                    type="prob")$posterior[,2]
adlrocobject=roc(as.numeric(treinamento$avalcluster) -1, 
                adlclassprob)

plot.roc(adlrocobject)
auc(adlrocobject)

#TESTE
adlclass=predict(adl, 
                 newdata=teste)$class

confusionMatrix(adlclass,
                teste$avalcluster,
                positive = "1")

adlclassprob=predict(adl, 
                     newdata=teste)$posterior[,2]

adlrocobject=roc(as.numeric(teste$avalcluster) -1, 
                 adlclassprob)

plot.roc(adlrocobject)
auc(adlrocobject)



### RANDOM FOREST ### 

ctrl=trainControl(method="cv",
                  number=10)

FitRF= train(
  avalcluster ~ eng + ime + des,
  method="rf",
  preProcess=c("scale"),
  trControl=ctrl,
  data=treinamento,
  importance=T
)

#TREINAMENTO
RFclass=predict(FitRF, 
                newdata=treinamento)

confusionMatrix(RFclass,
                treinamento$avalcluster,
                positive = "1")

RFclassprob=predict(FitRF, 
                    newdata=treinamento, 
                    type="prob")

RFrocobject=roc(treinamento$avalcluster, 
                RFclassprob[,2])

plot.roc(RFrocobject)
plot(varImp(FitRF))
auc(RFrocobject)

#TESTE
RFclass=predict(FitRF, 
                newdata=teste)

confusionMatrix(RFclass,
                teste$avalcluster,
                positive = "1")

RFclassprob=predict(FitRF, 
                    newdata=teste, 
                    type="prob")

RFrocobject=roc(teste$avalcluster, 
                RFclassprob[,2])

plot.roc(RFrocobject)
plot(varImp(FitRF))
auc(RFrocobject)

### KNN ### 

ctrl=trainControl(method="repeatedcv",
                  number=10)

FitKNN= train(
  avalcluster ~ eng + ime + des,
  method="knn",
  preProcess=c("scale"),
  trControl=ctrl,
  data=treinamento,
  tuneLength = 20
)

#TREINAMENTO

KNNclass=predict(FitKNN, 
                 newdata=treinamento)

confusionMatrix(KNNclass,
                treinamento$avalcluster,
                positive = "1")

KNNclassprob=predict(FitKNN, 
                     newdata=treinamento, 
                     type="prob")

KNNrocobject=roc(treinamento$avalcluster, 
                 KNNclassprob[,2])

plot.roc(KNNrocobject)
auc(KNNrocobject)

#TESTE
KNNclass=predict(FitKNN, 
                 newdata=teste)

confusionMatrix(KNNclass,
                teste$avalcluster,
                positive = "1")

KNNclassprob=predict(FitKNN, 
                     newdata=teste, 
                     type="prob")

KNNrocobject=roc(teste$avalcluster, 
                 KNNclassprob[,2])

plot.roc(KNNrocobject)
auc(KNNrocobject)


### NAIVE BAYES ### 

ctrl=trainControl(method="cv",
                  number=10)

Grid = expand.grid(usekernel = TRUE,
                   laplace = 0,
                   adjust = 1)

FitNB= train(
  avalcluster ~ eng + ime + des,
  method="naive_bayes",
  preProcess=c("scale"),
  trControl=ctrl,
  data=treinamento,
  tuneGrid = Grid
)

#TREINAMENTO
nbclass=predict(FitNB, 
                newdata=treinamento)

confusionMatrix(nbclass,
                treinamento$avalcluster,
                positive = "1")

nbclassprob=predict(FitNB, 
                    newdata=treinamento, 
                    type="prob")

nbrocobject=roc(treinamento$avalcluster, 
                nbclassprob[,2])

plot.roc(nbrocobject)
auc(nbrocobject)

#TESTE
nbclass=predict(FitNB, 
                newdata=teste)

confusionMatrix(nbclass,
                teste$avalcluster,
                positive = "1")

nbclassprob=predict(FitNB, 
                    newdata=teste, 
                    type="prob")

nbrocobject=roc(teste$avalcluster, 
                nbclassprob[,2])

plot.roc(nbrocobject)
auc(nbrocobject)


### LOGIT ### 

Fitlogit= train(
  avalcluster ~ eng + ime + des,
  method="glm",
  preProcess=c("scale"),
  family="binomial",
  data=treinamento
)

#TREINAMENTO
logitclass=predict(Fitlogit, 
                   newdata=treinamento)

confusionMatrix(logitclass,
                treinamento$avalcluster,
                positive = "1")

logitclassprob=predict(Fitlogit, 
                       newdata=treinamento, 
                       type="prob")

logitrocobject=roc(treinamento$avalcluster, 
                   logitclassprob[,2])

plot.roc(logitrocobject)
auc(logitrocobject)

#TESTE
logitclass=predict(Fitlogit, 
                   newdata=teste)

confusionMatrix(logitclass,
                teste$avalcluster,
                positive = "1")

logitclassprob=predict(Fitlogit, 
                       newdata=teste, 
                       type = "prob")

logitrocobject=roc(teste$avalcluster, 
                   logitclassprob[,2])

plot.roc(logitrocobject)
auc(logitrocobject)

plot(varImp(Fitlogit))

### SVM LINEAR ### 

svmfit=svm(avalcluster ~ eng + ime + des, 
           data=treinamento, 
           scale=T, 
           type= "C-classification", 
           kernel="linear",
           probability = TRUE)


#TREINAMENTO
svmclass=predict(svmfit, 
                 newdata=treinamento)

confusionMatrix(svmclass,
                treinamento$avalcluster,
                positive = "1")

svmclassprob=predict(svmfit, 
                     newdata=treinamento, 
                     type="prob")

svmrocobject=roc(treinamento$avalcluster, 
                as.numeric(svmclassprob))

plot.roc(svmrocobject)
auc(svmrocobject)

#TESTE
svmclass=predict(svmfit, 
                 newdata=teste)

confusionMatrix(svmclass,
                teste$avalcluster,
                positive = "1")

svmclassprob=predict(svmfit, 
                     newdata=teste, 
                     type="prob")

svmrocobject=roc(teste$avalcluster, 
                 as.numeric(svmclassprob))

plot.roc(svmrocobject)
auc(svmrocobject)


### BAGGING ###

baggedfit <- bagging(formula = avalcluster ~ eng + ime + des, 
                     data = treinamento,
                     coob = TRUE,
                     scale=T)

#TREINAMENTO

baggedclass <- predict(object = baggedfit,    
                       newdata = treinamento,  
                       type = "class")

confusionMatrix(data = baggedclass,       
                reference = treinamento$avalcluster,
                positive = "1")

baggedclassprob <- predict(object = baggedfit,    
                           newdata = treinamento,  
                           type = "prob")

baggedrocobject=roc(treinamento$avalcluster, 
                    baggedclassprob[,2])
plot.roc(baggedrocobject)
auc(baggedrocobject)

#TESTE
baggedclass <- predict(object = baggedfit,    
                       newdata = teste,  
                       type = "class")

confusionMatrix(data = baggedclass,       
                reference = teste$avalcluster,
                positive = "1")

baggedclassprob <- predict(object = baggedfit,    
                           newdata = teste,  
                           type = "prob")

baggedrocobject=roc(teste$avalcluster, 
                    baggedclassprob[,2])

plot.roc(baggedrocobject)
auc(baggedrocobject)

# Importância direta do modelo final
importancia <- varImp(baggedfit)
valores_importancia <- importancia$Overall
nomes_variaveis <- rownames(importancia)

# Criar o gráfico de barras
barplot(valores_importancia,
        names.arg = nomes_variaveis,        # Nomes das variáveis
        main = "Importance",
        #xlab = "Variáveis",
        #ylab = "Importância",
        col = "steelblue",                  # Cor das barras
        las = 2,                            # Rotacionar rótulos no eixo x (2 = vertical)
        cex.names = 1)   


### GRADIENT BOOSTING MACHINE ###

tc = trainControl(method = "cv", 
                  number=10)

fitboosted = train(avalcluster ~ eng + ime + des, 
                   data=treinamento, 
                   method="gbm", 
                   trControl=tc,
                   preProcess = c("scale"))

#TREINAMENTO
boostedclass=predict(fitboosted, 
                     newdata=treinamento)

confusionMatrix(boostedclass,
                treinamento$avalcluster, 
                positive = "1")

boostedclassprob=predict(fitboosted, 
                         newdata=treinamento, 
                         type="prob")

boostedrocobject=roc(treinamento$avalcluster, 
                     boostedclassprob[,2])

plot.roc(boostedrocobject)
auc(boostedrocobject)

#TESTE
boostedclass=predict(fitboosted, 
                     newdata=teste)

confusionMatrix(boostedclass,
                teste$avalcluster, 
                positive = "1")

boostedclassprob=predict(fitboosted, 
                         newdata=teste, 
                         type="prob")

boostedrocobject=roc(teste$avalcluster, 
                     boostedclassprob[,2])

plot.roc(boostedrocobject)
auc(boostedrocobject)
plot(varImp(fitboosted))

### GERAÇÃO DA CURVA ROC PARA O DESEMPENHO DOS TESTES ###

preds_list <- list(KNNclassprob[,2], nbclassprob[,2], RFclassprob[,2], logitclassprob[,2], svmclassprob, boostedclassprob[,2], baggedclassprob[,2])


m <- length(preds_list)
actuals_list <- rep(list(as.numeric(teste$avalcluster)), m)

# Plotando a curva ROC
pred <- prediction(predictions=preds_list, labels=actuals_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "ROC curves for test data")
legend(x = "bottomright", 
       legend = c("KNN", "Naive Bayes","Random Forest","Logit","SVM linear", "Gradient Boosting","Bagging"),
       fill = 1:m)