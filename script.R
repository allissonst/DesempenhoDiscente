
library(readxl)
Dados14 <- read_excel("C:/Users/carol/OneDrive/Doutorado/Disciplinas/2024.2/Métodos Quantitativos II/Trabalho/Gamificação/Base/Dados14.xlsx")
View(Dados14)

#Novo Dataframe com as variáveis de interesse

novo_df <- Dados14[, c(2:9,12,23,27,30,34,45,49,52)]
View(novo_df)

# Exibindo o novo data frame
print(novo_df)

#Correlação
novocor_df <- Dados14[, c(2:3,12,23,27,30,34,45,49,52)]

#Pearson
cor(novocor_df)

#Spearman
cor(novocor_df, method = "spearman")

#Correlação incluindo dimensões, notas e estilos de aprendizagem
novocor_df <- Dados14[, c(2:3,9, 12,23,27,30,34,45,49,52)]
df_acomodador <- subset(novocor_df,Estilo==1)
cor(df_acomodador)

#Fazer a correlação para os outros estilos de aprendizagem


##kmeans

set.seed(123) #para manter os mesmos resultados e não mudar os grupos
#Desempenho game
gruposgame <- kmeans(novo_df$game,centers = 2)
gruposgame
gruposgame$cluster[gruposgame$cluster==2]=0 #baixo desempenho

novo_df$desempenhogame = gruposgame$cluster #criar coluna desempenhogame

set.seed(123)
#Desempenho prova
gruposteorica <- kmeans(novo_df$teorica,centers = 2)
gruposteorica
gruposteorica$cluster[gruposteorica$cluster==2]=0 #baixo desempenho

novo_df$desempenhoteorica = gruposteorica$cluster

#Regressão logística 

summary()
Logit_game <- glm(desempenhogame ~ apr_jogo + eng_jogo + ime_jogo + des_jogo, data=novo_df, family = binomial)
summary(Logit_game)

Logit_teorica <- glm(desempenhoteorica ~ apr_prova + eng_prova + ime_prova + des_prova, data=novo_df, family = binomial)
summary(Logit_teorica)

#Regressão

Lm_game <- lm(desempenhogame ~ apr_jogo + eng_jogo + ime_jogo + des_jogo, data=novo_df)
summary(Lm_game)

Lm_teorica <- lm(desempenhoteorica ~ apr_prova + eng_prova + ime_prova + des_prova, data=novo_df)
summary(Lm_teorica)

#Random Forest

library(caTools)
set.seed(1)

divisao=sample.split(novo_df$desempenhogame,SplitRatio = 0.75)
treinamento=subset(novo_df,divisao==T)
teste=subset(novo_df,divisao==F)

treinamento$desempenhogame=as.factor(treinamento$desempenhogame);teste$desempenhogame=as.factor(teste$desempenhogame)

set.seed(1)
ctrl=trainControl(method="cv",
                  number=1000)
FitRF= train(
  desempenhogame ~ apr_jogo + eng_jogo + ime_jogo + des_jogo,
  method="rf",
  preProcess=c("scale"),
  trControl=ctrl,
  data=treinamento,
  importance=T
)

RFclass=predict(FitRF, newdata=teste)
RFclasst=predict(FitRF, newdata=treinamento)

confusionMatrix(RFclass,teste$desempenhogame,positive = "1")
RFclassprob=predict(FitRF, newdata=teste, type="prob")
RFrocobject=roc(teste$desempenhogame, RFclassprob[,2])
plot.roc(RFrocobject)
plot(varImp(FitRF))

13/14

##Dados unificados

library(readxl)
Dados14 <- Dados14 <- read_excel("C:/Users/carol/OneDrive/Doutorado/Disciplinas/2024.2/Métodos Quantitativos II/Trabalho/Gamificação/Base/Dados14.xlsx", 
                                 sheet = "Dados2")
View(Dados14)


#Dataframe
new_df <- Dados14[, c(2:6)]
View(new_df)

#Dataframe sem aprendizagem
df_dados <- Dados14[, c(2,4:6)]
View(df_dados)



#Regressão Múltipla

Lm_geral <- lm(game ~ apr + eng + ime + des, data=new_df)
summary(Lm_geral)


##kmeans

set.seed(123) #para manter os mesmos resultados e não mudar os grupos
#Desempenho game
grupos <- kmeans(df_dados$game,centers = 2)
grupos
grupos$cluster[grupos$cluster==2]=0 #baixo desempenho

df_dados$grupos = grupos$cluster #criar coluna desempenhogame

df_dados
View(df_dados)

#Estatística descritiva com base nos clusters

#cluster 1 - Alto desempenho
summary(df_dados[df_dados$grupos == 1,1:4])
sapply(df_dados[df_dados$grupos == 1, 1:4], sd, na.rm = TRUE)
sapply(df_dados[df_dados$grupos == 1, 1:4], quantile, na.rm = TRUE)

#cluster 0 - Baixo desempenho
summary(df_dados[df_dados$grupos == 0,1:4])
sapply(df_dados[df_dados$grupos == 0, 1:4], sd, na.rm = TRUE)
sapply(df_dados[df_dados$grupos == 0, 1:4], quantile, na.rm = TRUE)



#Regressão Logística

Logit_nota <- glm(grupos ~ apr + eng + ime + des, data=new_df, family = binomial)
summary(Logit_nota)

#Regressão Logística sem aprendizagem
Logit_nota <- glm(grupos ~ eng + ime + des, data=df_dados, family = binomial)
summary(Logit_nota)


#Random Forest

library(caTools)
library(caret)
library(pROC)
set.seed(1)

divisao=sample.split(new_df$grupos,SplitRatio = 0.75)
treinamento=subset(new_df,divisao==T)
teste=subset(new_df,divisao==F)

treinamento$grupos=as.factor(treinamento$grupos);teste$grupos=as.factor(teste$grupos)

set.seed(1)
ctrl=trainControl(method="cv")
FitRF= train(
  grupos ~ eng + ime + des,
  method="rf",
  preProcess=c("scale"),
  trControl=ctrl,
  data=treinamento,
  importance=T
)

RFclass=predict(FitRF, newdata=teste)
RFclasst=predict(FitRF, newdata=treinamento)

confusionMatrix(RFclass,teste$grupos,positive = "1")
RFclassprob=predict(FitRF, newdata=teste, type="prob")
RFrocobject=roc(teste$grupos, RFclassprob[,2])
plot.roc(RFrocobject)
plot(varImp(FitRF))



#APLICAÇÃO KNN (K-Nearest Neighbor, ou K Vizinhos Mais Próximos)
#-----------------------------------------------



ctrl <- trainControl(method="repeatedcv",
                     repeats = 3) 

knnFit <- train(dZSCORE ~ ., 
                data = mettreinamento, 
                method = "knn", trControl = ctrl, 
                preProcess = c("scale"), 
                tuneLength = 20)


knnclass=predict(knnFit, newdata=metteste)
#-----------------------------------------------
#     VN    FP
#     FN    VP
#-----------------------------------------------

confusionMatrix(knnclass,metteste$dZSCORE, positive = "1")
knnclassprob=predict(knnFit, newdata=metteste, type="prob")
rocobject=roc(metteste$dZSCORE, knnclassprob[,2])
plot.roc(rocobject)

auc(rocobject)

#-----------------------------------------------
#     OUTRAS MÉTRICAS DE AVALIAÇÃO - KNN
#-----------------------------------------------

predicttest=as.numeric(knnclass);
original=as.numeric(metteste$dZSCORE);

MAE(predicttest,original)
RMSE(predicttest,original)





