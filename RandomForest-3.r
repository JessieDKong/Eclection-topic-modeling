# Dataset 3, project 1, Random Forest

setwd("/Users/laetetia/Desktop/Queens/873/dataset3/3/project8package")
Sys.setlocale(locale="C")
# read in the 1000 words as a vector
#mostfreq1000words <- readLines("mostfreq1000word.csv", encoding="UTF-16LE")
mostfreq1000words <- readLines("mostfreq1000word.csv")

# Deal with special characters
mostfreq1000words <- gsub("$", "DOLLARSIGN", mostfreq1000words, fixed=TRUE)
mostfreq1000words <- gsub("'", "SINGLEQUOTE", mostfreq1000words, fixed=TRUE)
mostfreq1000words <- gsub("\"", "DOUBLEQUOTE", mostfreq1000words, fixed=TRUE)
mostfreq1000words <- gsub("([0-9]+)", "NUM\\1", mostfreq1000words, perl=TRUE)
mostfreq1000words <- gsub("-", "DASH", mostfreq1000words, fixed=TRUE)

# Deal with other special characters
special_idx = 0
for (i in 1:length(mostfreq1000words)) {
  # check whether the word contains special characters
  if (grepl("[^a-zA-Z0-9_]+", mostfreq1000words[i])) {
    special_idx <- special_idx + 1
    special_sub <- paste("SPECIAL", special_idx, sep="")
    POS <- unlist(strsplit(mostfreq1000words[i], "_"))[2]
    mostfreq1000words[i] <- paste(special_sub, POS, sep="_")
  }
    
}

#mostfreq1000words <- as.vector(mostfreq1000words)

# read in the the occurance rate of the words in each speech
freqwords_rate <- read.csv("mostfreq1000docword.csv", header = FALSE)
colnames(freqwords_rate) <- mostfreq1000words

####----------------------- Topic modeling --------------------------####
speech_length <- 10000
freqwords_freq <- floor(freqwords_rate * speech_length)
excludewords <- c("and", "the", "for", "to", "a", "of", "will",
                  "i", "you", "they", "in", "we", "that", "our",
                  "this", "is", "who", "he", "what", "have", "are",
                  "going", "with", "want", "on", "their", "your", "it", "but",
                  "my", "on", "be", "all", "their", "as", "can", "do", "or", "when",
                  "by", "thatSINGLEQUOTEs", "if", "from", "was", "make", "at", "he", "has",
                  "now", "us", "because", "more", "itSINGLEQUOTEs", "just", "new", "an", 
                  "about", "here", "up", "know", "them", "time", "me", "need", 
                  "every", "weSINGLEQUOTEre", "been", "get", "one", "than", "iSINGLEQUOTEm", "those",
                  "out", "in", "donSINGLEQUOTEt", "some", "weSINGLEQUOTEve", "got", "so", "would", 
                  "should", "back", "there", "his", "were", "had", "no", "not",
                  "good", "go", "way", "where", "also", "said", "like", "give", "many",
                  "why", "must", "last", "great", "say", "today", "other", "take",
                  "how", "into", "canSINGLEQUOTEt", "down", "keep", "thank", "same", "over",
                  "let", "own", "iSINGLEQUOTEve", "first", "see", "which", "come", "still",
                  "youSINGLEQUOTEre", "then", "even", "000", "four", "day", "any", "DOUBLEQUOTE", "middle", "create",
                  "next", "weSINGLEQUOTEll", "lot", "two", "again", "about", "sure", "small", "big",
                  "well", "did", "done", "too", "only", "its", "tell", "am", "most", "years",
                  "very", "plan", "believe", "think", "things")

#excludewords <- lapply(excludewords, function(x) paste(x, "_", sep=""))
#excludewords <- unlist(excludewords)

excludewords_columns <- unlist(lapply(colnames(freqwords_freq), 
                                      function(x) (unlist(strsplit(x, "_"))[1] %in% excludewords)))

freqwords_freq <- freqwords_freq[, -which(excludewords_columns)]
#### Topic modeling
library(topicmodels)

#Set parameters for Gibbs sampling
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE

#Number of topics
k <- 5

ldaOut <-LDA(as.matrix(freqwords_freq), k, method="VEM")

save.image("after_LDA_VEM")
			 
#write out results
#docs to topics
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))

#top 6 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,10))
write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))


#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))
####---------------------------------------------------------------------####
			 

# read in the binary data that indicates each speech is associated with a win or loss
win <- as.logical(as.numeric(readLines("winners.csv")))
win <- as.factor(win)

# combine the words occurance rates and the associated results
#data <- cbind(win, freqwords_rate)
colnames(topicProbabilities) <- c("Topic1", "Topic2", "Topic3", "Topic4", "Topic5")
data <- cbind(win, topicProbabilities)

# random forest
library(randomForest)
rf.fit <- randomForest(win ~ ., data = data, 
                       ntree=500, type='classification', importance=TRUE)
rf.fit

# plot the error rate versus number of trees
plot(rf.fit)

# plot the important variables
varImpPlot(rf.fit, n.var=5)

#importance <- importance(rf.fit, type=1, class="TRUE",scale=FALSE)

#if (0) {
## --------------------------------------------------------------##
## --------------10 folds cross validation-----------------------##
k = 10 # Folds

predictions <- data.frame()
trainingset <- data.frame()
testset <- data.frame()

precision_vector <- c()
recall_vector <- c()
accuracy_vector <- c()
AUC_vector <- c()
importance_collection <- data.frame()

library(caret)
require(ROSE)
folds <- createFolds(data$win, k = 10, list = TRUE, returnTrain=TRUE)
i = 1
for(inTrain in folds){
  trainingset <- data[inTrain,]
  testset <- data[-inTrain,]
  rf.fit <- randomForest(win ~ ., data = trainingset, 
                         ntree=100, type='classification', importance=TRUE)
  predictions <- predict(rf.fit, testset, type="response")
  predictionsProb <- predict(rf.fit, testset, type="prob")
  
  TP = sum((as.logical(predictions)) & (as.logical(testset$win)))
  precision = TP / sum(as.logical(predictions))
  recall = TP / sum(as.logical(testset$win))
  precision_vector <- c(precision_vector, precision)
  recall_vector <- c(recall_vector, recall)
  accuracy <- sum(as.logical(predictions) == as.logical(testset$win)) / length(predictions)
  accuracy_vector <- c(accuracy_vector, accuracy)
  rocValue=roc.curve(testset$win, 
                     predictionsProb[,c("TRUE")])
  AUC_vector <- c(AUC_vector, rocValue$auc)
  importance <- importance(rf.fit, type=1, class="TRUE",scale=FALSE)
  importance <- as.data.frame(importance)
  names(importance)[1] <- paste("Fold", i, sep="#")
  if (i == 1) {
    importance_collection <- importance
  } else {
    importance_collection <- merge(importance_collection, importance, by="row.names")
    if ("Row.names" %in% names(importance_collection)) {
      row.names(importance_collection) <- importance_collection$Row.names
      importance_collection$Row.names <- NULL
    }
  }
  print("Folds number: ")
  print(i)
  print("Precision:")
  print(precision)
  print("Recall: ")
  print(recall)
  print("Accuracy: ")
  print(accuracy)#print("AUC: ")
  print("AUC: ")
  print(rocValue)
  #break # for debug
  i = i + 1
}

avg_precision = mean(precision_vector)
avg_recall = mean(recall_vector)
avg_accuracy <- mean(accuracy_vector)
print("Average precision value: ")
print(avg_precision)
print("Average recall value: ")
print(avg_recall)
print("Average accuracy value: ")
print(avg_accuracy)
print("AUC Distribution, mean: ")
print(mean(AUC_vector))

boxplot(precision_vector, main="Precision distribution", ylim=c(0,1))
boxplot(recall_vector, main="Recall distribution", ylim=c(0,1))
boxplot(accuracy_vector, main="Accuracy distribution", ylim=c(0,1))
boxplot(AUC_vector, main="AUC distrubtion", ylim=c(0,1))
#}


## Statistically analyze and Visulize the importance using ScottKnott
require(ScottKnott)
require(reshape2)

## Using all the words

importance_copy <- importance_collection
importance_copy$metrics <- row.names(importance_copy)
importance_reshaped <- melt(importance_copy, id=c("metrics"))
sk <- with(importance_reshaped,
           SK(x=importance_reshaped, 
              model='value ~ metrics', which='metrics', dispersion='s'))

plot(sk,
     col=rainbow(max(sk$groups)),
     title="Importance of metrics",
     xlab="",
     ylab="Importance",
     mm.lty=3,
     yaxt="n",
     #axes=FALSE,
     #frame.plot=TRUE,
     rl=FALSE,
     las=2,
     id.col=FALSE)


## using only the top 20 variables
top_num <- 20
avg_importance <- apply(importance_copy[, 1:10], 1, mean)
# sort the words according to the average importance
importance_copy <- importance_copy[order(avg_importance, decreasing=TRUE), ]
top_importance <- importance_copy[1:top_num,]
importance_reshaped <- melt(top_importance, id=c("metrics"))
sk <- with(importance_reshaped,
           SK(x=importance_reshaped, 
              model='value ~ metrics', which='metrics', dispersion='s'))
opar <- par(no.readonly=TRUE)
par(mar=c(5,4,4,2)+1.5)
plot(sk,
     col=rainbow(max(sk$groups)),
     title="Top 20 Importance Words",
     xlab="",
     ylab="Importance",
     mm.lty=3,
     yaxt="n",
     #axes=FALSE,
     #frame.plot=TRUE,
     rl=FALSE,
     las=2,
     id.col=FALSE)
par(opar)
## --------------------------------------------------------------##
## --------------cross-president validation----------------------##
# get all the speeches
speeches <- readLines("speeches.csv")
# extract the speaker of each speech
speakers <- gsub("([0-9]+)([a-zA-Z]+)([0-9]+)(.*)", "\\2", speeches, perl=TRUE)
distinct_speakers <- names(table(speakers))

trainingset <- data.frame()
testset <- data.frame()

predictions <- data.frame()
precision_vector <- c()
recall_vector <- c()
accuracy_vector <- c()
#AUC_vector <- c()
importance_collection <- data.frame()

library(caret)
require(ROSE)
i = 1
for(speaker in distinct_speakers){
  trainingset <- data[-which(speakers==speaker),]
  testset <- data[which(speakers==speaker),]
  rf.fit <- randomForest(win ~ ., data = trainingset, 
                         ntree=100, type='classification', importance=TRUE)
  predictions <- predict(rf.fit, testset, type="response")
  predictionsProb <- predict(rf.fit, testset, type="prob")
  
  TP = sum((as.logical(predictions)) & (as.logical(testset$win)))
  precision = TP / sum(as.logical(predictions))
  recall = TP / sum(as.logical(testset$win))
  precision_vector <- c(precision_vector, precision)
  recall_vector <- c(recall_vector, recall)
  accuracy <- sum(as.logical(predictions) == as.logical(testset$win)) / length(predictions)
  accuracy_vector <- c(accuracy_vector, accuracy)
  #rocValue=roc.curve(testset$win, 
  #                   predictionsProb[,c("TRUE")])
  #AUC_vector <- c(AUC_vector, rocValue$auc)
  importance <- importance(rf.fit, type=1, class="TRUE",scale=FALSE)
  importance <- as.data.frame(importance)
  names(importance)[1] <- paste("Fold", i, sep="#")
  if (i == 1) {
    importance_collection <- importance
  } else {
    importance_collection <- merge(importance_collection, importance, by="row.names")
    if ("Row.names" %in% names(importance_collection)) {
      row.names(importance_collection) <- importance_collection$Row.names
      importance_collection$Row.names <- NULL
    }
  }
  print("Folds number: ")
  print(i)
  print(speaker)
  print("Precision:")
  print(precision)
  print("Recall: ")
  print(recall)
  print("Accuracy: ")
  print(accuracy)#print("AUC: ")
  #print(rocValue)
  #break # for debug
  i = i + 1
}

avg_precision = mean(precision_vector)
avg_recall = mean(recall_vector)
avg_accuracy <- mean(accuracy_vector)
print("Average precision value: ")
print(avg_precision)
print("Average recall value: ")
print(avg_recall)
print("Average accuracy value: ")
print(avg_accuracy)
#print("AUC Distribution, mean: ")
#print(mean(AUC_vector))

boxplot(precision_vector, main="Precision distribution", ylim=c(0,1))
boxplot(recall_vector, main="Recall distribution", ylim=c(0,1))
boxplot(accuracy_vector, main="Accuracy distribution", ylim=c(0,1))
#boxplot(AUC_vector, main="AUC distrubtion", ylim=c(0,1))


##--------------Logistic Regression -------------------##
require(glm)
l_model <- glm(win ~ ., data=data, family=binomial())
coefficients(l_model)
