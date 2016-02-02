# Toy model with collinearity
#
# With this code we want to test the claim
# that SVM is extremely robust aginst the problem of multicollinearity
# as stated for example at pag.190 of "Machine Learning with R Cookbook" (Di Yu-Wei, Chiu):
# "It also does not suffer from local optima and multicollinearity"

library(MASS)
library(caret)
library(kernlab)
library(ROCR)
library(ggplot2)

# Generation of the dataset X

n_trial <- 3000
x1 <- mvrnorm(n = n_trial, mu = 0, Sigma = 1.5)
x2 <- mvrnorm(n = n_trial, mu = -1, Sigma = 1.2)
x3 <- mvrnorm(n = n_trial, mu = 5, Sigma = 1.4)
alpha <- runif(n = n_trial, min = -1.4, max = 1.4)
#alpha <- 1
x4 <- alpha + x3 # almost linear relation

X <- data.frame(x1,x2,x3,x4)

func_gen <- function(row) {
  var1 <- row[1]
  var2 <- row[2]
  var3 <- row[3]
  var4 <- row[4]
  return(var1**2-var1*var2*var3+0.2*var3^2-0.5*var4**2)
} 

target_cont <- apply(X,1,func_gen)

func_thrs <- function(value,thrsh) {
  if (value>=thrsh) return(1)
  else {
    return(0)
  }
}

X$target_cont <- target_cont

target <- lapply(target_cont, function(x) func_thrs(x,2.))
X$target <- as.factor(unlist(target))


# Build some model

set.seed(1984)
perc_train_set <- 0.7
train_row_idx <- createDataPartition(X$target,
                                     p=perc_train_set,
                                     list = FALSE)

train_set <- X[train_row_idx,]
test_set <- X[-train_row_idx,]

# SVM Kernel
#svm_ker <- "vanilladot"
svm_ker <- "rbfdot"

# with correlated variable
svm_model <- ksvm(target ~ x1+x2+x3+x4,
                  data = train_set,
                  kernel = svm_ker,
                  C = 1
                  )

# without the correlated variable
svm_model_wox4 <- ksvm(target ~ x1+x2+x3,
                       data = train_set,
                       kernel = svm_ker,
                       C = 1
)

# comparison with logistic regression
logit_model <- glm(target ~ x1+x2+x3+x4,
                   data = train_set, 
                   binomial(link=logit))

# without the correlated variable
logit_model_wox4 <- glm(target ~ x1+x2+x3,
                   data = train_set, 
                   binomial(link=logit))

test_results <- predict(svm_model, test_set, type = "decision")
test_perf <- ROCR::prediction(predictions = test_results,
                             labels = test_set$target)
test_ROC <- performance(test_perf, measure = "tpr", x.measure = "fpr")
test_AUC <- performance(test_perf, measure = "auc")@y.values[[1]]
test_ROC_df <- data.frame(unlist(test_ROC@x.values),
                          unlist(test_ROC@y.values))

colnames(test_ROC_df) <- c("fpr","tpr")

test_results_wox4 <- predict(svm_model_wox4, test_set, type = "decision")
test_perf_wox4 <- ROCR::prediction(predictions = test_results_wox4,
                              labels = test_set$target)
test_ROC_wox4 <- performance(test_perf_wox4, measure = "tpr", x.measure = "fpr")
test_AUC_wox4 <- performance(test_perf_wox4, measure = "auc")@y.values[[1]]
test_ROC_df_wox4 <- data.frame(unlist(test_ROC_wox4@x.values),
                               unlist(test_ROC_wox4@y.values))

colnames(test_ROC_df_wox4) <- c("fpr","tpr")

test_results_logit <- predict(logit_model, test_set, type = "response")
test_perf_logit <- ROCR::prediction(predictions = test_results_logit,
                              labels = test_set$target)
test_ROC_logit <- performance(test_perf_logit, measure = "tpr", x.measure = "fpr")
test_AUC_logit <- performance(test_perf_logit, measure = "auc")@y.values[[1]]
test_ROC_df_logit <- data.frame(unlist(test_ROC_logit@x.values),
                                unlist(test_ROC_logit@y.values))

colnames(test_ROC_df_logit) <- c("fpr","tpr")

test_results_logit_wox4 <- predict(logit_model_wox4, test_set, type = "response")
test_perf_logit_wox4 <- ROCR::prediction(predictions = test_results_logit_wox4,
                                    labels = test_set$target)
test_ROC_logit_wox4 <- performance(test_perf_logit_wox4, measure = "tpr", x.measure = "fpr")
test_AUC_logit_wox4 <- performance(test_perf_logit_wox4, measure = "auc")@y.values[[1]]
test_ROC_df_logit_wox4 <- data.frame(unlist(test_ROC_logit_wox4@x.values),
                                     unlist(test_ROC_logit_wox4@y.values))

colnames(test_ROC_df_logit_wox4) <- c("fpr","tpr")

xline <- seq(0,1,0.02)
yline <- seq(0,1,0.02)
xyline <- data.frame(xline,yline)

ggplot() + 
  geom_line(data=test_ROC_df, aes(x=fpr, y=tpr, color='svm w x4',
                                  linetype = 'svm w x4')) + 
  geom_line(data=test_ROC_df_wox4, aes(x=fpr, y=tpr, 
                                       color='svm wo x4',linetype = 'svm wo x4')) +
  geom_line(data=test_ROC_df_logit, aes(x=fpr, y=tpr, 
                                        color='logit w x4',
                                        linetype = 'logit w x4')) +
  geom_line(data=test_ROC_df_logit_wox4, aes(x=fpr, y=tpr, 
                                             color='logit wo x4',
                                             linetype = 'logit wo x4')) +
  geom_line(data=xyline, aes(x=xline, y=yline), color='black',linetype = "dashed") +
  xlab("FPR") + ylab("TPR") +
  scale_colour_manual("Models",
                      values=c("svm w x4"="blue", 
                               "svm wo x4"="blue", 
                               "logit w x4"="green",
                               "logit wo x4"="green")) +
  scale_linetype_manual("Models",values = c("svm w x4"=1, 
                                   "svm wo x4"=8,
                                   "logit w x4"=1, 
                                   "logit wo x4"=8)) +
  ggtitle("ROC")

# AUC
print(paste("AUC (svm w x4) -->",format(test_AUC*100,digits = 4),"%"))
print(paste("AUC (svm wo x4) -->",format(test_AUC_wox4*100,digits = 4),"%"))
print(paste("AUC (logit w x4) -->",format(test_AUC_logit*100,digits = 4),"%"))
print(paste("AUC (logit wo x4) -->",format(test_AUC_logit_wox4*100,digits = 4),"%"))

# Correlations
cor(X[,c(1,2,3,4)])

# Display the correlations
# library(corrgram)
# corrgram(X[,c(1,2,3,4)])

# Distributions of each single var
# ggplot() + 
#   geom_density(fill = "blue", alpha = 0.5, aes(x=x1), data=X) +
#   geom_density(fill = "blue", alpha = 0.5, aes(x=x2), data=X) +
#   geom_density(fill = "blue", alpha = 0.5, aes(x=x3), data=X) +
#   geom_density(fill = "red", alpha = 0.5, aes(x=x4), data=X)