# Toy model with collinearity

library(MASS)
library(caret)
library(kernlab)
library(ROCR)
library(ggplot2)

# Generation of the dataset X

n_trial <- 5000
x1 <- mvrnorm(n = n_trial, mu = 0, Sigma = 1.5)
x2 <- mvrnorm(n = n_trial, mu = -1, Sigma = 2.)
x3 <- mvrnorm(n = n_trial, mu = 5, Sigma = 1.8)
alpha <- runif(n = n_trial, min = -2.5, max = 2.5)
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
perc_train_set <- 0.8
train_row_idx <- createDataPartition(X$target,
                                     p=perc_train_set,
                                     list = FALSE)

train_set <- X[train_row_idx,]
test_set <- X[-train_row_idx,]

# SVM Kernel
svm_ker <- "vanilladot"

# with correlated variable
svm_model <- ksvm(target ~ x1+x2+x3+x4,
                  data = train_set,
                  kernel = svm_ker,
                  C = 1.0
                  )

# without the correlated variable
svm_model_wox4 <- ksvm(target ~ x1+x2+x3,
                       data = train_set,
                       kernel = svm_ker,
                       C = 1.0
)

test_results <- predict(svm_model, test_set, type = "decision")
test_perf <- ROCR::prediction(predictions = test_results,
                             labels = test_set$target)
test_ROC <- performance(test_perf, measure = "tpr", x.measure = "fpr")

test_ROC_df <- data.frame(unlist(test_ROC@x.values),
                          unlist(test_ROC@y.values))

colnames(test_ROC_df) <- c("fpr","tpr")

test_results_wox4 <- predict(svm_model_wox4, test_set, type = "decision")
test_perf_wox4 <- ROCR::prediction(predictions = test_results_wox4,
                              labels = test_set$target)
test_ROC_wox4 <- performance(test_perf_wox4, measure = "tpr", x.measure = "fpr")

test_ROC_df_wox4 <- data.frame(unlist(test_ROC_wox4@x.values),
                               unlist(test_ROC_wox4@y.values))

colnames(test_ROC_df_wox4) <- c("fpr","tpr")


xline <- seq(0,1,0.02)
yline <- seq(0,1,0.02)
xyline <- data.frame(xline,yline)

ggplot() + 
  geom_line(data=test_ROC_df, aes(x=fpr, y=tpr), color='blue') + 
  geom_line(data=test_ROC_df_wox4, aes(x=fpr, y=tpr), color='red') +
  geom_line(data=xyline, aes(x=xline, y=yline), color='black',linetype = "dashed") +
  xlab("False positive rate") + ylab("True positive rate") +
  ggtitle("ROC")

# Correlations
cor(X[,c(1,2,3,4)])

# Display the correlations
# library(corrgram)
# corrgram(X[,c(1,2,3,4)])