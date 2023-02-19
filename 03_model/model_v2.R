############################################################

# Data Challenge competition
# Budapest, February 2023
# Author: Team Bella Vita

# Aim of this code: To build a predictive models to predict the outcome of football matches using historical data.
# Note: 
# - Data was provided by the organizer of the competition. It is found to be clean, no missing values, no duplicates.
# - There are 23 divisions and 10 seasons in the data.

############################################################  
# BASIC SETUP
############################################################  

# empty memory
rm(list=ls())

# KOMMENT: install library if not exist -> megoldani majd
# csekkolni melyik kell ezek közül
# Load libraries

library("tidyverse") # imap function for roc curve

# library(ggplot2)
# library(ggthemes)

# library(grid)
# library(ggplot2)
# library(lattice)
# library(glmnet)

# install.packages("data.table")

library("data.table") # to use data in an efficient way
library("readxl") # to read xlsx data
library("caret") # to split data into train and holdput sets
library("dplyr") # to append commands easier

# library("skimr") 
# library(gridExtra)
library("ROCR") # to evaluate model performance


# Set the working directory
dir <-  "~/Library/CloudStorage/GoogleDrive-gulyati@gmail.com/My Drive/DataChallenge/"

#location folders
data_in <- paste0(dir,"01_raw/")

# func <- paste0(dir, "work/") - nem használt
output <- paste0(dir, "04_output/")

# load ggplot theme function
# source(paste0(func, "theme_bg.R"))
# # Created a helper function with some useful stuff
# source(paste0(func, "da_helper_functions.R"))
# source(paste0(func, "airbnb_prediction_functions.R"))
# options(digits = 3) 

############################################################  
# LOAD DATA
############################################################  

data <- as.data.table(read_excel(paste0(data_in, "competition_table.xlsx")))


# BASIC STISTICS
message(paste0("Döntetlen aránya: ", round(sum(data$draw_flag) / nrow(data), digits = 2)))
message(paste0("Hazai győzelem aránya: ", round(sum(data$home_win_flag) / nrow(data), digits = 2)))
message(paste0("Vendég győzelem aránya: ", round(sum(data$away_win_flag) / nrow(data), digits = 2)))

############################################################  
# LABEL ENGINEERING
############################################################  

# set data types # -------------------------------------------------
# data$home_win_flag <- as.factor(data$home_win_flag)

data <- data %>%
  mutate(home_win_flag = factor(ifelse(home_win_flag == 0, "no", "yes"), levels = c("no", "yes")))

data <- data %>%
  mutate(away_win_flag = factor(ifelse(away_win_flag == 0, "no", "yes"), levels = c("no", "yes")))

data <- data %>%
  mutate(draw_flag = factor(ifelse(draw_flag == 0, "no", "yes"), levels = c("no", "yes")))

############################################################  
# FEATURE ENGINEERING
############################################################  

# data structure
str(data)

# itt hozzunk majd létre az új változókat



############################################################  
# MODEL FOR PREDICTING HOME WIN
############################################################  

# Create training set and holdout set samples # -------------------------------------------------
 
set.seed(1990)

# smaller than usual training set so that models run faster
train_indices <- createDataPartition(data$home_win_flag, p = 0.7, list = FALSE)
data_train <- data[train_indices, ]

data_holdout <- data[-train_indices, ]

dim(data_train)
dim(data_holdout)

# Define models -------------------------------------------------


# Basic numeric variables
basic_numeric_vars <- c(
  "odds_home_team_win",
  "odds_away_team_win",
  "odds_draw"
) 

# # Log variables
# log_numeric_vars <- c(
# #"ln_accommodates",
# "ln_number_of_reviews",
# #"ln_days_since",
# "ln_review_scores_rating"
# )
# 
# # Higher order variables
# poly_numeric_vars <- c(
#   "n_accommodates2" #,
#   #"ln_accommodates2"
# )
# 
# # Factorized variables
# factor_vars <- c(
#   "f_room_type",
#   "f_cancellation_policy",
#   "f_bed_type"
# )
# 
# # Dummy variables
# amenities <-  grep("^d_.*", names(data), value = TRUE) 

#######################################################
predictors_1 <- c(basic_numeric_vars)
predictors_1 <- colnames(data[,-c("home_win_flag", "draw_flag", "away_win_flag")])
# predictors_2 <- c(basic_numeric_vars, log_numeric_vars)
# predictors_3 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars)
# predictors_4 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars, factor_vars)
# predictors_5 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars, factor_vars, amenities)

#######################################################
# set evaluation rules

# train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

train_control <- trainControl(
  method = "cv", 
  n = 5,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary  # necessary!
)

#######################################################

# Simple logistic regression
# -------------------------------------------------
set.seed(857)
glm_model <- train(formula(paste0("home_win_flag ~", paste0(predictors_1, collapse = " + "))),
                   method = "glm",
                   data = data_train,
                   trControl = train_control)

# AUC on training setß
glm_model
varImp(glm_model)
varimp <- varImp(glm_model)
# compare model performance based on CV using AUC
summary(glm_model)

# the previously seen CV AUC
auc_cv <- glm_model[["results"]][["ROC"]]
message(paste0("AUC from CV: ", round(auc_cv, digits = 4)))


# variable importance
plot(varImp(glm_model[1:10]))

# AUC on holdout set
predicted_glm_holdout <- predict(glm_model, 
                                 newdata = data_holdout, 
                                 type = "prob")[["yes"]]


prediction_holdout_glm <- prediction(predicted_glm_holdout, data_holdout$home_win_flag)
auc_holdout_glm <- performance(prediction_holdout_glm, measure = "auc")@y.values[[1]]
message(paste0("AUC on holdout set for GLM: ", round(auc_holdout_glm, digits = 4)))
# -------------------------------------------------

# gradient boosting
# -------------------------------------------------

library("xgboost")
set.seed(19900829)
tune_grid <- expand.grid(
  nrounds = c(100),  # this is n_estimators in the python code above
  max_depth = c(10),
  colsample_bytree = seq(0.5, 0.9, length.out = 5),
  ## The values below are default values in the sklearn-api. 
  eta = 0.1,
  gamma=0,
  min_child_weight = 1,
  subsample = 1
)


xgb <- train(
  formula(paste0("home_win_flag ~", paste0(predictors_1, collapse = " + "))),
  tuneLength = 1,
  data = data_train, 
  method = 'xgbTree',
  na.action = na.omit,
  # importance = 'permutation',
  # mtry: cannot be more than the number of predictors 
  tuneGrid = tune_grid,
  trControl = train_control)

xgb

# Naive Bayes
# -------------------------------------------------
library("e1071")
# Create a preprocessing pipeline
# preprocess <- preProcess(data_train, method = c("center", "scale"))
# 
# # Apply the preprocessing to the training and testing sets
# train_data_processed <- predict(preprocess, data_train)
# test_data_processed <- predict(preprocess, data_holdout)

# Define the control parameters for the train function
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, 
                     classProbs = TRUE, summaryFunction = twoClassSummary)

# Train the Naive Bayes model using the train function
nb_model <- train(home_win_flag ~ ., data = data_train[,-c("away_win_flag", "draw_flag")], 
                  method = "naive_bayes",
                  trControl = ctrl, verbose = FALSE)

# SVMß
# -------------------------------------------------

# rf_model_1
# -------------------------------------------------
set.seed(19900829)
tune_grid <- expand.grid(
  .mtry = 3,
  .splitrule = "gini",
  .min.node.size = c(5) # implicitly sets the depth of your trees
)


rf_model_1 <- train(
  formula(paste0("home_win_flag ~", paste0(predictors_1, collapse = " + "))),
  tuneLength = 1,
  data = data_train, 
  method = 'ranger',
  na.action = na.omit,
  importance = 'permutation',
  # mtry: cannot be more than the number of predictors 
  tuneGrid = tune_grid,
  trControl = train_control)


# EVALUATION, model diagnostics -------------------------------------------------

# AUC on training setß
rf_model_1
varImp(rf_model_1)
varimp <- varImp(rf_model_1)
# summary(rf_model_1)


# variable importance
plot(varImp(rf_model_1[1:10]))

# AUC on holdout set
predicted_rf1_holdout <- predict(rf_model_1, 
                         newdata = data_holdout, 
                         type = "prob")[["yes"]]


prediction_holdout_rf1 <- prediction(predicted_rf1_holdout, data_holdout$home_win_flag)
auc_holdout_rf <- performance(prediction_holdout_rf1, measure = "auc")@y.values[[1]]
message(paste0("AUC on holdout set for RF: ", round(auc_holdout_rf, digits = 4)))

# ROC curve on the holdout set -------------------------------------------------

roc_df_logit_rf <- imap(list("random forest" = rf_model_1), ~ {
  model <- .x
  predicted_probabilities <- predict(model, newdata = data_holdout, type = "prob")
  rocr_prediction <- prediction(predicted_probabilities[["yes"]], data_holdout[["home_win_flag"]]) 
  
  tpr_fpr_performance <- performance(rocr_prediction, "tpr", "fpr")
  tibble(
    model = .y,
    FPR = tpr_fpr_performance@x.values[[1]],
    TPR = tpr_fpr_performance@y.values[[1]],
    cutoff = tpr_fpr_performance@alpha.values[[1]]
  )  
}) %>% bind_rows()

ggplot(roc_df_logit_rf) +
  geom_line(aes(FPR, TPR, color = model), size = 1) +
  geom_abline(intercept = 0, slope = 1,  linetype = "dotted", col = "black") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab("False Positive Rate") + ylab("True Positive Rate")




# calibration plot: how well do estimated vs actual event probabilities relate to each other?
# -------------------------------------------------
actual_vs_predicted_rf1 <- data.frame(
  actual = ifelse(data_holdout[["home_win_flag"]] == "yes", 1, 0),
  predicted = predicted_rf1_holdout
)

# order observations accordingly

num_groups <- 10

calibration_logit <- actual_vs_predicted_rf1 %>% 
  # ntile: assign group ids based on a variable. similar to "cut".
  mutate(predicted_score_group = ntile(predicted, num_groups)) %>% 
  group_by(predicted_score_group) %>% 
  summarise(mean_actual = mean(actual), mean_predicted = mean(predicted), num_obs = n())

ggplot(calibration_logit,aes(x = mean_actual, y = mean_predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Actual event probability", y = "Predicted event probability") +
  ylim(0, 1) + xlim(0, 1)


# the Brier score: MSE applied to classification evaluation

brier <- RMSE(actual_vs_predicted_rf1[["predicted"]], actual_vs_predicted_rf1[["actual"]])^2
message(paste0("Brier score: ", round(brier, digits = 4)))



# Compare random forests models based on training sets -------------------------------------------------

results <- resamples(
  list(
    model_1 = rf_model_1,
    model_2 = rf_model_1
    # model_3 = rf_model_3,
    # model_4 = rf_model_4,
    # model_5 = rf_model_5
  )
)
# summary(results)







############################################################  
# MODEL FOR PREDICTING AWAY WIN
############################################################  

# Create training set and holdout set samples # -------------------------------------------------

set.seed(1990)

# smaller than usual training set so that models run faster
train_indices <- createDataPartition(data$away_win_flag, p = 0.7, list = FALSE)
data_train <- data[train_indices, ]

data_holdout <- data[-train_indices, ]

dim(data_train)
dim(data_holdout)

# Define models -------------------------------------------------


# Basic numeric variables
basic_numeric_vars <- c(
  "odds_home_team_win",
  "odds_away_team_win",
  "odds_draw"
) 

# # Log variables
# log_numeric_vars <- c(
# #"ln_accommodates",
# "ln_number_of_reviews",
# #"ln_days_since",
# "ln_review_scores_rating"
# )
# 
# # Higher order variables
# poly_numeric_vars <- c(
#   "n_accommodates2" #,
#   #"ln_accommodates2"
# )
# 
# # Factorized variables
# factor_vars <- c(
#   "f_room_type",
#   "f_cancellation_policy",
#   "f_bed_type"
# )
# 
# # Dummy variables
# amenities <-  grep("^d_.*", names(data), value = TRUE) 

#######################################################
predictors_1 <- c(basic_numeric_vars)
predictors_1 <- colnames(data[,-c("home_win_flag", "draw_flag", "away_win_flag")])
# predictors_2 <- c(basic_numeric_vars, log_numeric_vars)
# predictors_3 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars)
# predictors_4 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars, factor_vars)
# predictors_5 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars, factor_vars, amenities)

#######################################################
# set evaluation rules

train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

train_control <- trainControl(
  method = "cv", 
  n = 10,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary  # necessary!
)

#######################################################

set.seed(19900829)

# rf_model_1
tune_grid <- expand.grid(
  .mtry = 3,
  .splitrule = "gini",
  .min.node.size = c(5) # implicitly sets the depth of your trees
)


rf_model_1 <- train(
  formula(paste0("away_win_flag ~", paste0(predictors_1, collapse = " + "))),
  tuneLength = 1,
  data = data_train, 
  method = 'ranger',
  na.action = na.omit,
  importance = 'permutation',
  # mtry: cannot be more than the number of predictors 
  tuneGrid = tune_grid,
  trControl = train_control)


# EVALUATION, model diagnostics -------------------------------------------------

# AUC on training setß
rf_model_1
varImp(rf_model_1)
varimp <- varImp(rf_model_1)
# summary(rf_model_1)


# variable importance
plot(varImp(rf_model_1))

# AUC on holdout set
predicted_rf1_holdout <- predict(rf_model_1, 
                                 newdata = data_holdout, 
                                 type = "prob")[["yes"]]


prediction_holdout_rf1 <- prediction(predicted_rf1_holdout, data_holdout$away_win_flag)
auc_holdout_rf <- performance(prediction_holdout_rf1, measure = "auc")@y.values[[1]]
message(paste0("AUC on holdout set for RF: ", round(auc_holdout_rf, digits = 4)))

# ROC curve on the holdout set -------------------------------------------------

roc_df_logit_rf <- imap(list("random forest" = rf_model_1), ~ {
  model <- .x
  predicted_probabilities <- predict(model, newdata = data_holdout, type = "prob")
  rocr_prediction <- prediction(predicted_probabilities[["yes"]], data_holdout[["away_win_flag"]]) 
  
  tpr_fpr_performance <- performance(rocr_prediction, "tpr", "fpr")
  tibble(
    model = .y,
    FPR = tpr_fpr_performance@x.values[[1]],
    TPR = tpr_fpr_performance@y.values[[1]],
    cutoff = tpr_fpr_performance@alpha.values[[1]]
  )  
}) %>% bind_rows()

ggplot(roc_df_logit_rf) +
  geom_line(aes(FPR, TPR, color = model), size = 1) +
  geom_abline(intercept = 0, slope = 1,  linetype = "dotted", col = "black") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab("False Positive Rate") + ylab("True Positive Rate")




# calibration plot: how well do estimated vs actual event probabilities relate to each other?
# -------------------------------------------------
actual_vs_predicted_rf1 <- data.frame(
  actual = ifelse(data_holdout[["away_win_flag"]] == "yes", 1, 0),
  predicted = predicted_rf1_holdout
)

# order observations accordingly

num_groups <- 10

calibration_logit <- actual_vs_predicted_rf1 %>% 
  # ntile: assign group ids based on a variable. similar to "cut".
  mutate(predicted_score_group = ntile(predicted, num_groups)) %>% 
  group_by(predicted_score_group) %>% 
  summarise(mean_actual = mean(actual), mean_predicted = mean(predicted), num_obs = n())

ggplot(calibration_logit,aes(x = mean_actual, y = mean_predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Actual event probability", y = "Predicted event probability") +
  ylim(0, 1) + xlim(0, 1)


# the Brier score: MSE applied to classification evaluation

brier <- RMSE(actual_vs_predicted_rf1[["predicted"]], actual_vs_predicted_rf1[["actual"]])^2
message(paste0("Brier score: ", round(brier, digits = 4)))


############################################################  
# MODEL FOR PREDICTING DRAW
############################################################  

# Create training set and holdout set samples # -------------------------------------------------

set.seed(1990)

# smaller than usual training set so that models run faster
train_indices <- createDataPartition(data$draw_flag, p = 0.7, list = FALSE)
data_train <- data[train_indices, ]

data_holdout <- data[-train_indices, ]

dim(data_train)
dim(data_holdout)

# Define models -------------------------------------------------


# Basic numeric variables
basic_numeric_vars <- c(
  "odds_home_team_win",
  "odds_away_team_win",
  "odds_draw"
) 

# # Log variables
# log_numeric_vars <- c(
# #"ln_accommodates",
# "ln_number_of_reviews",
# #"ln_days_since",
# "ln_review_scores_rating"
# )
# 
# # Higher order variables
# poly_numeric_vars <- c(
#   "n_accommodates2" #,
#   #"ln_accommodates2"
# )
# 
# # Factorized variables
# factor_vars <- c(
#   "f_room_type",
#   "f_cancellation_policy",
#   "f_bed_type"
# )
# 
# # Dummy variables
# amenities <-  grep("^d_.*", names(data), value = TRUE) 

#######################################################
predictors_1 <- c(basic_numeric_vars)
predictors_1 <- colnames(data[,-c("home_win_flag", "draw_flag", "away_win_flag")])
# predictors_2 <- c(basic_numeric_vars, log_numeric_vars)
# predictors_3 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars)
# predictors_4 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars, factor_vars)
# predictors_5 <- c(basic_numeric_vars, log_numeric_vars, poly_numeric_vars, factor_vars, amenities)

#######################################################
# set evaluation rules

train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

train_control <- trainControl(
  method = "cv", 
  n = 10,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary  # necessary!
)

#######################################################

set.seed(19900829)

# rf_model_1
tune_grid <- expand.grid(
  .mtry = 3,
  .splitrule = "gini",
  .min.node.size = c(5) # implicitly sets the depth of your trees
)


rf_model_1 <- train(
  formula(paste0("draw_flag ~", paste0(predictors_1, collapse = " + "))),
  tuneLength = 1,
  data = data_train, 
  method = 'ranger',
  na.action = na.omit,
  importance = 'permutation',
  # mtry: cannot be more than the number of predictors 
  tuneGrid = tune_grid,
  trControl = train_control)


# EVALUATION, model diagnostics -------------------------------------------------

# AUC on training setß
rf_model_1
varImp(rf_model_1)
varimp <- varImp(rf_model_1)
# summary(rf_model_1)


# variable importance
plot(varImp(rf_model_1))

# AUC on holdout set
predicted_rf1_holdout <- predict(rf_model_1, 
                                 newdata = data_holdout, 
                                 type = "prob")[["yes"]]


prediction_holdout_rf1 <- prediction(predicted_rf1_holdout, data_holdout$draw_flag)
auc_holdout_rf <- performance(prediction_holdout_rf1, measure = "auc")@y.values[[1]]
message(paste0("AUC on holdout set for RF: ", round(auc_holdout_rf, digits = 4)))

# ROC curve on the holdout set -------------------------------------------------

roc_df_logit_rf <- imap(list("random forest" = rf_model_1), ~ {
  model <- .x
  predicted_probabilities <- predict(model, newdata = data_holdout, type = "prob")
  rocr_prediction <- prediction(predicted_probabilities[["yes"]], data_holdout[["draw_flag"]]) 
  
  tpr_fpr_performance <- performance(rocr_prediction, "tpr", "fpr")
  tibble(
    model = .y,
    FPR = tpr_fpr_performance@x.values[[1]],
    TPR = tpr_fpr_performance@y.values[[1]],
    cutoff = tpr_fpr_performance@alpha.values[[1]]
  )  
}) %>% bind_rows()

ggplot(roc_df_logit_rf) +
  geom_line(aes(FPR, TPR, color = model), size = 1) +
  geom_abline(intercept = 0, slope = 1,  linetype = "dotted", col = "black") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1)) +
  xlab("False Positive Rate") + ylab("True Positive Rate")




# calibration plot: how well do estimated vs actual event probabilities relate to each other?
# -------------------------------------------------
actual_vs_predicted_rf1 <- data.frame(
  actual = ifelse(data_holdout[["draw_flag"]] == "yes", 1, 0),
  predicted = predicted_rf1_holdout
)

# order observations accordingly

num_groups <- 10

calibration_logit <- actual_vs_predicted_rf1 %>% 
  # ntile: assign group ids based on a variable. similar to "cut".
  mutate(predicted_score_group = ntile(predicted, num_groups)) %>% 
  group_by(predicted_score_group) %>% 
  summarise(mean_actual = mean(actual), mean_predicted = mean(predicted), num_obs = n())

ggplot(calibration_logit,aes(x = mean_actual, y = mean_predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Actual event probability", y = "Predicted event probability") +
  ylim(0, 1) + xlim(0, 1)


# the Brier score: MSE applied to classification evaluation

brier <- RMSE(actual_vs_predicted_rf1[["predicted"]], actual_vs_predicted_rf1[["actual"]])^2
message(paste0("Brier score: ", round(brier, digits = 4)))
