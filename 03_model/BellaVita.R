# Load packages

# Load home win models
home_model_rf <- readRDS(paste0(output,"home_model_rf.rda"))
home_model_glm <- readRDS(paste0(output,"home_model_glm.rda"))
home_model_gbm <- readRDS(paste0(output,"home_model_gbm.rda"))
home_model_xgb <- readRDS(paste0(output,"home_model_xgb.rda"))
home_model_nb <- readRDS(paste0(output,"home_model_nb.rda"))
home_model_knn <- readRDS(paste0(output,"home_model_knn.rda"))
home_model_svm <- readRDS(paste0(output,"home_model_svm.rda"))