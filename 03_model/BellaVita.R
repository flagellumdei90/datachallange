# Load packages

# Load home win models
home_model_rf <- readRDS(paste0(output,"home_model_rf.rda"))
home_model_glm <- readRDS(paste0(output,"home_model_glm.rda"))
home_model_gbm <- readRDS(paste0(output,"home_model_gbm.rda"))
home_model_xgb <- readRDS(paste0(output,"home_model_xgb.rda"))
home_model_nb <- readRDS(paste0(output,"home_model_nb.rda"))
home_model_knn <- readRDS(paste0(output,"home_model_knn.rda"))
home_model_svm <- readRDS(paste0(output,"home_model_svm.rda"))

link_home_glm <- "https://github.com/flagellumdei90/datachallange/blob/main/04_output/home_model_glm.rda?raw=true"

home_model_glm <- readRDS(url(link_home_glm))

# Create modell_input tables
  # Load data
  data <- as.data.table(read_excel(paste0(data_in, "competition_table.xlsx")))
  
  # Create modell_input tables
  data_input_home <- data[,-c("home_win_flag", "draw_flag", "away_win_flag")]
  
  # Feature engineering

# Predict
predicted_proba_holdout <- predict(home_model_glm, 
                                   newdata = data_input_home, 
                                   type = "prob")[["yes"]]


# Output - 3 táblát állítson elő
# match_id
# home_win_p
# draw_p
# away_win_p