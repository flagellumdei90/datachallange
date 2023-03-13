# Load packages
library("tidyverse") # imap function for roc curve
library("data.table") # to use data in an efficient way
library("readxl") # to read xlsx data
library("caret") # to split data into train and holdput sets
library("dplyr") # to append commands easier
library("ROCR") # to evaluate model performance


# Load  models
link_home_model<- "https://github.com/flagellumdei90/datachallange/blob/main/04_output/home_model_glm.rda?raw=true"
home_model <- readRDS(url(link_home_model))

# Create modell_input tables
  # Load data
  data <- as.data.table(read_excel(paste0(data_in, "competition_table.xlsx")))
  
  # Create modell_input tables
  model_input <- data[,-c("home_win_flag", "draw_flag", "away_win_flag")]
  

# Predict
predicted_home_win <- predict(home_model, 
                                   newdata = model_input, 
                                   type = "prob")[["yes"]]



# Output - 3 táblát állítson elő
# match_id
# home_win_p
# draw_p
# away_win_p