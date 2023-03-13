# empty memory
rm(list=ls())

# Load packages / install if not installed
# install.packages("data.table")
# install.packages("readxl")

library("data.table") # to use data in an efficient way
library("readxl") # to read xlsx data
# library("caret") # used for training
# library("dplyr") # to append commands easier
# library("ROCR") # to evaluate model performance

# Set the working directory
dir <-  "/Users/attilagulyas/Documents/GitHub/DataChallenge2023/"
data_in <- paste0(dir,"01_raw/") # input file folder
data_out <- paste0(dir,"04_output/") # prediction output folder

# Load data
data <- as.data.table(read_excel(paste0(data_in, "competition_table.xlsx")))

# Load  models github repo (public access)
link_home_model<- "https://github.com/flagellumdei90/datachallange/blob/main/04_output/home_model.rda?raw=true"
home_model <- readRDS(url(link_home_model))

link_away_model<- "https://github.com/flagellumdei90/datachallange/blob/main/04_output/away_model.rda?raw=true"
away_model <- readRDS(url(link_away_model))

link_draw_model<- "https://github.com/flagellumdei90/datachallange/blob/main/04_output/draw_model.rda?raw=true"
draw_model <- readRDS(url(link_draw_model))

# Create modell_input table (drop outcome variables)
model_input <- data[,-c("home_win_flag", "draw_flag", "away_win_flag")]
# Get mach_id-s
match_id_list <- model_input$match_id

# Predict
predicted_home_win <- predict(home_model, 
                                   newdata = model_input, 
                                   type = "prob")[["yes"]]

predicted_away_win <- predict(away_model, 
                              newdata = model_input, 
                              type = "prob")[["yes"]]

predicted_draw_win <- predict(draw_model, 
                              newdata = model_input, 
                              type = "prob")[["yes"]]

# Create Prediction table output
prediction_table <- cbind(match_id = match_id_list,
                          home_win_p1 = predicted_home_win, 
                          away_win_p1 = predicted_away_win, 
                          draw_p1 = predicted_draw_win)

# Save csv
write.csv(prediction_table, file = paste0(data_out, "bellavita.csv"), sep = ";", fileEncoding = "UTF8")
