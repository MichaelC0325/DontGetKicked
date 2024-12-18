# load libraries
suppressMessages(library(tidyverse))
suppressMessages(library(tidymodels))
suppressMessages(library(vroom))
suppressMessages(library(corrplot))
suppressMessages(library(xgboost))
suppressMessages(library(embed)) # for target encoding
suppressMessages(library(themis)) # for balancing

# predict and format function
predict_and_format <- function(workflow, newdata, filename){
  predictions <- predict(workflow, new_data = newdata, type = "prob")
  
  submission <- predictions %>% 
    mutate(RefId = idNumbers$RefId) %>% 
    rename("IsBadBuy" = ".pred_1") %>% 
    select(3,2)
  
  vroom_write(submission, filename, delim = ',')
}

# Read in Data
train <- vroom('training.csv')
test <- vroom('test.csv')
idNumbers <- vroom('test.csv')

# Correct missing values
train[train == "NULL"] <- NA
test[test == "NULL"] <- NA

# Change datatypes

# 
# train$WheelTypeID <- as.double(train$WheelTypeID)
# train$MMRCurrentAuctionAveragePrice <- as.double(train$MMRCurrentAuctionAveragePrice)
# train$MMRCurrentAuctionCleanPrice <- as.double(train$MMRCurrentAuctionCleanPrice)
# train$MMRCurrentRetailAveragePrice <- as.double(train$MMRCurrentRetailAveragePrice)
# train$MMRCurrentRetailCleanPrice <- as.double(train$MMRCurrentRetailCleanPrice)

# this is for the correlation plot
# numeric <- train %>%
#   select(IsBadBuy, VehYear, VehicleAge, WheelTypeID, VehOdo, MMRAcquisitionAuctionAveragePrice,
#          MMRAcquisitionAuctionCleanPrice, MMRAcquisitionRetailAveragePrice, MMRAcquisitonRetailCleanPrice,
#          MMRCurrentAuctionAveragePrice, MMRCurrentAuctionCleanPrice, MMRCurrentRetailAveragePrice, MMRCurrentRetailCleanPrice,
#          BYRNO, VNZIP1, VehBCost, IsOnlineSale, WarrantyCost) %>% 
#   na.omit()
# ## PLOT COR
# mat <- cor(numeric)
# corrplot(mat)

# Drop columns
# IDs <- c('RefId', 'WheelTypeID', 'BYRNO')
# categories <- c('PurchDate', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'VNZIP1', 'VNST')
# high_corr <- c('MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailCleanPrice',
#                'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice', 'VehYear')
# 
# drop_cols <- c(IDs, categories, high_corr)
# # remove cols from train and test
# train <- train[, !(names(train) %in% drop_cols)]
# test <- test[, !(names(test) %in% drop_cols)]

# # MISSING VALUES
# columns_with_missing_values <- colnames(train)[apply(is.na(train), 2, any)]
# columns_with_missing_values

# recipe for modeling
my_recipe <-  recipe(IsBadBuy ~ ., train) %>%
  step_mutate(WheelTypeID = as.double(WheelTypeID), 
              MMRCurrentAuctionAveragePrice = as.double(MMRCurrentAuctionAveragePrice),
              MMRCurrentAuctionCleanPrice = as.double(MMRCurrentAuctionCleanPrice),
              MMRCurrentRetailAveragePrice = as.double(MMRCurrentRetailAveragePrice),
              MMRCurrentRetailCleanPrice = as.double(MMRCurrentRetailCleanPrice)
              ) %>%
  step_rm(RefId, WheelTypeID, BYRNO, PurchDate, Make, Model, Trim, SubModel, Color, VNZIP1, VNST, 
          MMRCurrentAuctionCleanPrice, MMRCurrentRetailCleanPrice, MMRAcquisitionAuctionCleanPrice, MMRAcquisitonRetailCleanPrice, VehYear
          ) %>%
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
  step_novel(all_nominal_predictors(), -all_outcomes()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>% # target encoding
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold= 0.7) %>%
  step_zv() %>%
  step_normalize(all_numeric_predictors())

#train$IsBadBuy <- as.factor(train$IsBadBuy)

prepped <- prep(my_recipe)
baked <- bake(prepped, new_data = train)

# -------------------------------------------------------------------------


xgboost_model <- boost_tree(trees = 100,
                            tree_depth = tune(), 
                            min_n = tune(),
                            loss_reduction = tune(),
                            mtry = tune(),
                            learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

## SET UP WORKFLOW
xgboost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(xgboost_model)

tuneGrid <- grid_regular(
  tree_depth(),
  min_n(),
  learn_rate(),
  loss_reduction(),
  mtry(range = c(1, ncol(train))),
  levels = 3
)

folds <- vfold_cv(train, v = 5, repeats = 1)

## RUN CV
cl <- makePSOCKcluster(6) # num_cores to use
registerDoParallel(cl)
CV_results <- xgboost_wf %>%
  tune_grid(resamples = folds,
            grid = tuneGrid,
            metrics = metric_set(roc_auc))
stopCluster(cl) #need to run this at the end or might face errors

## FIND BEST TUNING PARAMETERS
xgboost_best_tune <- select_best(CV_results, metric = "roc_auc")

final_xgboost_wf <- xgboost_wf %>%
  finalize_workflow(xgboost_best_tune) %>%
  fit(data = train)

## PREDICTIONS
predict_and_format(final_xgboost_wf, test, "xgb_submission.csv")
