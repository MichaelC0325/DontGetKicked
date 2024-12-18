
# Libraries and Loading ---------------------------------------------------
# score cutoff A: >= 0.237

library(stacks)
library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(doParallel)
library(themis)
library(bonsai)
library(lightgbm)
library(ranger)
library(discrim)
library(kernlab)
library(naivebayes)
library(kknn)



train <- vroom("training.csv", na = c("", "NA", "NULL", "NOT AVAIL"))
test <- vroom("test.csv", na = c("", "NA", "NULL", "NOT AVAIL"))

# Recipe -------------------------------------------------------------------------

my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  # Convert characters to numeric for specified columns
  step_mutate(
    WheelTypeID = as.numeric(WheelTypeID),
    MMRCurrentAuctionAveragePrice = as.numeric(MMRCurrentAuctionAveragePrice),
    MMRCurrentAuctionCleanPrice = as.numeric(MMRCurrentAuctionCleanPrice),
    MMRCurrentRetailAveragePrice = as.numeric(MMRCurrentRetailAveragePrice),
    MMRCurrentRetailCleanPrice = as.numeric(MMRCurrentRetailCleanPrice)
  ) %>%
  # Remove unnecessary columns
  step_rm(c('RefId', 'WheelTypeID', 'BYRNO', 
            'PurchDate', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 
            'VNZIP1', 'VNST', 
            'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailCleanPrice',
            'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice', 
            'VehYear')) %>%
  # Impute missing numeric values with the median
  step_impute_median(all_numeric_predictors()) %>%
  # Impute missing character values with "Unknown"
  step_mutate(
    Transmission = ifelse(is.na(Transmission), 'Unknown', Transmission),
    WheelType = ifelse(is.na(WheelType), 'Unknown', WheelType),
    Nationality = ifelse(is.na(Nationality), 'Unknown', Nationality),
    Size = ifelse(is.na(Size), 'Unknown', Size),
    TopThreeAmericanName = ifelse(is.na(TopThreeAmericanName), 'Unknown', TopThreeAmericanName),
    PRIMEUNIT = ifelse(is.na(PRIMEUNIT), 'Unknown', PRIMEUNIT),
    AUCGUART = ifelse(is.na(AUCGUART), 'Unknown', AUCGUART)
  ) %>%
  # Apply target encoding to categorical predictors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>% 
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE)




# apply the recipe to your data
prep <- prep(my_recipe)
#baked <- bake(prep, new_data=test)
baked <- bake(prep, new_data = train)

# Models ------------------------------------------------------------------

## Random Forest
rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees = 100) %>% 
  set_engine("ranger") %>%
  set_mode("classification")

rand_forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(train)-1))),
                                        min_n(),
                                        levels = 5)


## Naive Bayes
nb_mod <- naive_Bayes(smoothness = tune(), Laplace = tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_mod)

nb_tuning_grid <- grid_regular(smoothness(),
                               Laplace(),
                               levels = 5)

## XGBoost
xgboost_model <- boost_tree(trees = 50,
                            tree_depth = tune(), 
                            min_n = tune(),
                            loss_reduction = tune(),
                            mtry = tune(),
                            learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgboost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(xgboost_model)

xgboost_tuning_grid <- grid_regular(
  tree_depth(range = c(3, 6)),
  min_n(range = c(10, 30)),
  learn_rate(range = c(0.1, 0.3)),
  loss_reduction(range = c(1, 10)),
  mtry(range = c(1, 5)),
  levels = 3
)


# Stacking ----------------------------------------------------------------

## models
folds <- vfold_cv(train, v = 3, repeats=1)
untunedModel <- control_stack_grid()

cl <- makePSOCKcluster(8) # num_cores to use
registerDoParallel(cl)
randforest_models <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=rand_forest_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)
stopCluster(cl) #need to run this at the end or might face errors

#testing to see if this helps
xgboost_models <- xgboost_wf %>% 
  tune_grid(resamples = folds,
            grid = xgboost_tuning_grid,
            metrics = metric_set(roc_auc),
            control = untunedModel)


nb_models <- nb_wf %>% 
  tune_grid(resamples=folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

## stack
my_stack <- stacks() %>%
  add_candidates(xgboost_models) %>% 
  add_candidates(randforest_models) %>% 
  add_candidates(nb_models)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

# Predictions -------------------------------------------------------------------------

# probability predictions
predictions <- stack_mod %>%
  predict(new_data = test,
          type = "prob")

# writing out submission
submission <- predictions %>%
  mutate(RefId = test$RefId) %>% 
  rename("IsBadBuy" = ".pred_1") %>% 
  select(3,2)

vroom_write(x = submission, file = "stacked_predictions6.csv", delim=",")

