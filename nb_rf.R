# load libraries
suppressMessages(library(tidyverse))
suppressMessages(library(tidymodels))
suppressMessages(library(vroom))
suppressMessages(library(corrplot))
suppressMessages(library(discrim)) # naive bayes
suppressMessages(library(embed)) # for target encoding
suppressMessages(library(themis)) # for balancing
library(stacks)
library(doParallel)

# read in data
train <- suppressMessages(vroom("training.csv"))
train[train == "NULL"] <- NA

test <- suppressMessages(vroom("test.zip"))
test[test == "NULL"] <- NA

# just to retain original ID numbers
test2 <- suppressMessages(vroom("test.zip"))

# predict and format function
predict_and_format <- function(workflow, newdata, filename){
  predictions <- predict(workflow, new_data = newdata, type = "prob")
  
  submission <- predictions %>% 
    mutate(RefId = test2$RefId) %>% 
    rename("IsBadBuy" = ".pred_1") %>% 
    select(3,2)
  
  vroom_write(submission, filename, delim = ',')
}

# convert characters to doubles
train$WheelTypeID <- as.double(train$WheelTypeID)
train$MMRCurrentAuctionAveragePrice <- as.double(train$MMRCurrentAuctionAveragePrice)
train$MMRCurrentAuctionCleanPrice <- as.double(train$MMRCurrentAuctionCleanPrice)
train$MMRCurrentRetailAveragePrice <- as.double(train$MMRCurrentRetailAveragePrice)
train$MMRCurrentRetailCleanPrice <- as.double(train$MMRCurrentRetailCleanPrice)

# select all numeric columns for correlation plot
numeric <- train %>%
  select(IsBadBuy, VehYear, VehicleAge, WheelTypeID, VehOdo, MMRAcquisitionAuctionAveragePrice,
         MMRAcquisitionAuctionCleanPrice, MMRAcquisitionRetailAveragePrice, MMRAcquisitonRetailCleanPrice,
         MMRCurrentAuctionAveragePrice, MMRCurrentAuctionCleanPrice, MMRCurrentRetailAveragePrice, MMRCurrentRetailCleanPrice,
         BYRNO, VNZIP1, VehBCost, IsOnlineSale, WarrantyCost) %>% 
  na.omit()

mat <- cor(numeric)
corrplot(mat)

# unnecesary cols
IDs <- c('RefId', 'WheelTypeID', 'BYRNO')
categories <- c('PurchDate', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'VNZIP1', 'VNST')
high_corr <- c('MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailCleanPrice',
               'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitonRetailCleanPrice', 'VehYear')

drop_cols <- c(IDs, categories, high_corr)

# remove cols from train and test
train <- train[, !(names(train) %in% drop_cols)]
test <- test[, !(names(test) %in% drop_cols)]

columns_with_missing_values <- colnames(train)[apply(is.na(train), 2, any)]
columns_with_missing_values

# replace missing numeric values with the median
train$MMRAcquisitionAuctionAveragePrice[is.na(train$MMRAcquisitionAuctionAveragePrice)] <- median(train$MMRAcquisitionAuctionAveragePrice,na.rm = TRUE)
train$MMRAcquisitionRetailAveragePrice[is.na(train$MMRAcquisitionRetailAveragePrice)] <- median(train$MMRAcquisitionRetailAveragePrice,na.rm = TRUE)
train$MMRCurrentAuctionAveragePrice[is.na(train$MMRCurrentAuctionAveragePrice)] <- median(train$MMRCurrentAuctionAveragePrice,na.rm = TRUE)
train$MMRCurrentRetailAveragePrice[is.na(train$MMRCurrentRetailAveragePrice)] <- median(train$MMRCurrentRetailAveragePrice,na.rm = TRUE)

test$MMRAcquisitionAuctionAveragePrice[is.na(test$MMRAcquisitionAuctionAveragePrice)] <- median(test$MMRAcquisitionAuctionAveragePrice,na.rm = TRUE)
test$MMRAcquisitionRetailAveragePrice[is.na(test$MMRAcquisitionRetailAveragePrice)] <- median(test$MMRAcquisitionRetailAveragePrice,na.rm = TRUE)
test$MMRCurrentAuctionAveragePrice[is.na(test$MMRCurrentAuctionAveragePrice)] <- median(test$MMRCurrentAuctionAveragePrice,na.rm = TRUE)
test$MMRCurrentRetailAveragePrice[is.na(test$MMRCurrentRetailAveragePrice)] <- median(test$MMRCurrentRetailAveragePrice,na.rm = TRUE)

# replace missing character values with unknown category
missing <- c('Transmission', 'WheelType', 'Nationality', 'Size',
             'TopThreeAmericanName', 'PRIMEUNIT', 'AUCGUART')

for (i in missing) {
  train[[i]] <- ifelse(is.na(train[[i]]), 'Unknown', train[[i]])
  test[[i]] <- ifelse(is.na(test[[i]]), 'Unknown', test[[i]])
}

columns_with_missing_values <- colnames(train)[apply(is.na(train), 2, any)]
columns_with_missing_values # none!

# recipe for modeling
my_recipe <-  recipe(IsBadBuy ~ ., train) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%  # target encoding
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE)

train$IsBadBuy <- as.factor(train$IsBadBuy)

# naive bayes -------------------------------------------------------------
nb_mod <- naive_Bayes(Laplace = tune(),
                      smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_mod)

# cross validation
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

#nb_folds <- vfold_cv(train, v = 5, repeats = 1)

#CV_results <- nb_wf %>%
#  tune_grid(resamples = nb_folds,
#            grid = nb_tuning_grid,
#            metrics = metric_set(accuracy))

#nb_bestTune <- CV_results %>%
#  select_best("accuracy")

#final_nb_wf <- nb_wf %>%
#  finalize_workflow(nb_bestTune) %>%
#  fit(data = train)

#predict_and_format(final_nb_wf, test, "naive_bayes_predictions.csv")
# score: 0.20395

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

#forest_folds <- vfold_cv(train, v = 5, repeats = 1)

#CV_results <- rand_forest_wf %>%
#  tune_grid(resamples = forest_folds,
#            grid = rand_forest_tuning_grid,
#            metrics = metric_set(accuracy))

#forest_bestTune <- CV_results %>%
#  select_best("accuracy")

#final_forest_wf <- rand_forest_wf %>%
#  finalize_workflow(forest_bestTune) %>%
#  fit(data = train)

#predict_and_format(final_forest_wf, test, "random_forest_predictions.csv")

# stacking
folds <- vfold_cv(train, v = 5, repeats=1)
untunedModel <- control_stack_grid()


cl <- makePSOCKcluster(8) # num_cores to use
registerDoParallel(cl)
randforest_models <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=rand_forest_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)
stopCluster(cl) #need to run this at the end or might face errors



nb_models <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=nb_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

my_stack <- stacks() %>%
  add_candidates(nb_models) %>%
  add_candidates(randforest_models)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

predictions <- stack_mod %>%
  predict(new_data = test,
          type = "prob")

submission <- predictions %>%
  mutate(RefId = test2$RefId) %>% 
  rename("IsBadBuy" = ".pred_1") %>% 
  select(3,2)

vroom_write(x = submission, file = "stacked_predictions.csv", delim=",")