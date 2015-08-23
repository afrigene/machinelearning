library(caret)
library(data.table)

predictors <- c( "user_name", "num_window", 
                 "roll_belt", "pitch_belt","yaw_belt","total_accel_belt",
                 "gyros_belt_x","gyros_belt_y","gyros_belt_z",
                 "accel_belt_x","accel_belt_y","accel_belt_z",
                 "magnet_belt_x","magnet_belt_y","magnet_belt_z",
                 "roll_arm",	"pitch_arm", "yaw_arm",	"total_accel_arm",
                 "gyros_arm_x", "gyros_arm_y",	"gyros_arm_z",	
                 "accel_arm_x",	"accel_arm_y",	"accel_arm_z",	
                 "magnet_arm_x",	"magnet_arm_y",	"magnet_arm_z",
                 "roll_dumbbell", "pitch_dumbbell",	"yaw_dumbbell", "total_accel_dumbbell",
                 "gyros_dumbbell_x","gyros_dumbbell_y",	"gyros_dumbbell_z",	
                 "accel_dumbbell_x",	"accel_dumbbell_y",	"accel_dumbbell_z",	
                 "magnet_dumbbell_x",	"magnet_dumbbell_y","magnet_dumbbell_z",
                 "roll_forearm", "pitch_forearm",	"yaw_forearm", "total_accel_forearm",
                 "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z",
                 "accel_forearm_x", "accel_forearm_y", "accel_forearm_z",	
                 "magnet_forearm_x",	"magnet_forearm_y",	"magnet_forearm_z"
                )

# Pre-process the data by centering, scaling and imputing missing values
preprocess <- function(data, train=TRUE)
{
  data <- data.table(data)
  
  if (train)
  {
    data <- data[,lapply(.SD, mean), by = list(user_name, num_window, target)]
    vars <- data[,colnames(data)[4:ncol(data)], with=FALSE]
    
  } else {
    
    data <- data[,lapply(.SD, mean), by = list(user_name, num_window)]
    vars <- data[,colnames(data)[3:ncol(data)], with=FALSE]
  }
  
  # center and scale variables
  preProcValues <- preProcess(vars, method = c("center", "scale", "medianImpute"))
  result <- predict(preProcValues, vars)
  
  if (train)
  {
    result <- data.frame(data$target, result)
    
    colnames(result)[1] <- "target"  
  }
  result
}


# Read training data
data <- read.csv(file="pml-training.csv", header=TRUE, )

# Partition data in TRAIN and TEST sets
inTrain <- createDataPartition(y=data$classe, p=0.7, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]


# Generate model on TRAIN data
#########################################
train_data <- preprocess(data.frame(training[,predictors], target=training$classe), train=TRUE)
modelFit <- train(target ~ ., data=train_data, method="rf")

# calculate in sample error
predictions <- predict(modelFit, newdata = train_data)
confusionMatrix(predictions, train_data$target)


# Evaluate model on TEST data
#########################################
test_data <- preprocess(data.frame(testing[,predictors], target=testing$classe), train=TRUE)

# calculate out of sample error
predictions <- predict(modelFit, newdata = test_data)
confusionMatrix(predictions, test_data$target)


# Apply model to HOLDOUT data
########################################
holdout <- read.csv(file="pml-testing.csv", header=TRUE, )
holdout_data <- preprocess(holdout[,predictors], train=FALSE)
answers <- predict(modelFit, newdata = holdout_data)


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
