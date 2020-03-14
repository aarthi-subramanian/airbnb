### Load dependencies:
#install.packages('dummies')
#install.packages('ggRandomForests')
#install.packages('leaps')
#install.packages('glmnet')

library(readr)
library(data.table)
library(stringr)
library(Rcpp)
library(ggplot2)
library(tidyr)
library(dplyr)
library(tidyverse)
library(devtools)
library(dummies)
library(leaps)
library(caret)
library(glmnet)
library(gbm)
library(randomForest)
library(ggRandomForests)

### Read data:
analysisDataPath = file.path("~", "Documents/School/Courses/APAN 5200 - Frameworks/Kaggle/all", "analysisData.csv");
analysisData = read.csv(analysisDataPath);
scoringDataPath = file.path("~", "Documents/School/Courses/APAN 5200 - Frameworks/Kaggle/all", "scoringData.csv");
scoringData = read.csv(scoringDataPath);

### Explore a bit:
head(analysisData, n=1) # 6 rows by dfault
str(analysisData)
glimpse(analysisData) # like str but enhanced format (dplyr)
summary(analysisData)

boxplot(analysisData$price) # no outliers, no need to throw out single rows upfront
boxplot(log(analysisData$price)) # no outliers, no need to throw out single rows upfront

# Explore raw data: 1) understand structure data, 2) looking at data, 3) visualising data:
# 1) Understand structure of data:
class(analysisData) # make sure it's "data.frame"
dim(analysisData) # rows, cols: 29142, 96
colnames(analysisData)

# 2), 3) looking + visualising data:
hist(analysisData$price) # visualise a single var
hist(log(analysisData$price)) # visualise a single var
plot(analysisData$price, analysisData$zipcode) # visualise relation between 2 vars


### Data cleaning:
# rm the cols where ALL values are NA:
analysisData = analysisData[, colSums(is.na(analysisData)) != nrow(analysisData)]
colnames(analysisData)
scoringData = scoringData[, colSums(is.na(scoringData)) != nrow(scoringData)]

# rm columns with more than 40% NA:
analysisData = analysisData[, colMeans(is.na(analysisData)) <= .4]
colnames(analysisData)
scoringData = scoringData[, colMeans(is.na(scoringData)) <= .4] # 88 vars

# rm columns where single value for all rows:
analysisData = Filter(function(x)(length(unique(x))>1), analysisData)
colnames(analysisData)
scoringData = Filter(function(x)(length(unique(x))>1), scoringData) # 83 vars

## Specific cases for cleaning:
# row with "country= Uruguay" is wrong data so leave this row in but remove the 2 country cols because  
# single value for all rows otherwise:
colnames(scoringData)
analysisData = subset(analysisData, select = -c(country_code,country))
scoringData = subset(scoringData, select = -c(country_code,country))
colnames(scoringData) #can maybe remove this bcos country cols already removed before this?

# rm "url","id" cols: this value is randomly assigned-cannot control for this when predicting price next time:
analysisData = subset(analysisData, select = -c(listing_url, picture_url, host_url, host_thumbnail_url, host_picture_url, host_id))
scoringData = subset(scoringData, select = -c(listing_url, picture_url, host_url, host_thumbnail_url, host_picture_url, host_id ))
colnames(analysisData)

# drop a load (based on list in excel- bucketing as per categories): 76->44 vars
analysisData = dplyr::select(analysisData, -c(availability_30,availability_365,availability_60,availability_90,
                                calculated_host_listings_count,calendar_last_scraped,calendar_updated,
                                host_about,host_location,host_name,host_neighbourhood, last_scraped,
                                host_verifications,jurisdiction_names,
                                market,neighbourhood,
                                neighbourhood_cleansed,require_guest_phone_verification,
                                require_guest_profile_picture,smart_location,street,
                                zipcode))
scoringData = dplyr::select(scoringData, -c(availability_30,availability_365,availability_60,availability_90,
                                calculated_host_listings_count,calendar_last_scraped,calendar_updated,
                                host_about,host_location,host_name,host_neighbourhood, last_scraped,
                                host_verifications,jurisdiction_names,
                                market,neighbourhood,
                                neighbourhood_cleansed,require_guest_phone_verification,
                                require_guest_profile_picture,smart_location,street,
                                zipcode))
dim(analysisData) # 29142 x 53
dim(scoringData) # 7286 x 53
# Note: ^ leave in text colsfor FE later: name, summary, space, description, neighborhood_overview, notes, access, interaction


### Feature Engineering: create dummy variables (based on list in excel), convert dates, build new cols:
# create dummy variables:
analysisData_cln <- dummy.data.frame(analysisData, sep="_", names = c("host_response_time","host_is_superhost","host_has_profile_pic",
                                                        "neighbourhood_group_cleansed","room_type","instant_bookable",
                                                        "cancellation_policy","bed_type","host_identity_verified",
                                                        "is_business_travel_ready","is_location_exact")) # 44->68 vars

scoringData_cln <- dummy.data.frame(scoringData, sep="_", names = c("host_response_time","host_is_superhost","host_has_profile_pic",
                                                             "neighbourhood_group_cleansed","room_type","instant_bookable",
                                                             "cancellation_policy","bed_type","host_identity_verified",
                                                             "is_business_travel_ready","is_location_exact")) # 44->68 vars
colnames(analysisData_cln)

# drop duplicate col of boolean factors:
#host_is_superhost, host_has_profile_pic, instant_bookable, host_identity_verified, is_business_travel_ready, is_location_exact.
analysisData_cln = dplyr::select(analysisData_cln, -c(host_is_superhost_f,host_has_profile_pic_f,instant_bookable_f,
                                                      host_identity_verified_f,is_business_travel_ready_f,is_location_exact_f))
scoringData_cln = dplyr::select(scoringData_cln, -c(host_is_superhost_f,host_has_profile_pic_f,instant_bookable_f,
                                                      host_identity_verified_f,is_business_travel_ready_f,is_location_exact_f))
dim(analysisData_cln) # 29142 x 71
dim(scoringData_cln) # 7286 x 71

# convert date cols to "days since" cols. Interesting date cols: host_since, last_review, first_review:
# (dont worry we will cut down later/once we see if there's anything useful here.)
today = Sys.Date(); today # '2018-11-25' format

analysisData_cln$days_since_host <- as.Date(as.character(today), format="%Y-%m-%d")-
  as.Date(as.character(analysisData_cln$host_since), format="%Y-%m-%d")
head(analysisData_cln$days_since_host)

analysisData_cln$days_since_last_review <- as.Date(as.character(today), format="%Y-%m-%d")-
  as.Date(as.character(analysisData_cln$last_review), format="%Y-%m-%d")
head(analysisData_cln$days_since_last_review)

analysisData_cln$days_since_first_review <- as.Date(as.character(today), format="%Y-%m-%d")-
  as.Date(as.character(analysisData_cln$first_review), format="%Y-%m-%d")
head(analysisData_cln$days_since_first_review)

analysisData_cln$days_since_host = as.numeric(analysisData_cln$days_since_host)
analysisData_cln$days_since_last_review = as.numeric(analysisData_cln$days_since_last_review)
analysisData_cln$days_since_first_review = as.numeric(analysisData_cln$days_since_first_review)
class(analysisData_cln$days_since_host)
class(analysisData_cln$days_since_last_review)
class(analysisData_cln$days_since_first_review)

analysisData_cln = dplyr::select(analysisData_cln, -c(host_since,last_review, first_review))

#test:
scoringData_cln$days_since_host <- as.Date(as.character(today), format="%Y-%m-%d")-
  as.Date(as.character(scoringData_cln$host_since), format="%Y-%m-%d")
head(scoringData_cln$days_since_host)

scoringData_cln$days_since_last_review <- as.Date(as.character(today), format="%Y-%m-%d")-
  as.Date(as.character(scoringData_cln$last_review), format="%Y-%m-%d")
head(scoringData_cln$days_since_last_review)

scoringData_cln$days_since_first_review <- as.Date(as.character(today), format="%Y-%m-%d")-
  as.Date(as.character(scoringData_cln$first_review), format="%Y-%m-%d")
head(scoringData_cln$days_since_first_review)

scoringData_cln$days_since_host = as.numeric(scoringData_cln$days_since_host)
scoringData_cln$days_since_last_review = as.numeric(scoringData_cln$days_since_last_review)
scoringData_cln$days_since_first_review = as.numeric(scoringData_cln$days_since_first_review)
class(scoringData_cln$days_since_host)
class(scoringData_cln$days_since_last_review)
class(scoringData_cln$days_since_first_review)

scoringData_cln = dplyr::select(scoringData_cln, -c(host_since,last_review, first_review))

# build some new columns:
# 1. Convert host_response_rate from % to decimal:
analysisData_cln$host_response_rate_numeric = as.numeric(sub("%", "",analysisData_cln$host_response_rate,fixed=TRUE))/100 #NAs introduced
scoringData_cln$host_response_rate_numeric = as.numeric(sub("%", "",scoringData_cln$host_response_rate,fixed=TRUE))/100 #NAs introduced

# handle NAs: treat as 0 for now: (bcos coincides with no data col and NOT superhost) **can change later if find reason*
analysisData_cln[c("host_response_rate_numeric")][is.na(analysisData_cln[c("host_response_rate_numeric")])] = 0
scoringData_cln[c("host_response_rate_numeric")][is.na(scoringData_cln[c("host_response_rate_numeric")])] = 0

analysisData_cln = dplyr::select(analysisData_cln, -c(host_response_rate)) #rm the og column
scoringData_cln = dplyr::select(scoringData_cln, -c(host_response_rate)) #rm the og column

# 2. Has vs. has no house rules:
analysisData_cln$house_rules[analysisData_cln$house_rules==""] <- NA
analysisData_cln$has_house_rules[!(is.na(analysisData_cln$house_rules))] = 1
analysisData_cln$has_house_rules[is.na(analysisData_cln$house_rules)] = 0

analysisData_cln = dplyr::select(analysisData_cln, -c(house_rules)) #rm the og house_rules column

#test:
scoringData_cln$house_rules[scoringData_cln$house_rules==""] <- NA 
scoringData_cln$has_house_rules[!(is.na(scoringData_cln$house_rules))] = 1
scoringData_cln$has_house_rules[is.na(scoringData_cln$house_rules)] = 0

scoringData_cln = dplyr::select(scoringData_cln, -c(house_rules)) #rm the og house_rules column

# 3. Has transit details:
analysisData_cln$transit[analysisData_cln$transit == ""] <- NA
analysisData_cln$has_transit_details[!(is.na(analysisData_cln$transit))] = 1
analysisData_cln$has_transit_details[is.na(analysisData_cln$transit)] = 0
analysisData_cln = dplyr::select(analysisData_cln, -c(transit)) #rm the og col

#test:
scoringData_cln$transit[scoringData_cln$transit == ""] <- NA
scoringData_cln$has_transit_details[!(is.na(scoringData_cln$transit))] = 1
scoringData_cln$has_transit_details[is.na(scoringData_cln$transit)] = 0
scoringData_cln = dplyr::select(scoringData_cln, -c(transit)) #rm the og col

# 4. Number of amenities (looks like standard list of ~20):
# analysisData_cln$number_of_amenities = length(as.list(strsplit(as.character(analysisData_cln$amenities), ",")[[1]])) # not working
analysisData_cln$number_of_amenities = str_count(analysisData_cln$amenities, ",") + 1 # not the best way. fine since no trailing comma
scoringData_cln$number_of_amenities = str_count(scoringData_cln$amenities, ",") + 1 # not the best way. fine since no trailing comma

analysisData_cln = dplyr::select(analysisData_cln, -c(amenities)) #rm the og col
scoringData_cln = dplyr::select(scoringData_cln, -c(amenities)) #rm the og col

# 5. Length of text in description cols: name, summary, space, description, neighborhood_overview, notes, access, interaction:
# (dont worry we will cut down later/once we see if there's anything useful here.)
analysisData_cln$description_length_name = nchar(as.character(analysisData_cln$name))
scoringData_cln$description_length_name = nchar(as.character(scoringData_cln$name))
analysisData_cln$description_length_summary = nchar(as.character(analysisData_cln$summary))
scoringData_cln$description_length_summary = nchar(as.character(scoringData_cln$summary))
analysisData_cln$description_length_space = nchar(as.character(analysisData_cln$space))
scoringData_cln$description_length_space = nchar(as.character(scoringData_cln$space))
analysisData_cln$description_length_description = nchar(as.character(analysisData_cln$description))
scoringData_cln$description_length_description = nchar(as.character(scoringData_cln$description))
analysisData_cln$description_length_neighborhood_overview = nchar(as.character(analysisData_cln$neighborhood_overview))
scoringData_cln$description_length_neighborhood_overview = nchar(as.character(scoringData_cln$neighborhood_overview))
analysisData_cln$description_length_notes = nchar(as.character(analysisData_cln$notes))
scoringData_cln$description_length_notes = nchar(as.character(scoringData_cln$notes))
analysisData_cln$description_length_access = nchar(as.character(analysisData_cln$access))
scoringData_cln$description_length_access = nchar(as.character(scoringData_cln$access))
analysisData_cln$description_length_interaction = nchar(as.character(analysisData_cln$interaction))
scoringData_cln$description_length_interaction = nchar(as.character(scoringData_cln$interaction))

# see if any of the above 8 fields help:
subsetDescriptionlength = analysisData_cln[,c('price','description_length_name','description_length_summary','description_length_space',
                                              'description_length_description','description_length_neighborhood_overview',
                                              'description_length_notes','description_length_access','description_length_interaction')]
str(subsetDescriptionlength)
round(cor(subsetDescriptionlength),2)*100

# based on above results, just leave in description_length_space as has a decent correlation with price:
analysisData_cln = dplyr::select(analysisData_cln, -c(description_length_name, description_length_summary,
                                                      description_length_description,description_length_neighborhood_overview,
                                                      description_length_notes,description_length_access,description_length_interaction))
scoringData_cln = dplyr::select(scoringData_cln, -c(description_length_name, description_length_summary,
                                                      description_length_description,description_length_neighborhood_overview,
                                                      description_length_notes,description_length_access,description_length_interaction))

#6. Number of exclamations in description cols: name, summary, space, description, neighborhood_overview, notes, access, interaction:
# (dont worry we will cut down later/once we see if there's anything useful here.)
analysisData_cln$exclamations_name = str_count(analysisData_cln$name, "!")
scoringData_cln$exclamations_name = str_count(scoringData_cln$name, "!")
analysisData_cln$exclamations_summary = str_count(analysisData_cln$summary, "!")
scoringData_cln$exclamations_summary = str_count(scoringData_cln$summary, "!")
analysisData_cln$exclamations_space = str_count(analysisData_cln$space, "!")
scoringData_cln$exclamations_space = str_count(scoringData_cln$space, "!")
analysisData_cln$exclamations_description = str_count(analysisData_cln$description, "!")
scoringData_cln$exclamations_description = str_count(scoringData_cln$description, "!")
analysisData_cln$exclamations_neighborhood_overview = str_count(analysisData_cln$neighborhood_overview, "!")
scoringData_cln$exclamations_neighborhood_overview = str_count(scoringData_cln$neighborhood_overview, "!")
analysisData_cln$exclamations_notes = str_count(analysisData_cln$notes, "!")
scoringData_cln$exclamations_notes = str_count(scoringData_cln$notes, "!")
analysisData_cln$exclamations_access = str_count(analysisData_cln$access, "!")
scoringData_cln$exclamations_access = str_count(scoringData_cln$access, "!")
analysisData_cln$exclamations_interaction = str_count(analysisData_cln$interaction, "!")
scoringData_cln$exclamations_interaction = str_count(scoringData_cln$interaction, "!")

analysisData_cln = dplyr::select(analysisData_cln, -c(name, summary, space, description, neighborhood_overview,
                                                      notes, access, interaction)) #rm the og cols?
scoringData_cln = dplyr::select(scoringData_cln, -c(name, summary, space, description, neighborhood_overview,
                                                    notes, access, interaction)) #rm the og cols?

# see if any of the above 8 fields help:
subsetExclamations = analysisData_cln[,c('price','exclamations_name','exclamations_summary','exclamations_space',
                                              'exclamations_description','exclamations_neighborhood_overview',
                                              'exclamations_notes','exclamations_access','exclamations_interaction')]
str(subsetExclamations)
round(cor(subsetExclamations),2)*100

# based on above results, remove ALL bcos none have a decent correlation with price:
analysisData_cln = dplyr::select(analysisData_cln, -c(exclamations_name, exclamations_summary, exclamations_space,
                                                      exclamations_description,exclamations_neighborhood_overview,
                                                      exclamations_notes,exclamations_access,exclamations_interaction))
scoringData_cln = dplyr::select(scoringData_cln, -c(exclamations_name, exclamations_summary, exclamations_space,
                                                      exclamations_description,exclamations_neighborhood_overview,
                                                      exclamations_notes,exclamations_access,exclamations_interaction))

# Convert all col names to lowercase, snake case:
names(analysisData_cln)
names(analysisData_cln) = tolower(gsub(" ","_",names(analysisData_cln)))
names(analysisData_cln) = tolower(gsub("/","_",names(analysisData_cln)))
names(analysisData_cln) = tolower(gsub("-","_",names(analysisData_cln)))
names(analysisData_cln) = make.names(names(analysisData_cln), unique=TRUE)
names(analysisData_cln)

names(scoringData_cln)
names(scoringData_cln) = tolower(gsub(" ","_",names(scoringData_cln)))
names(scoringData_cln) = tolower(gsub("/","_",names(scoringData_cln)))
names(scoringData_cln) = tolower(gsub("-","_",names(scoringData_cln)))
names(scoringData_cln) = make.names(names(scoringData_cln), unique=TRUE)
names(scoringData_cln)

# Lasso experiment results:
#Looked at results of Lasso, and the following vars seem to have no impact: host_response_xyz, bed_type_xyz. Drop:
colnames(analysisData_cln)
subsetHostResponse = analysisData_cln[,c('price', 'host_response_time_a_few_days_or_more','host_response_time_n_a',
                                         'host_response_time_within_a_day','host_response_time_within_a_few_hours',
                                         'host_response_time_within_an_hour')]
str(subsetHostResponse)
round(cor(subsetHostResponse),2)*100

subsetBedType = analysisData_cln[,c('price', 'bed_type_airbed','bed_type_couch',
                                         'bed_type_futon','bed_type_pull_out_sofa',
                                         'bed_type_real_bed')]
str(subsetBedType)
round(cor(subsetBedType),2)*100

# rm all and see:
analysisData_cln = dplyr::select(analysisData_cln, -c(host_response_time_a_few_days_or_more, host_response_time_n_a,
                                                      host_response_time_within_a_day, host_response_time_within_a_few_hours,
                                                      host_response_time_within_an_hour, bed_type_airbed, bed_type_couch,
                                                      bed_type_futon, bed_type_pull_out_sofa, bed_type_real_bed))
scoringData_cln = dplyr::select(scoringData_cln, -c(host_response_time_a_few_days_or_more, host_response_time_n_a,
                                                    host_response_time_within_a_day, host_response_time_within_a_few_hours,
                                                    host_response_time_within_an_hour, bed_type_airbed, bed_type_couch,
                                                    bed_type_futon, bed_type_pull_out_sofa, bed_type_real_bed))

# See if any of the 7 review_scores fields help:
colnames(analysisData_cln)
subsetReviewScores = analysisData_cln[,c('price','review_scores_rating', 'review_scores_accuracy','review_scores_cleanliness',
                                              'review_scores_checkin','review_scores_communication','review_scores_location',
                                              'review_scores_value')]
str(subsetReviewScores)
round(cor(subsetReviewScores),2)*100

# based on above results, just leave in review_scores_location, (review_scores_cleanliness, review_scores_rating) as has a decent correlation with price:
analysisData_cln = dplyr::select(analysisData_cln, -c(review_scores_accuracy, review_scores_checkin,
                                                      review_scores_communication, review_scores_value))
scoringData_cln = dplyr::select(scoringData_cln, -c(review_scores_accuracy, review_scores_checkin,
                                                    review_scores_communication, review_scores_value))


# Drop the remaining categorical vars for now:
analysisData_cln = dplyr::select(analysisData_cln, -c(city, state, property_type)) #rm this code once figure out
scoringData_cln = dplyr::select(scoringData_cln, -c(city, state, property_type)) #rm this code once figure out

# NAs: check where the NAs are **Really try to revisit and adjust later**:
as.data.frame(apply(analysisData_cln, 2, function(x) any(is.na(x))))
as.data.frame(apply(scoringData_cln, 2, function(x) any(is.na(x))))

# impute means where possible:
for(i in 1:ncol(analysisData_cln)){
  analysisData_cln[is.na(analysisData_cln[,i]), i] <- mean(analysisData_cln[,i], na.rm = TRUE)
}

for(i in 1:ncol(scoringData_cln)){
  scoringData_cln[is.na(scoringData_cln[,i]), i] <- mean(scoringData_cln[,i], na.rm = TRUE)
}

# check where the NAs are/ that the above succeeded:
as.data.frame(apply(analysisData_cln, 2, function(x) any(is.na(x))))
as.data.frame(apply(scoringData_cln, 2, function(x) any(is.na(x))))

dim(analysisData_cln) # 29142 x 47
dim(scoringData_cln) # 7286 x 47


### Feature Selection: Lasso
x = model.matrix(price~.,data=analysisData_cln)
y = analysisData_cln$price
lassoModel = glmnet(x,y, alpha=1) # Note default for alpha is 1 which corresponds to Lasso
lassoModel
summary(lassoModel)
lassoModel$beta
coef(lassoModel) # just run this to gauge what's goin on. use cv.lasso() instead.

cv.lasso = cv.glmnet(x,y,alpha=1) # 10-fold cross-validation
plot(cv.lasso)
summary(cv.lasso)
coef(cv.lasso)
# Select from coef(cv.lasso) the vars with coefficients aka useful to your model.


### Split data:
set.seed(1234)
split = createDataPartition(y=analysisData_cln$price,p = 0.7,list = F,groups = 100)
trainA = analysisData_cln[split,]
testA = analysisData_cln[-split,]

dim(trainA) # 20430 --> 0.701
dim(testA) # 8712 --> 0.299


### Start Modelin': Random Forest without & with k-fold cross-validation:
## Random Forest Model:
set.seed(100)
forest = randomForest(price~.,trainA,ntree = 500, na.action=na.exclude) # set ntree at 2, 100, 500 and 1000 if possible.

predForet = predict(forest,trainA,n.trees = 500)
predForet = predict(forest,newdata=testA,n.trees = 500)
rmseForet = sqrt(mean((predForet-testA$price)^2)); rmseForet # ***{ntrees: 100, rmse: 56.25423}, {ntrees: 500, rmse: 55.8763}

pred = predict(forest,newdata=scoringData_cln,n.trees=500)
sqrt(mean((pred-scoringData_cln$price)^2)); # rmse: NaN

submissionFile = data.frame(id = scoringData_cln$id, price = pred)
currDate = Sys.Date() # date in format "2018-11-12"
write.csv(submissionFile, paste0("airbnb_", currDate,"_randomForest.csv"), row.names = F) # submit!

names(forest)
summary(forest)
plot(forest)
varImpPlot(forest); # *important plot
importance(forest)  # see variable importance
getTree(forest,k=100)   # View Tree 100
hist(treesize(forest))  # size of trees constructed 

# Plotting:
plot(gg_vimp(forest, nvar=10)) + 
  labs( x='Feature', y= 'Relative Feature Importance', title='Feature Importance: Random Forest Model (RMSE = 53.658)')


## Random Forest Model with 10-fold Cross-validation
# Note: Aarthi, run this on your VM! Very long run-time
trControl=trainControl(method="cv",number=10)
tuneGrid = expand.grid(mtry=1:5)
set.seed(100)
cvForest = train(price~.,data=trainA,
                 method="rf",ntree=500,trControl=trControl,tuneGrid=tuneGrid ) # change ntree accordingly
cvForest  # best mtry was: {ntree:2, mtry:5}, {ntree:500, mtry:5} 
set.seed(100)
forest = randomForest(price~.,data=trainA,ntree = 500,mtry=5) # change mtry accordingly
predForest = predict(forest,newdata=testA)
rmseForest = sqrt(mean((predForest-testA$price)^2)); rmseForest # rmse: 54.2873
# 500 trees runs overnight.

pred = predict(forest,newdata=scoringData_cln,n.trees=500)
sqrt(mean((pred-scoringData_cln$price)^2)); # rmse: NaN
submissionFile = data.frame(id = scoringData_cln$id, price = pred)
currDate = Sys.Date() # date in format "2018-11-12"
write.csv(submissionFile, paste0("airbnb_", currDate,"_randomForestWCV.csv"), row.names = F) # submit!
