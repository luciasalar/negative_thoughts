require(plyr)
require(stringr)
require(tidyverse)
require(stringi)
require(psych)
require(reshape2)
#result from sentistrengh
#a1228449 contains the agreement data
thoughts_agg <- read.csv("a1228449.csv", header = T, fill=TRUE,row.names=NULL)
thoughts <- read.csv("f1228449.csv", header = T, fill=TRUE,row.names=NULL)
thoughts %>% select(X_unit_id, negative_yn, thoughtcat, text) -> thought2

thoughts_agg %>% select(X_unit_id, negative_yn, negative_yn.confidence, thoughtcat, thoughtcat.confidence, text) -> thought_agg2

#crowdflower agreement
#Once a job is complete, all of the judgments on a row of data will be aggregated with a confidence score. 
#The confidence score describes the level of agreement between multiple contributors (weighted by each contributors’ trust scores), and indicates our “confidence” in the validity of the aggregated answer for each row of data. The aggregate result is chosen based on the response with the greatest confidence.
mean(thoughts_agg$negative_yn.confidence)   
# 0.8573093

#remove na and 0 because user who selected 'no neg thought' at first step did not do any labeling on second step, therefore, 0 should be treated
#as NA
thought_agg2$thoughtcat.confidence[is.na(thoughts_agg$thoughtcat.confidence)] <- 0
filter(thoughts_agg, thoughtcat.confidence != "0" ) -> thought_cat

mean(thought_cat$thoughtcat.confidence, na.rm = TRUE)
#0.7535739



#recode label
thought2$negative_yn <- revalue(thought2$negative_yn, c("yes"="1", "no"="0", "mixed" = "2"))
thought2$negative_yn <- as.numeric(thought2$negative_yn)
#users who selected no neg thoughts assign 0 at the second step 
thought2$thoughtcat[is.na(thought2$thoughtcat)] <- 0


ICCbare(as.factor(thought2$X_unit_id), thought2$thoughtcat, thought2)
#0.617
ICCbare(as.factor(thought2$X_unit_id), thought2$negative_yn, thought2)
#0.55


#####################select the first 3 in each group
#### ICC negative thought
group_by(thought2, X_unit_id) -> grouped_thought
top_three <- ddply(grouped_thought,.(X_unit_id), head,3) %>% select(X_unit_id, negative_yn)
top_three$negative_yn <- sapply(top_three$negative_yn, function(x) as.numeric(x))

#reshape data into right format for ICC
mdata <- dcast(top_three, c('s1','s2','s3') ~ X_unit_id, value.var = "negative_yn")
colnames(mdata)[1] <- "subjects"
#mdata <- transform(mdata, char = as.numeric(unlist(mdata)))

mdata <- as.matrix(mdata)

mdata2 <-mapply(mdata[,2:226], FUN = as.numeric)

m <- matrix(data=mdata2, ncol=225, nrow=3)
 
ICC(m,missing=TRUE,alpha=.05)

# Call: ICC(x = m, missing = TRUE, alpha = 0.05)
# 
# Intraclass correlation coefficients 
# type    ICC   F df1 df2      p lower bound upper bound
# Single_raters_absolute   ICC1 0.0081 2.8   2 672 0.0593     -0.0010        0.33
# Single_random_raters     ICC2 0.0104 6.1   2 448 0.0025      0.0013        0.33
# Single_fixed_raters      ICC3 0.0221 6.1   2 448 0.0025      0.0028        0.52
# Average_raters_absolute ICC1k 0.6476 2.8   2 672 0.0593     -0.3073        0.99
# Average_random_raters   ICC2k 0.7033 6.1   2 448 0.0025      0.2292        0.99
# Average_fixed_raters    ICC3k 0.8356 6.1   2 448 0.0025      0.3885        1.00


#### ICC context
top_three2 <- ddply(grouped_thought,.(X_unit_id), head,3) %>% select(X_unit_id, thoughtcat)
top_three2$thoughtcat[is.na(top_three2$thoughtcat)] <- 2
top_three2$thoughtcat<- sapply(top_three2$thoughtcat, function(x) as.numeric(x))

#reshape data into right format for ICC
mdata_t <- dcast(top_three2, c('s1','s2','s3') ~ X_unit_id, value.var = "thoughtcat")
colnames(mdata_t)[1] <- "subjects"
#mdata <- transform(mdata, char = as.numeric(unlist(mdata)))

mdata_t <- as.matrix(mdata_t)

mdata3 <-mapply(mdata_t[,2:226], FUN = as.numeric)

m2 <- matrix(data=mdata3, ncol=225, nrow=3)

ICC(m2,missing=TRUE,alpha=.05)

# Call: ICC(x = m2, missing = TRUE, alpha = 0.05)
# 
# Intraclass correlation coefficients 
# type    ICC   F df1 df2     p lower bound upper bound
# Single_raters_absolute   ICC1 0.0058 2.3   2 672 0.099    -0.00167        0.29
# Single_random_raters     ICC2 0.0077 4.1   2 448 0.018     0.00024        0.29
# Single_fixed_raters      ICC3 0.0134 4.1   2 448 0.018     0.00041        0.41
# Average_raters_absolute ICC1k 0.5693 2.3   2 672 0.099    -0.59759        0.99
# Average_random_raters   ICC2k 0.6364 4.1   2 448 0.018     0.05123        0.99
# Average_fixed_raters    ICC3k 0.7538 4.1   2 448 0.018     0.08418        0.99
# 
# Number of subjects = 3     Number of Judges =  225

# ICC1: Each target is rated by a different judge and the judges are selected at random. (This is a
#                                                                                         one-way ANOVA fixed effects model and is found by (MSB- MSW)/(MSB+ (nr-1)*MSW))
# ICC2: A random sample of k judges rate each target. The measure is one of absolute agreement in
# the ratings. Found as (MSB- MSE)/(MSB + (nr-1)*MSE + nr*(MSJ-MSE)/nc)
# ICC3: A fixed set of k judges rate each target
#ICC1k, ICC2k, ICC3K reflect the means of k raters.

