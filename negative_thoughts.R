require(plyr)
require(stringr)
require(tidyverse)
require(stringi)
require(ICC)
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
thought2$thoughtcat[is.na(thought2$thoughtcat)] <- 0


ICCbare(as.factor(thought2$X_unit_id), thought2$thoughtcat, thought2)
#0.617
ICCbare(as.factor(thought2$X_unit_id), thought2$negative_yn, thought2)
#0.55
















