NLP
================

This is project to introduce myself to NLP in R.

With Kaggle disaster tweets dataset:

[Natural Language Processing with Disaster Tweets \|
Kaggle](https://www.kaggle.com/c/nlp-getting-started)

``` r
tweets <- read_csv('./train.csv', show_col_types = FALSE)
glimpse(tweets)
```

    ## Rows: 7,613
    ## Columns: 5
    ## $ id       <dbl> 1, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24,~
    ## $ keyword  <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, N~
    ## $ location <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, N~
    ## $ text     <chr> "Our Deeds are the Reason of this #earthquake May ALLAH Forgi~
    ## $ target   <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0~

I will use object type corpus from the tm package to process the data. A
corpus is a collection of text style character strings or documents.

``` r
Corpus(VectorSource(tweets$text))
```

    ## <<SimpleCorpus>>
    ## Metadata:  corpus specific: 1, document level (indexed): 0
    ## Content:  documents: 7613

``` r
# using tm package

preprocess_tweets <- function(text, prob = 0.99, extra_blacklist = c()) {
  # Sort of One-Hot-Encoding but for entire tweet and encodes each word,
  # removes sparse words
  # input char vector of tweets, prob, extra words to remove
  
  # words to remove
  blacklist <- c( 
    stopwords("english"),
    'û',
    extra_blacklist)
  
  text %>% 
    VectorSource() %>%                  # makes a vector input formatted for corpus()
    Corpus() %>%                        # creates list of documents called a corpus
    tm_map(PlainTextDocument) %>%       # converts to plain text
    tm_map(tolower) %>%                 # makes all lowercase
    tm_map(removePunctuation) %>%       # remove punctuation
    tm_map(removeWords, blacklist) %>%  # remove blacklisted words
    tm_map(stemDocument) %>%            # combine stem words, ex: jumping/jumped
    DocumentTermMatrix() %>%            # converts to binary matrix
    removeSparseTerms(prob) %>%         # keeps top prob/99.5% words
    as.matrix() %>%
    as.tibble() %>% 
    suppressWarnings() %>% 
    return()
}
```

I will use the same function on text, keyword, and location to make it
simple. This adds about 100 more variables than just using text.

``` r
n <- nrow(tweets)
test <- read_csv('./test.csv', show_col_types = FALSE)

all_tweets <- bind_rows(tweets, test)
all_tweets_processed <- bind_cols(
  preprocess_tweets(all_tweets$text),
  preprocess_tweets(all_tweets$location),
  preprocess_tweets(all_tweets$keyword)
)
```

    ## New names:
    ## * evacu -> evacu...3
    ## * california -> california...4
    ## * flood -> flood...11
    ## * new -> new...31
    ## * scream -> scream...66
    ## * ...

``` r
data_x <- all_tweets_processed[1:n,]
data_y <- tweets$target

test_x <- all_tweets_processed[(n + 1):nrow(all_tweets_processed),]
```

``` r
tweets$text[1:4]
```

    ## [1] "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"                                                                
    ## [2] "Forest fire near La Ronge Sask. Canada"                                                                                               
    ## [3] "All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected"
    ## [4] "13,000 people receive #wildfires evacuation orders in California"

``` r
data_x[1:4, 1:10]
```

    ## # A tibble: 4 x 10
    ##    fire  near evacu...3 california...4 peopl wildfir   got  just  caus disast
    ##   <dbl> <dbl>     <dbl>          <dbl> <dbl>   <dbl> <dbl> <dbl> <dbl>  <dbl>
    ## 1     0     0         0              0     0       0     0     0     0      0
    ## 2     1     1         0              0     0       0     0     0     0      0
    ## 3     0     0         1              0     0       0     0     0     0      0
    ## 4     0     0         1              1     1       1     0     0     0      0

Splitting into validation set if training is too slow to use cross
validation

``` r
train_group <- sample(c(TRUE, FALSE), nrow(data_x), replace = TRUE, prob = c(0.75, 0.25))
train_x <- data_x[train_group,]
valid_x <- data_x[!train_group,]

train_y <- data_y[train_group]
valid_y <- data_y[!train_group]
```

I will just apply an easy rf model for now.

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
rf.model <- randomForest(x = train_x, y = as.factor(train_y))
pred_y <- predict(rf.model, newdata = valid_x)

accuracy_rf <- mean(as.factor(pred_y) == as.factor(valid_y))
accuracy_rf
```

    ## [1] 0.7116256

I’ll attempt to best that with an xgboost model.

``` r
library(xgboost)
```

    ## 
    ## Attaching package: 'xgboost'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     slice

``` r
xgb.model <- xgboost(
  data = as.matrix(train_x),
  label = train_y,
  max.depth = 99999999,
  eta = 0.3,
  nthread = 12,
  nrounds = 1000,
  objective = "binary:logistic",
  early_stopping_rounds = 5,
  verbose = 0
  )
```

    ## [20:47:11] WARNING: amalgamation/../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.

``` r
pred_y <- predict(xgb.model, newdata = as.matrix(valid_x))

accuracy_xgb <- mean(round(pred_y) == valid_y)
accuracy_xgb
```

    ## [1] 0.70307

These models are not as accurate as I would like. I plan to try other
methods in the future, use the whole dataset with CV, or tune the models
better using a grid searching method. I also want to use the linked
tweets as more predictors.

``` r
if(accuracy_xgb < accuracy_rf) {
  pred_test_y <- predict(xgb.model, newdata = as.matrix(test_x)) %>% 
    round()
} else {
  pred_test_y <- predict(rf.model, newdata = test_x) %>% 
    as.character() %>% 
    as.numeric()
}

tibble(id = test$id, target = pred_test_y) %>% 
  write_csv('./my_submission.csv')
```
