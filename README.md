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

``` r
mean(tweets$target)
```

    ## [1] 0.4296598

This is close enough to 0.5 that I will still use an accuracy score. If
it becomes a problem I might use f1 score.

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

preprocess_tweets <- function(text, prob = 0.999, extra_blacklist = c()) {
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

x_text <- preprocess_tweets(all_tweets$text)
x_location <- preprocess_tweets(all_tweets$location)
x_keyword <- preprocess_tweets(all_tweets$keyword, 0.9999)

y_target <- tweets$target

# only for use in kaggle predictions
test_x_text <- x_text[(n + 1):nrow(all_tweets),]
test_x_location <- x_location[(n + 1):nrow(all_tweets),]
test_x_keyword <- x_keyword[(n + 1):nrow(all_tweets),]

x_text <- x_text[1:n,]
x_location <- x_location[1:n,]
x_keyword <- x_keyword[1:n,]
```

``` r
tweets$text[1:4]
```

    ## [1] "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"                                                                
    ## [2] "Forest fire near La Ronge Sask. Canada"                                                                                               
    ## [3] "All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected"
    ## [4] "13,000 people receive #wildfires evacuation orders in California"

``` r
x_text[1:4, 1:10]
```

    ## # A tibble: 4 x 10
    ##   earthquak   may reason canada  fire forest  near   ask evacu expect
    ##       <dbl> <dbl>  <dbl>  <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl>  <dbl>
    ## 1         1     1      1      0     0      0     0     0     0      0
    ## 2         0     0      0      1     1      1     1     0     0      0
    ## 3         0     0      0      0     0      0     0     1     1      1
    ## 4         0     0      0      0     0      0     0     0     1      0

Splitting into validation set if training is too slow to use cross
validation

``` r
train_group <- sample(c(TRUE, FALSE), length(y_target), replace = TRUE, prob = c(0.75, 0.25))

train_x_text <- x_text[train_group,]
train_x_location <- x_location[train_group,]
train_x_keyword <- x_keyword[train_group,]

valid_x_text <- x_text[!train_group,]
valid_x_location <- x_location[!train_group,]
valid_x_keyword <- x_keyword[!train_group,]

train_y <- y_target[train_group]
valid_y <- y_target[!train_group]
```

I want to stack 3 random forest models for each of the three variables
so there is better control over the weights of variables that would not
be present in a single large random forest model.

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
rf.model.text <- randomForest(x = train_x_text, y = as.factor(train_y))
rf.model.location <- randomForest(x = train_x_location, y = as.factor(train_y))
rf.model.keyword <- randomForest(x = train_x_keyword, y = as.factor(train_y))

pred_y_text <- predict(rf.model.text, newdata = valid_x_text) %>% as.character() %>% as.numeric()
pred_y_location <- predict(rf.model.location, newdata = valid_x_location) %>% as.character() %>% as.numeric()
pred_y_keyword <- predict(rf.model.keyword, newdata = valid_x_keyword) %>% as.character() %>% as.numeric()

stacked.rf.lm <- lm(valid_y ~ pred_y_text + pred_y_keyword + pred_y_location, family = 'binomial')

pred_y = predict(
  stacked.rf.lm,
  newdata = data.frame(
    pred_y_text = as.numeric(as.character(predict(
      rf.model.text,
      newdata = valid_x_text
    ))),
    pred_y_location = as.numeric(as.character(predict(
      rf.model.location,
      newdata = valid_x_location
    ))),
    pred_y_keyword = as.numeric(as.character(predict(
      rf.model.keyword,
      newdata = valid_x_keyword
    )))
  )
)

accuracy_rf <- mean(round(pred_y) == valid_y)
accuracy_rf
```

    ## [1] 0.7705083

``` r
stacked.rf.lm
```

    ## 
    ## Call:
    ## lm(formula = valid_y ~ pred_y_text + pred_y_keyword + pred_y_location, 
    ##     family = "binomial")
    ## 
    ## Coefficients:
    ##     (Intercept)      pred_y_text   pred_y_keyword  pred_y_location  
    ##         0.16998          0.44545          0.14612          0.05626

This model is not as accurate as I would like. I plan to try other
methods in the future, use the whole dataset with CV, or tune the models
better using a grid searching method. I also want to use the linked
tweets as more predictors.

``` r
tibble(id = test$id,
       target = round(predict(
        stacked.rf.lm,
        newdata = data.frame(
          pred_y_text = as.numeric(as.character(predict(
            rf.model.text,
            newdata = test_x_text
          ))),
          pred_y_location = as.numeric(as.character(predict(
            rf.model.location,
            newdata = test_x_location
          ))),
          pred_y_keyword = as.numeric(as.character(predict(
            rf.model.keyword,
            newdata = test_x_keyword
          )))
        )
       ))) %>% 
  write_csv('./my_submission.csv')
```
