Disaster Tweets
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
  # bag of words style encoding
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
    removeSparseTerms(prob) %>%         # keeps top 99.9% or specified % of words
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
x_location <- preprocess_tweets(all_tweets$location, 0.9995)
x_keyword <- preprocess_tweets(all_tweets$keyword, 0.9999)

names(x_text) <- paste0('text_', names(x_text))
names(x_location) <- paste0('location_', names(x_location))
names(x_keyword) <- paste0('keyword_', names(x_keyword))

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

Testing accuracy with inclusion of different text fields and cost with
linear svm.

``` r
train_x <- list(text = train_x_text, keyword = train_x_keyword, location = train_x_location)
valid_x <- list(text = valid_x_text, keyword = valid_x_keyword, location = valid_x_location)

c_grid <- c(0.1, 1, 10)
trys <- list(
  c(T, F, F),
  c(T, F, T),
  c(T, T, F),
  c(T, T, T)
)
options <- c('text', 'keyword', 'location')

results <- data.frame()

combinations <- length(c_grid)*length(trys)
j = 0
for (i in trys) {
  for (c in c_grid) {
    svm <- svm(x = bind_cols(train_x[i]), y = train_y, type = 'C', kernel = 'linear', cost = c, scale = FALSE)
    pred_y <- predict(svm, newdata = bind_cols(valid_x[i]))
    accuracy <- mean(pred_y == valid_y)
    results <- bind_rows(results, data.frame(accuracy = accuracy, cost = c, text = i[1], keyword = i[2], location = i[3]))
    
    j = j + 1
    print(paste0(as.character(j), '/', combinations, ' completed'))
  }
}
```

    ## [1] "1/12 completed"
    ## [1] "2/12 completed"
    ## [1] "3/12 completed"
    ## [1] "4/12 completed"
    ## [1] "5/12 completed"
    ## [1] "6/12 completed"
    ## [1] "7/12 completed"
    ## [1] "8/12 completed"
    ## [1] "9/12 completed"
    ## [1] "10/12 completed"
    ## [1] "11/12 completed"
    ## [1] "12/12 completed"

``` r
results %>% arrange(desc(accuracy))
```

    ##     accuracy cost text keyword location
    ## 1  0.7856064  0.1 TRUE    TRUE    FALSE
    ## 2  0.7835934  0.1 TRUE   FALSE     TRUE
    ## 3  0.7835934  0.1 TRUE    TRUE     TRUE
    ## 4  0.7800705  1.0 TRUE   FALSE    FALSE
    ## 5  0.7795672  0.1 TRUE   FALSE    FALSE
    ## 6  0.7775541  1.0 TRUE   FALSE     TRUE
    ## 7  0.7770508  1.0 TRUE    TRUE    FALSE
    ## 8  0.7715148  1.0 TRUE    TRUE     TRUE
    ## 9  0.7579265 10.0 TRUE   FALSE    FALSE
    ## 10 0.7544036 10.0 TRUE    TRUE    FALSE
    ## 11 0.7483644 10.0 TRUE   FALSE     TRUE
    ## 12 0.7393055 10.0 TRUE    TRUE     TRUE

A model trained on text and keywords with cost at 0.1 performed the best
with an accuracy of 0.7856.

``` r
train_x <- bind_cols(train_x_text,train_x_keyword)
valid_x <- bind_cols(valid_x_text,valid_x_keyword)
x <- bind_rows(train_x, valid_x)
y <- c(train_y, valid_y)

svm <- svm(x = x, y = y, type = 'C', kernel = 'linear', cost = 0.1, scale = FALSE)
```

fitting test data

``` r
pred_submit <- predict(svm, newdata = bind_cols(test_x_text, test_x_keyword))
head(data.frame(prediction = pred_submit, text = test$text), 10)
```

    ##    prediction
    ## 1           1
    ## 2           1
    ## 3           1
    ## 4           1
    ## 5           1
    ## 6           1
    ## 7           0
    ## 8           0
    ## 9           0
    ## 10          0
    ##                                                                                                text
    ## 1                                                                Just happened a terrible car crash
    ## 2                                  Heard about #earthquake is different cities, stay safe everyone.
    ## 3  there is a forest fire at spot pond, geese are fleeing across the street, I cannot save them all
    ## 4                                                          Apocalypse lighting. #Spokane #wildfires
    ## 5                                                     Typhoon Soudelor kills 28 in China and Taiwan
    ## 6                                                                We're shaking...It's an earthquake
    ## 7                          They'd probably still show more life than Arsenal did yesterday, eh? EH?
    ## 8                                                                                 Hey! How are you?
    ## 9                                                                                  What a nice hat?
    ## 10                                                                                        Fuck off!

There are a few things I may want to do in the future:  
1. fill out the rest of the keywords  
2. fill out locations with places in text  
3. work with the links  
4. use tweets being replied to
