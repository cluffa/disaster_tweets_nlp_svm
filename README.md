NLP
================

This is project to introduce myself to NLP in R.

With Kaggle disaster tweets dataset:

[Natural Language Processing with Disaster Tweets \|
Kaggle](https://www.kaggle.com/c/nlp-getting-started)

``` r
tweets <- read_csv('./train.csv', show_col_types = FALSE) %>% 
  unnest_tokens(word, text)
head(tweets)
```

    ## # A tibble: 6 x 5
    ##      id keyword location target word  
    ##   <dbl> <chr>   <chr>     <dbl> <chr> 
    ## 1     1 <NA>    <NA>          1 our   
    ## 2     1 <NA>    <NA>          1 deeds 
    ## 3     1 <NA>    <NA>          1 are   
    ## 4     1 <NA>    <NA>          1 the   
    ## 5     1 <NA>    <NA>          1 reason
    ## 6     1 <NA>    <NA>          1 of

### filtering stop words

``` r
tweets <- tweets %>%
  anti_join(tidytext::stop_words)
```

    ## Joining, by = "word"

``` r
tweets
```

    ## # A tibble: 75,829 x 5
    ##       id keyword location target word      
    ##    <dbl> <chr>   <chr>     <dbl> <chr>     
    ##  1     1 <NA>    <NA>          1 deeds     
    ##  2     1 <NA>    <NA>          1 reason    
    ##  3     1 <NA>    <NA>          1 earthquake
    ##  4     1 <NA>    <NA>          1 allah     
    ##  5     1 <NA>    <NA>          1 forgive   
    ##  6     4 <NA>    <NA>          1 forest    
    ##  7     4 <NA>    <NA>          1 fire      
    ##  8     4 <NA>    <NA>          1 la        
    ##  9     4 <NA>    <NA>          1 ronge     
    ## 10     4 <NA>    <NA>          1 sask      
    ## # ... with 75,819 more rows
