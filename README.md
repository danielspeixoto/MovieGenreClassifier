# Movie Genre Classifier

In this project we use Multinomial Naive Bayes to predict a Movie Genre using its plot. The data is from [Wikipedia Movie Plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots).

# Preprocessing

These are the steps:

1. Remove stopwords using the nltk package.
2. Text vectorization using the Bag of Words method.
3. Weights the words using Term Frequency - Inverse Document Frequency (TFIDF)

# Results

Here are the results found.

| Algorithm           | Accuracy | Precision | Recall | F-Score |
|---------------------|----------|-----------|--------|---------|
| Multinomial NB      | 0.75     | 0.74      | 0.78   | 0.76    |
| Decision Tree (ID3) | 0.59     | 0.59      | 0.61   | 0.60    |

As seen, the Multinomial Naive Bayes approach is much better than the Decision Tree.

# How to run

Install those libraries:

- nltk
- pandas
- sklearn

And then execute `python3 main.py`, if you want to pre-process the data again and not use the already available one in this repo, run `python3 pre_process.py` before.

