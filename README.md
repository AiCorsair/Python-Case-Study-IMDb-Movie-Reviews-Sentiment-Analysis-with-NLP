# Python Case Study: IMDb Movie Reviews Sentiment Analysis with NLP


## Table of Contents

- [I. Introduction](#I.-Introduction)

- [II. Benefits of Sentiment Analysis](#II.-Benefits-of-Sentiment-Analysis)

- [III. Data Importing & Preprocessing](#III.-Data-Importing-&-Preprocessing)

- [IV. Notebook 1 – Initial Data Exploration](#IV.-Notebook-1-–-Initial-Data-Exploration)

- [V. Notebook 2 – Feature Selection, Model Training, Evaluation, & Tuning](#V.-Notebook-2-–-Feature-Selection,-Model-Training,-Evaluation,-&-Tuning)

- [VI. Notebook 3 – Logistic Regression & Feature Contributions Evaluation](#VI.-Notebook-3-–-Logistic-Regression-&-Feature-Contributions-Evaluation)

- [VII. Key Visualizations](#VII.-Key-Visualizations)

    - [1. Distribution of Review Lengths](#1.-Distribution-of-Review-Lengths)

    - [2. Unigrams in Positive Reviews](#2.-Unigrams-in-Positive-Reviews)

    - [3. Unigrams in Negative Reviews](#3.-Unigrams-in-Negative-Reviews)

    - [4. Bigrams in Positive Reviews](#4.-Bigrams-in-Positive-Reviews)

    - [5. Bigrams in Negative Reviews](#5.-Bigrams-in-Negative-Reviews)

- [VIII. Evaluation of Baseline & Optimized Models](#VIII.-Evaluation-of-Baseline-&-Optimized-Models)


## I. Introduction

In this project, we developed several machine learning models (e.g. Logistic Regression, Naïve Bayes) to classify movie reviews as positive or negative using the [IMDb Dataset of Movie Reviews](https://www.kaggle.com/datasets/crisbam/imdb-dataset-of-65k-movie-reviews-and-translation).

Our workflow included data preprocessing, visualization, feature selection, model building and tuning. We concluded by visualizing feature contributions and explaining predictions for logistic regression, the best-performing model.

The dataset contains `150,000` movie reviews and `14,206` unique movie titles, sourced from [Kaggle](https://www.kaggle.com/), with a total size of 384.06 MB in CSV format. Each entry includes:

- **Rating:** User rating from `1` to `10`.

- **Review:** User review in English.

- **Movie:** Name of the movie.

- **Resenhas:** Portuguese translation of the reviews.

Since our focus was on English reviews, we excluded the Portuguese ones.


## II. Benefits of Sentiment Analysis

Implementing sentiment analysis in NLP offers several advantages:

-	**Understanding Customers:** Helps businesses learn customer opinions about their products or services, leading to meaningful improvements.

-	**Better Customer Service:** Enables quick responses to customer concerns, boosting satisfaction and loyalty.

-	**Staying Competitive:** Monitors competitor mentions to reveal market trends and opportunities.

-	**Smart Market Research:** Gauges public sentiment on new products, predicting success and guiding refinements.

Sentiment analysis of IMDb movie reviews offers several benefits:

-	**Content Improvement:** Filmmakers can assess audience reactions to pinpoint strengths and areas for improvement.

-	**Marketing Strategies:** Helps craft targeted campaigns that align with audience preferences.

-	**Reputation Management:** Enables timely responses to negative reviews, maintaining a positive brand image.

-	**Consumer Insights:** Aggregates sentiment data to reveal audience trends and guide future content creation.

In summary, sentiment analysis of IMDb reviews provides actionable insights to improve content, refine marketing, and understand audience preferences.


## III. Data Importing & Preprocessing

For each notebook, we began by:

-	Importing relevant libraries and loading the IMDb reviews dataset from CSV (the data was stored in a folder named “Data” using the path `Data/IMDB_Dataset.csv`).

-	Customizing stop words by adding and removing specific words.

-	Cleaning review data using multiple preprocessing steps (e.g. removing stop words, expanding contractions).

**Note:** Our cleaning method was updated early on to reduce or remove specific repeated word sequences discovered during initial exploration.

For each notebook (except the first), we also:

-	Assigned multiclass labels to ratings and filtered data to exclude neutral ratings (`0` for negative and `1` for positive).

-	Defined a tokenizer class to lemmatize words in the text.

-	Split the data into training and testing sets.


## IV. Notebook 1: Initial Data Exploration

-	We performed exploratory data analysis (EDA) with statistics (e.g. checking for missing values, counting unique values) and visuals (word clouds of positive and negative reviews, distributions of reviews, ratings, and words with histograms).

-	We visualized the top `20` unigrams, bigrams, trigrams, 4-grams, and 5-grams in positive and negative reviews.

-	We identified bigrams, trigrams, 4-grams, and 5-grams with identical words (e.g. "good good," "blah blah blah," "la la la la") across all reviews. These were either reduced to a single word (if meaningful) or completely removed after updating our cleaning process.


## V. Notebook 2: Feature Selection, Model Training, Evaluation, & Tuning

-	We vectorized text data using TfidfVectorizer with lemmatization and `5,000` 1- to 3-grams, then trained and evaluated baseline models (e.g. Logistic Regression, Naïve Bayes).

-	After tuning model hyperparameters using GridSearchCV, the optimized models performed better overall compared to the baselines, with Random Forest showing significantly less overfitting.


## VI. Notebook 3: Logistic Regression & Feature Contributions Evaluation

-	We vectorized text data using TfidfVectorizer with lemmatization and `5,000` 1- to 3-grams.

-	Using the baseline logistic regression model, we plotted a confusion matrix to visualize true vs. predicted labels.

-	We stored false positive and false negative reviews in dictionaries and displayed the first few ones based on their ratings.

-	We visualized the marginal contribution of features for the logistic regression model, with notable features including “bad,” “great,” “worst,” and “fun.”

-	We explained model predictions for a false positive and a false negative review based on feature contributions, providing better insight into how features influenced outcomes.


## VII. Key Visualizations

### 1. Distribution of Review Lengths

![1  Distribution of Review Lengths](https://github.com/user-attachments/assets/703b79e8-3274-4998-b74f-dabcb6dbc3ff)

-	Both positive and negative review lengths are right-skewed and follow a similar pattern.

-	Many reviews are under `1,000` characters, with only some exceeding `2,000` characters.

### 2. Unigrams in Positive Reviews

![2  Unigrams in Positive Reviews](https://github.com/user-attachments/assets/d31131f1-14cd-43fb-b99a-8313d5a5a516)

-	Notable words include "like," "good," and "great."

-	Interestingly, "not" also appears frequently due to the lack of contextual pairing.

### 3. Unigrams in Negative Reviews

![3  Unigrams in Negative Reviews](https://github.com/user-attachments/assets/4d1a536e-e68d-4db6-9f5e-6634019cbb42)

-	Common words include "not" and "bad."

-	Surprisingly, "like" and "good" are also frequent due to the lack of contextual pairing.

### 4. Bigrams in Positive Reviews

![4  Bigrams in Positive Reviews](https://github.com/user-attachments/assets/0ff9a990-4c9b-4fae-a512-5fdc47dd58ed)

- Examples include "one best," "see movie," and "good movie."

-	However, pairs like "movie not," "not like," and "not really" highlight the inability of bigrams to capture full context.

### 5. Bigrams in Negative Reviews

![5  Bigrams in Negative Reviews](https://github.com/user-attachments/assets/623fbd01-b34e-4480-a9cd-ff6832a393ef)

-	Common pairs include "waste time," "not good," and "bad movie."

-	Others like "much better" and "watch movie" also appear, again showing limitations in contextual interpretation.


## VIII. Evaluation of Baseline & Optimized Models

![image](https://github.com/user-attachments/assets/16f95349-c515-44ae-a08d-92d9f9b32559)

Based on the table above:

- The Logistic Regression model performs best in both the baseline and optimized versions, followed by Naïve Bayes, Random Forest, and AdaBoost.

- Random Forest is the only model that overfitted, especially in the baseline version, with the optimized version showing significantly less overfitting and better performance.

- Here are the explanations of the baseline logistic regression test evaluation metrics:

    - **Accuracy (89.16%):** Percentage of correctly predicted sentiments (positive or negative) out of all reviews; measures overall performance in classifying IMDb reviews accurately.

    - **AUC (95.81%):** Measures the model's ability to distinguish between positive and negative sentiments in IMDb reviews; higher AUC indicates better separation between the two sentiment classes.

    - **F1 (89.18%):** Harmonic mean of precision and recall; balances false positives and false negatives, useful for imbalanced datasets.

- The similar accuracy (`89.16%`) and F1 score (`89.18%`) suggest that the model's predictions are well-balanced between true positives and true negatives, as the classes were balanced.
