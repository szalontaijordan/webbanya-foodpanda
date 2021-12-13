#
# Foodpanda - Sentiment analysis
# https://play.google.com/store/apps/details?id=hu.viala.newiapp
#
# Szalontai Jordán
#
# %%
import json
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, plot_roc_curve, plot_confusion_matrix

import matplotlib.pyplot as plt

from scipy.stats import uniform

# %%
f = open('ratings.json', encoding="utf8")
tmp = json.load(f)
tmp = tmp['ratings']

all_stars = np.array([rating['stars'] for rating in tmp])

labels = range(1, 6)
sizes = [len(all_stars[all_stars == '1']), len(all_stars[all_stars == '2']), len(
    all_stars[all_stars == '3']), len(all_stars[all_stars == '4']), len(all_stars[all_stars == '5'])]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.show()

# 0 = negative
# 1 = positive
data = list()

for rating in tmp:
    stars = int(rating['stars'])
    sentiment = 0 if stars <= 3 else 1
    data.append([rating['text'], sentiment])

data = np.array(data)
stopwords = set(np.loadtxt('stopwords.txt', dtype=str, encoding="utf8"))

print(data[:2])
print(list(stopwords)[:10])

X = data[:, 0]
y = data[:, 1]

# print(X[:3])
# print(y[:3])

# %%
labels = 'Negative', 'Positive'
sizes = [len(y[y == '0']), len(y[y == '1'])]
explode = (0, 0.1)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.show()


# %%

# Unigram Counts
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords)
unigram_vectorizer.fit(X)

X_train_unigram = unigram_vectorizer.transform(X)

# Unigram Tf-Idf
unigram_tf_idf_transformer = TfidfTransformer()
unigram_tf_idf_transformer.fit(X_train_unigram)

X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)

# Bigram Counts
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords)
bigram_vectorizer.fit(X)

X_train_bigram = bigram_vectorizer.transform(X)

# Bigram Tf-Idf
bigram_tf_idf_transformer = TfidfTransformer()
bigram_tf_idf_transformer.fit(X_train_bigram)

X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)

# 3gram Counts
trigram_vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=stopwords)
trigram_vectorizer.fit(X)

X_train_trigram = trigram_vectorizer.transform(X)

# 3gram Tf-Idf
trigram_tf_idf_transformer = TfidfTransformer()
trigram_tf_idf_transformer.fit(X_train_trigram)

X_train_trigram_tf_idf = trigram_tf_idf_transformer.transform(X_train_trigram)

# %%


def check(X, y, optimize=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, stratify=y, random_state=2021
    )

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Train score {train_score}")
    print(f"Test score {test_score}")

    if optimize:
        distributions = dict(
            loss=['hinge', 'log', 'modified_huber',
                  'squared_hinge', 'perceptron'],
            learning_rate=['optimal', 'invscaling', 'adaptive'],
            eta0=uniform(loc=1e-7, scale=1e-2)
        )

        random_search_cv = RandomizedSearchCV(
            estimator=clf,
            param_distributions=distributions,
            cv=5,
            n_iter=50
        )
        random_search_cv.fit(X_train, y_train)
        print(f'Best params: {random_search_cv.best_params_}')
        print(f'Best score: {random_search_cv.best_score_}')
        print("-"*20)
    print()
    return clf


# %%
for X in [X_train_unigram, X_train_unigram_tf_idf, X_train_bigram, X_train_bigram_tf_idf, X_train_trigram, X_train_trigram_tf_idf]:
    check(X, y, optimize=True)

# %%
# bigrams, count vectorizer only
# Best params: {'eta0': 0.006833514503363619, 'learning_rate': 'adaptive', 'loss': 'log'}
# Best score: 0.8264840182648403
X_train, X_test, y_train, y_test = train_test_split(
    X_train_bigram, y, train_size=0.75, stratify=y, random_state=2021
)
classifier_on_bigrams = SGDClassifier()
classifier_on_bigrams.eta0 = 0.004419285645977994
classifier_on_bigrams.learning_rate = "adaptive"
classifier_on_bigrams.loss = "log"
classifier_on_bigrams.fit(X_train, y_train)

train_score = classifier_on_bigrams.score(X_train, y_train)
test_score = classifier_on_bigrams.score(X_test, y_test)
print(f"Train score {train_score}")
print(f"Test score {test_score}")

y_pred = classifier_on_bigrams.predict(X_test)

conf_m = confusion_matrix(y_pred, y_test, normalize="true")
print(conf_m)
plot_confusion_matrix(classifier_on_bigrams, X_test, y_test)

plot_roc_curve(classifier_on_bigrams, X_train, y_train, name="Train")
plot_roc_curve(classifier_on_bigrams, X_test, y_test, name="Test")


#%%

def testWithBigrams(text):
    sentiment = classifier_on_bigrams.predict(
        bigram_vectorizer.transform([text])
    )

    outcome = "Negatív" if int(sentiment[0]) == 0 else "Pozitív"
    print(f"{text} --> {outcome}")
    print()


# %%
positives = [
    "5 csillag, kiváló",
    "gyors kiszállítás olcsón mindenképp ajánlom",
    "ennél jobb nincs szerintem, összességében teljesen meg vagyok elégedve",
    "átlátható barátságos felület gyors kiszállítás"
]
negatives = [
    "Használhatatlan rossz alkalmazás!",
    "Hihetetlen, hogy két óra várakozás után sem érkezik meg semmi",
    "1 csillag az biztos",
    "tegnap még működött, mára már be sem tölt az alkalmazás szerintem a szerverrel van a baj"
]

for text in positives:
    testWithBigrams(text)

for text in negatives:
    testWithBigrams(text)

testWithBigrams("😁")
testWithBigrams("🙂")

testWithBigrams("🤔🤔🤔🤔")
testWithBigrams("🤔🤔🤔🤔 rossz")
