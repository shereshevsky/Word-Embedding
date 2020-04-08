import re
import os
import pickle
import spacy
import logging
import multiprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from collections import defaultdict, Counter
from spacy_langdetect import LanguageDetector
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

sns.set_style("darkgrid")
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
non_alpha = re.compile("[^A-Za-z']+")
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

model_id = "model_4"

if not os.path.exists(f'data/{model_id}'):
    os.mkdir(f'data/{model_id}')
clean_corpus_file = f'data/{model_id}/clean_texts.pkl'
model_file = f'data/{model_id}/word2vec.model'
bigrams_file = f'data/{model_id}/phrases'
trigrams_file = f'data/{model_id}/phrases_2'
vocabulary_file = f'data/{model_id}/vocabulary'
predicted_sentiment_file = f"data/{model_id}/predicted_sentiment.txt"
genre_counts_file = f"data/{model_id}/genre_counts.txt"
relative_abundance_file = f"data/{model_id}/relative_abundance.csv"
X_w2v_vectors_file = f"data/{model_id}/X_w2v_vectors.txt"
X_w2v_vectors_pickled_file = f"data/{model_id}/X_w2v_vectors.np"

# Load sentiment dataset
df = pd.read_csv("/home/alexs/Dropbox/ydata/DL/week3/assignment 3/380000-lyrics-from-metrolyrics.zip", header=0)
sentiment = pd.read_csv('data/SemEval2015-English-Twitter-Lexicon.txt', delimiter='\t', header=0,
                        names=['sentiment', 'term'])
sentiment.term = sentiment.term.str.replace('#', '')
sentiment.set_index('term', inplace=True)

sentiment_terms = ['_'.join(i.split(' ')).replace("'", "_'_") for i in sentiment.index if ' ' in i]

if not Path(clean_corpus_file).exists():
    # Clean texts
    def cleaning(doc):
        _txt = [token.text for token in doc if not token.is_stop]  # tokenize, remove stopwords, lowercase
        if len(_txt) > 2 and doc._.language["language"] == "en":
            return ' '.join(_txt)


    brief_cleaning = (non_alpha.sub(' ', str(row)).lower() for row in df.lyrics)  # Remove any non-alpha words

    t = time()
    texts = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=6)]
    pickle.dump(texts, open(clean_corpus_file, 'wb'))

    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

texts = pickle.load(open(clean_corpus_file, 'rb'))

df_clean = pd.DataFrame({'clean': texts})
df['clean_lyrics'] = df_clean

print(df_clean.shape)


# Build Word2Vec model
if not Path(model_file).exists():
    sent = [row.split() for row in df['clean_lyrics'] if row]
    # Build collocations
    if not Path(bigrams_file).exists():
        bigram_phrases = Phrases(sent, min_count=30, progress_per=10000, max_vocab_size=200000,
                                 common_terms=sentiment_terms)
        bigram = Phraser(bigram_phrases)
        bigram.save(bigrams_file)
        trigram_phrases = Phrases(bigram[sent], min_count=30, progress_per=10000, max_vocab_size=200000,
                                  common_terms=sentiment_terms)
        trigram = Phraser(trigram_phrases)
        trigram.save(trigrams_file)

    trigram = Phrases.load(trigrams_file)

    sentences = trigram[sent]

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=20,  # Remove rare words
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1)

    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.vocabulary.save(vocabulary_file)

    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.save(model_file)

trigram = Phrases.load(trigrams_file)
w2v_model = Word2Vec.load(model_file)

# Word2Vec sanity check
print(w2v_model.wv.most_similar(positive=["boy"], negative=["man"]))
print(w2v_model.wv.most_similar(positive=["doctor"], negative=["man"]))
print(w2v_model.wv.most_similar(positive=["doctor"], negative=["woman"]))
print(w2v_model.wv.most_similar(positive=["love", "vodka"]))

print(w2v_model.wv.most_similar(positive=["demon"]))
print(w2v_model.wv.most_similar(positive=["heaven"]))
print(w2v_model.wv.most_similar(positive=["best"]))

print(w2v_model.wv.doesnt_match(["best", "good", "better", "god"]))

# res = []
# for i, row in enumerate(df.lyrics):
#     song_score = 0
#     song = non_alpha.sub(' ', str(row)).lower()
#     for term in sentiment.index.values:
#         if term in song:
#             song_score += float(sentiment.loc[term].sentiment.max()) * len(list(re.finditer(term, song)))
#     if i % 1000 == 0:
#         print(i, song_score)
#     res.append(song_score)
# pickle.dump(res, open("data/song_scores_with_multiplication.pkl", "wb"))
# l = pickle.load(open('data/song_scores.pkl', 'rb'))
# np.array(l).argmax()
# Out[51]: 356347
# np.array(l).argmin()
# Out[52]: 336979


# Prepare sentiment for words found in sentiment dataset and w2v model
results = []
for v in sentiment.index.values:
    if v in w2v_model.wv:  # found in w2v by exact match
        results.append(np.array(w2v_model.wv[v]))
    elif '_'.join([i.lemma_ for i in nlp(v)]) in w2v_model.wv:  # found in w2v by lemma
        results.append(np.array(w2v_model.wv['_'.join([i.lemma_ for i in nlp(v)])]))
    else:  # not found in w2v
        results.append(None)

print(len(results))

print(np.array(results).shape)

t = np.concatenate((sentiment.reset_index(), np.array(results).reshape(1514, -1)), axis=1)
p = pd.DataFrame(t)
p = p[~p[2].isnull()]

# train linear regression model on using w2v vectors and known sentiment
X = p[2].values.tolist()
y = list(p[1].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
np.stack((model.predict(X_test), y_test), axis=1)

print(np.mean((y_train - model.predict(X_train)) ** 2))
print(np.mean((y_test - model.predict(X_test)) ** 2))

# predict sentiment for all the words in the w2v vocabulary
with open(predicted_sentiment_file, "wt") as f:
    for i, k in enumerate(w2v_model.wv.vocab):
        print(f"{k}:{model.predict([w2v_model.wv[k]])[0]}", file=f)

# Visualizing Word Vectors
# Perform the following:
#
# Keep only the 3,000 most frequent words (after removing stopwords)
# For this list, compute for each word its relative abundance in each of the genres
# Compute the ratio between the proportion of each word in each genre and the proportion of the word
# in the entire corpus (the background distribution)
# Pick the top 50 words for each genre. These words give good indication for that genre.
# Join the words from all genres into a single list of top significant words.
# Compute tSNE transformation to 2D for all words, based on their word vectors
# Plot the list of the top significant words in 2D. Next to each word output its text.
# The color of each point should indicate the genre for which it is most significant.


if not Path(genre_counts_file).exists():
    counts = defaultdict(Counter)

    for i, (k, v) in enumerate(df.iterrows()):
        if v.clean_lyrics:
            counts[v.genre] += Counter([t for t in v.clean_lyrics.split() if len(t) > 2])

    pickle.dump(counts, open(genre_counts_file, "wb"))
else:
    counts = pickle.load(open(genre_counts_file, "br"))

if not Path(relative_abundance_file).exists():
    total_count = Counter()
    for k in counts.keys():
        total_count += counts[k]

    total_elements = sum(total_count.values())

    top_3k = total_count.most_common(3000)

    results = []
    for w in top_3k:
        res = {"term": w[0], "proportion_in_corpus": w[1] / total_elements, }
        for k in counts.keys():
            res[k] = (counts[k][w[0]] / sum(counts[k].values())) / w[1] / total_elements
        results.append(res)

    relative_abundance_df = pd.DataFrame.from_records(results).set_index('term')
    relative_abundance_df.to_csv(relative_abundance_file)

relative_abundance_df = pd.read_csv(relative_abundance_file).set_index('term')

print(relative_abundance_df['Country'].sort_values(ascending=False)[:10])
print(relative_abundance_df['Metal'].sort_values(ascending=False)[:10])

genre_colors = zip(
    ['Pop', 'Hip-Hop', 'Not Available', 'Rock', 'Metal', 'Other', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B',
     'Indie'],
    ['yellow', 'red', 'blue', 'orange', 'purple', 'green', 'brown', 'black', 'grey', 'pink', 'teal', 'tan']
)

arrays = np.empty((0, 300), dtype='f')
word_labels = []
color_list = []

for genre, color in genre_colors:
    for word in relative_abundance_df[genre].sort_values(ascending=False)[:50].index:
        arrays = np.append(arrays, w2v_model.wv[word].reshape(1, 300), axis=0)
        word_labels.append(word)
        color_list.append(color)

# Reduces the dimensionality from 300 to 50 dimensions with PCA
reduc = PCA(n_components=50).fit_transform(arrays)

# Finds t-SNE coordinates for 2 dimensions
np.set_printoptions(suppress=True)

Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)

# Sets everything up to plot
tnse_df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                        'y': [y for y in Y[:, 1]],
                        'words': word_labels,
                        'color': color_list})

fig, _ = plt.subplots()
fig.set_size_inches(15, 15)

# Basic plot
p1 = sns.regplot(data=tnse_df,
                 x="x",
                 y="y",
                 fit_reg=False,
                 marker="o",
                 scatter_kws={'s': 40,
                              'facecolors': tnse_df['color']
                              }
                 )

# Adds annotations one by one with a loop
for line in range(0, tnse_df.shape[0]):
    p1.text(tnse_df["x"][line],
            tnse_df['y'][line],
            '  ' + tnse_df["words"][line].title(),
            horizontalalignment='left',
            verticalalignment='bottom', size='medium',
            color=tnse_df['color'][line],
            weight='normal'
            ).set_size(15)

plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

plt.title('t-SNE visualization for top 50 terms for each genre')
plt.show()

# Build a Naive Bayes classifier based on the bag of Words.
# You will need to divide your dataset into a train and test sets.


sklearn_pipeline = Pipeline([
    ('vect', CountVectorizer(max_features=10000)),
    ('clf', MultinomialNB()),
])

y = list(df[~df.genre.isin(['Not Available', 'Other'])].drop_duplicates().dropna().genre.values)
X = list(df[~df.genre.isin(['Not Available', 'Other'])].drop_duplicates().dropna().clean_lyrics.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sklearn_pipeline.fit(X_train, y_train)

print("sklearn multinomial NB accuracy =", sum(sklearn_pipeline.predict(X_test) == y_test)/len(y_test))

y_pred = sklearn_pipeline.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cnf_matrix, annot=True, fmt="d")
plt.show()

# Text classification using Word Vectors
# Average word vectors
# Do the same, using a classifier that averages the word vectors of words in the document.

if not Path(X_w2v_vectors_file).exists():
    t = time()
    # X_vectors = np.empty((0, 300), dtype='f')
    with open(X_w2v_vectors_file, 'a') as f_handle:
        for song in X:
            song_arrays = np.empty((0, 300), dtype='f')
            for word in trigram[song.split()]:
                if word in w2v_model.wv:
                    song_arrays = np.append(song_arrays, w2v_model.wv[word].reshape(1, 300), axis=0)

            np.savetxt(f_handle, song_arrays.mean(axis=0))
        # X_vectors = np.vstack([X_vectors, song_arrays.mean(axis=0).reshape(1, 300)])
    print('Time to prepare mean vectors: {} mins'.format(round((time() - t) / 60, 2)))

    # X_vectors.dump(X_w2v_vectors_file)

    X_vectors = np.loadtxt(X_w2v_vectors_file)

    print(X_vectors.shape)
    X_vectors.dump(X_w2v_vectors_pickled_file)

X_vectors = np.load(X_w2v_vectors_pickled_file, allow_pickle=True).reshape(-1, 300)

X_vec = X_vectors[~np.any(np.isnan(X_vectors), axis=1)]
y_vec = np.array(y)[~np.any(np.isnan(X_vectors), axis=1)]

X_train_vec, X_test_vec, y_train_vec, y_test_vec = train_test_split(X_vec, y_vec, test_size=0.2, random_state=42)

clf = ExtraTreesClassifier()
clf.fit(X_train_vec, y_train_vec)

print("sklearn ExtraTreesClassifier on W2V vectors accuracy =",
      sum(clf.predict(X_test_vec) == y_test_vec)/len(y_test_vec))
# 0.6246185852981969


# TfIdf Weighting
# Do the same, using a classifier that averages the word vectors of words in the document, weighting each word by its TfIdf.

print(len(texts) == len(df.genre.values))

X = pd.Series(texts).values[~pd.Series(texts).isnull()]
y = df.genre.values[~pd.Series(texts).isnull()]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(analyzer=lambda x: x)
tfidf.fit([ii.split() for ii in X_train])
max_idf = max(tfidf.idf_)
word2weight = defaultdict(
    lambda: max_idf,
    [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

train_transformed = np.array([
    np.mean([w2v_model.wv[w] * word2weight[w]
             for w in words if w in w2v_model.wv] or
            [np.zeros(300)], axis=0)
    for words in [ii.split() for ii in X_train]
])

test_transformed = np.array([
    np.mean([w2v_model.wv[w] * word2weight[w]
             for w in words if w in w2v_model.wv] or
            [np.zeros(300)], axis=0)
    for words in [ii.split() for ii in X_test]
])

clf = ExtraTreesClassifier()
clf.fit(train_transformed, y_train)
print("sklearn ExtraTreesClassifier on weighted W2V vectors accuracy =",
      sum(clf.predict(test_transformed) == y_test) / len(y_test))
