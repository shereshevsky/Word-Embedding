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
text_in_brackets = re.compile("\[(.*?)\]")
non_alpha = re.compile("[^A-Za-z']+")
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)

model_id = "model_5"

if not os.path.exists(f'data/{model_id}'):
    os.mkdir(f'data/{model_id}')
clean_corpus_file = f'data/{model_id}/clean_texts.pkl'
model_file = f'data/{model_id}/word2vec.model'
bigrams_file = f'data/{model_id}/bigrams'
trigrams_file = f'data/{model_id}/trigrams'
vocabulary_file = f'data/{model_id}/vocabulary'
predicted_sentiment_file = f"data/{model_id}/predicted_sentiment.txt"
genre_counts_file = f"data/{model_id}/genre_counts.txt"
relative_abundance_file = f"data/{model_id}/relative_abundance.csv"
X_w2v_vectors_file = f"data/{model_id}/X_w2v_vectors.txt"
X_w2v_vectors_pickled_file = f"data/{model_id}/X_w2v_vectors.np"
X_w2v_indexes_for_nn = f"data/{model_id}/X_w2v_indexes"

# Load sentiment dataset
df = pd.read_csv("data/380000-lyrics-from-metrolyrics.zip", header=0)
sentiment = pd.read_csv('data/SemEval2015-English-Twitter-Lexicon.txt', delimiter='\t', header=0,
                        names=['sentiment', 'term'])
sentiment.term = sentiment.term.str.replace('#', '')
sentiment.set_index('term', inplace=True)

sentiment_terms = ['_'.join(i.split(' ')).replace("'", "_'_") for i in sentiment.index if ' ' in i]


def prepare_clean_data(songs_df: pd.DataFrame) -> pd.DataFrame:
    if not Path(clean_corpus_file).exists():
        def cleaning(i, doc):
            if i % 10000 == 0:
                print(f"cleaning data: song {i} / {df.shape[0]} - {round(i / df.shape[0], 2) * 100}%")
            if not doc._.language["language"] == "en":
                return
            _txt = [token.text for token in doc if not token.is_stop]  # tokenize, remove stopwords, lowercase
            if len(_txt) > 2:
                return ' '.join(_txt)

        # Remove any non-alpha words and text in square brackets
        brief_cleaning = (non_alpha.sub(' ', text_in_brackets.sub(' ', str(row))).lower() for row in songs_df.lyrics)
        t = time()
        texts = [cleaning(i, doc) for i, doc in enumerate(nlp.pipe(brief_cleaning, batch_size=6000, n_threads=16))]
        pickle.dump(texts, open(clean_corpus_file, 'wb'))
        print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    texts = pickle.load(open(clean_corpus_file, 'rb'))
    df_clean = pd.DataFrame({'clean': texts})
    songs_df['clean_lyrics'] = df_clean
    return songs_df


def train_w2v_model() -> (Phraser, Word2Vec):
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

    return trigram, w2v_model


def prepare_sentiment_for_known_words():
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
    return p


def prepare_relative_abundance(songs_df):
    """
    # Keep only the 3,000 most frequent words (after removing stopwords)
    # For this list, compute for each word its relative abundance in each of the genres
    # Compute the ratio between the proportion of each word in each genre and the proportion of the word
    # in the entire corpus (the background distribution)
    # Pick the top 50 words for each genre. These words give good indication for that genre.
    :param songs_df:
    :return:
    """
    if not Path(genre_counts_file).exists():
        counts = defaultdict(Counter)

        for i, (k, v) in enumerate(songs_df.iterrows()):
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

    return relative_abundance_df


def plot_tsne(relative_abundance_df):
    """
    # Join the words from all genres into a single list of top significant words.
    # Compute tSNE transformation to 2D for all words, based on their word vectors
    # Plot the list of the top significant words in 2D. Next to each word output its text.
    # The color of each point should indicate the genre for which it is most significant.

    :param relative_abundance_df:
    :return:
    """
    colors_map = zip(
        ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie'],
        ['yellow', 'red', 'blue', 'orange', 'purple', 'green', 'brown', 'black', 'grey', 'pink']
    )
    arrays = np.empty((0, 300), dtype='f')
    word_labels = []
    color_list = []
    for genre, color in colors_map:
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


def train_sentiment_model():
    known_words_sentiment = prepare_sentiment_for_known_words()
    X = known_words_sentiment[2].values.tolist()
    y = list(known_words_sentiment[1].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(normalize=True)
    model.fit(X_train, y_train)
    np.stack((model.predict(X_test), y_test), axis=1)
    print(np.mean((y_train - model.predict(X_train)) ** 2))
    print(np.mean((y_test - model.predict(X_test)) ** 2))
    return model


def predict_sentiment_for_new_tesrms(sentiment_model):
    # predict sentiment for all the words in the w2v vocabulary
    if not Path(predicted_sentiment_file).exists():
        with open(predicted_sentiment_file, "wt") as f:
            for i, k in enumerate(w2v_model.wv.vocab):
                print(f"{k}:{sentiment_model.predict([w2v_model.wv[k]])[0]}", file=f)


def nb_classification_on_bow(songs_df):
    sklearn_pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=10000)),
        ('clf', MultinomialNB()),
    ])
    y = list(songs_df[~df.genre.isin(['Not Available', 'Other'])].drop_duplicates().dropna().genre.values)
    X = list(songs_df[~df.genre.isin(['Not Available', 'Other'])].drop_duplicates().dropna().clean_lyrics.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sklearn_pipeline.fit(X_train, y_train)
    print("sklearn multinomial NB accuracy =", sum(sklearn_pipeline.predict(X_test) == y_test) / len(y_test))
    y_pred = sklearn_pipeline.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 8))
    sns.heatmap(cnf_matrix, annot=True, fmt="d")
    plt.show()


def classification_on_word_vectors(phraser, w2v_model, X, y):
    if not Path(X_w2v_vectors_file).exists():
        t = time()
        with open(X_w2v_vectors_file, 'a') as f_handle:
            for song in X:
                song_arrays = np.empty((0, 300), dtype='f')
                for word in phraser[song.split()]:
                    if word in w2v_model.wv:
                        song_arrays = np.append(song_arrays, w2v_model.wv[word].reshape(1, 300), axis=0)

                np.savetxt(f_handle, song_arrays.mean(axis=0))
        print('Time to prepare mean vectors: {} mins'.format(round((time() - t) / 60, 2)))
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
          sum(clf.predict(X_test_vec) == y_test_vec) / len(y_test_vec))


def classification_with_tfidf_weighting(phraser, w2v_model, X, y):
    """
    # Do the same, using a classifier that averages the word vectors of words in the document, weighting each word by its TfIdf.
    :return:
    """
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
        for words in [phraser[ii.split()] for ii in X_train]
    ])
    test_transformed = np.array([
        np.mean([w2v_model.wv[w] * word2weight[w]
                 for w in words if w in w2v_model.wv] or
                [np.zeros(300)], axis=0)
        for words in [phraser[ii.split()] for ii in X_test]
    ])
    clf = ExtraTreesClassifier()
    clf.fit(train_transformed, y_train)
    print("sklearn ExtraTreesClassifier on weighted W2V vectors accuracy =",
          sum(clf.predict(test_transformed) == y_test) / len(y_test))


if __name__ == '__main__':

    songs_df = prepare_clean_data(df)
    phraser, w2v_model = train_w2v_model()

    # print(w2v_model.wv.most_similar(positive=["boy"], negative=["man"]))
    # print(w2v_model.wv.most_similar(positive=["doctor"], negative=["man"]))
    # print(w2v_model.wv.most_similar(positive=["doctor"], negative=["woman"]))
    # print(w2v_model.wv.most_similar(positive=["love", "vodka"]))
    #
    # print(w2v_model.wv.most_similar(positive=["demon"]))
    # print(w2v_model.wv.most_similar(positive=["heaven"]))
    # print(w2v_model.wv.most_similar(positive=["best"]))
    #
    # print(w2v_model.wv.doesnt_match(["best", "good", "better", "god"]))
    #
    # print("train sentiment")
    # sentiment_model = train_sentiment_model()
    #
    # predict_sentiment_for_new_tesrms(sentiment_model)
    #
    # relative_abundance_df = prepare_relative_abundance(songs_df)
    #
    # print(relative_abundance_df['Country'].sort_values(ascending=False)[:10])
    # print(relative_abundance_df['Metal'].sort_values(ascending=False)[:10])
    #
    # plot_tsne(relative_abundance_df)
    #
    # nb_classification_on_bow(songs_df)
    #
    # X = songs_df.clean_lyrics.values[~songs_df.genre.isin(['Not Available', 'Other']) & ~songs_df.clean_lyrics.isnull()]
    # y = songs_df.genre.values[~songs_df.genre.isin(['Not Available', 'Other']) & ~songs_df.clean_lyrics.isnull()]
    #
    # classification_on_word_vectors(phraser, w2v_model, X, y)
    #
    # classification_with_tfidf_weighting(phraser, w2v_model, X, y)

    import os
    import time
    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.optim as optim
    import numpy as np

    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        print(f"CUDA is available! Training on {torch.cuda.get_device_name(0)}...")
        device = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training on CPU...")
        device = 'cpu'


    def clip_gradient(model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)


    def train_model(model, X_train, y_train, epoch):
        total_epoch_loss = 0
        total_epoch_acc = 0
        if train_on_gpu:
            model.to(device)
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        steps = 0
        model.train()

        batches = range(0, len(X_train), batch_size)
        for idx in batches:
            texts = X_train[idx: idx + batch_size]
            target = y_train[idx: idx + batch_size]
            target = torch.autograd.Variable(torch.LongTensor(target))
            if train_on_gpu:
                texts = torch.FloatTensor(texts).to(device)
                target = target.to(device)

            optim.zero_grad()
            prediction = model(texts)
            loss = loss_fn(prediction, target)
            num_corrects = (prediction.argmax(axis=1).data == target.data).sum()
            acc = 100.0 * num_corrects / len(texts)
            loss.backward()
            clip_gradient(model, 1e-1)
            optim.step()
            steps += 1

            if steps % 100 == 0:
                print(
                    f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss / len(batches), total_epoch_acc / len(batches)


    def eval_model(model, X_test, y_test):
        total_epoch_loss = 0
        total_epoch_acc = 0
        model.eval()
        with torch.no_grad():
            batches = range(0, len(X_test), batch_size)
            for idx in batches:
                texts = X_test[idx: idx + batch_size]
                target = y_test[idx: idx + batch_size]
                target = torch.autograd.Variable(torch.LongTensor(target))
                if train_on_gpu:
                    texts = torch.FloatTensor(texts).to(device)
                    target = target.to(device)

                prediction = model(texts)
                loss = loss_fn(prediction, target)
                num_corrects = (prediction.argmax(axis=1).data == target.data).sum()
                acc = 100.0 * num_corrects / len(texts)
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

        return total_epoch_loss / len(batches), total_epoch_acc / len(batches)


    learning_rate = 2e-5
    batch_size = 32
    output_size = 11
    embedding_length = 300

    from cnn import CNN

    embedding_matrix = np.zeros((len(w2v_model.wv.vocab), 300))
    for i in range(len(w2v_model.wv.vocab)):
        embedding_vector = w2v_model.wv[w2v_model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = CNN(batch_size=batch_size, output_size=output_size, in_channels=1, out_channels=128, kernel_heights=(3, 4, 5),
                stride=1, padding=0, keep_probab=0.2, vocab_size=200000, embedding_length=embedding_length, weights=embedding_matrix)

    loss_fn = F.cross_entropy

    y = list(songs_df[~df.genre.isin(['Not Available', 'Other'])].drop_duplicates().dropna().genre.values)
    X = list(songs_df[~df.genre.isin(['Not Available', 'Other'])].drop_duplicates().dropna().clean_lyrics.values)

    if not Path(X_w2v_indexes_for_nn).exists():
        X_ind = []
        for song in X:
            l = [w2v_model.wv.vocab[i].index if i in w2v_model.wv else 0 for i in phraser[song.split()]][:128]
            l = l + [0] * (128 - len(l))
            X_ind.append(l)
        pickle.dump(X_ind, open(X_w2v_indexes_for_nn, "wb"))
    X_ind = pickle.load(open(X_w2v_indexes_for_nn, "br"))
    generes_mapping = {'Pop': 1, 'Hip-Hop': 2, 'Rock': 3, 'Metal': 4, 'Country': 5, 'Jazz': 6, 'Electronic': 7,
                       'Folk': 8, 'R&B': 9, 'Indie': 10}
    y_ind = [generes_mapping.get(i, 0) for i in y]
    X_train, X_test, y_train, y_test = train_test_split(X_ind, y_ind, test_size=0.2, random_state=42)

    for epoch in range(10):
        train_loss, train_acc = train_model(model, X_train, y_train, epoch)
        val_loss, val_acc = eval_model(model, X_test, y_test)

        print(
            f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
            f'Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
