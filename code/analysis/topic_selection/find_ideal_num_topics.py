import matplotlib.pyplot as plt
import pandas as pd
import gensim
from tqdm import tqdm
import csv
import os

N = 2
N_GRAMS = "{}-gram".format(N)
VOCAB_SIZE = 15000
MIN_SPEECH_LENGTH = 50

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
ASSETS_DIR = os.path.join(ROOT_DIR, "analysis", "topic_selection", "assets")
MASTER_DF = os.path.join(DATA_DIR, "ngrams_assets_df.csv")
COHERENCE_CHART_NAME = os.path.join(ASSETS_DIR, "coherence_values.png")
COHERENCE_SCORE_FNAME = os.path.join(ASSETS_DIR, "coherence_by_topic_num.csv")


def write(lst, fname):
    with open(fname, "a") as f:
        writer = csv.writer(f)
        writer.writerow(lst)


def get_ngrams(df):
    def ngram_str_to_lst(ngram_str):
        if isinstance(ngram_str, float):
            ngram_str = ""
        return ngram_str.split(".")

    col = df[N_GRAMS].copy()
    return col.apply(ngram_str_to_lst)


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, limit, step)):

        alpha = 50 / num_topics  # <- recommended values per Griffiths and Steyvers 2004
        eta = 0.025

        model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=2,
            workers=2,
            alpha=alpha,
            eta=eta,
        )
        model_list.append(model)
        coherencemodel = gensim.models.CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


if __name__ == "__main__":

    # read data
    print("Reading data...")
    df = pd.read_csv(MASTER_DF).drop("Unnamed: 0", axis=1)
    dates = df["Date"].copy()

    # get ngrams, build vocab, and bag of ngrams
    doc_ngrams = get_ngrams(df)
    vocab_dict = gensim.corpora.Dictionary(doc_ngrams)
    vocab_dict.filter_extremes(no_below=15, no_above=0.25, keep_n=VOCAB_SIZE)
    corpus = [vocab_dict.doc2bow(doc) for doc in doc_ngrams]

    # Show graph
    limit = 41
    start = 5
    step = 1

    model_list, coherence_values = compute_coherence_values(
        dictionary=vocab_dict,
        corpus=corpus,
        texts=doc_ngrams,
        start=start,
        limit=limit,
        step=step,
    )

    x = range(start, limit, step)
    sorted_scores = sorted(
        list(zip(*(x, coherence_values))), reverse=True, key=lambda x: x[1]
    )

    headers = ["NumTopics", "CoherenceScore"]
    write(headers, COHERENCE_SCORE_FNAME)
    for i, c in sorted_scores:
        print("{} topics : {} coherence score".format(i, c))
        write([i, c], COHERENCE_SCORE_FNAME)

    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc="best")
    plt.savefig(COHERENCE_CHART_NAME)
    plt.show()
