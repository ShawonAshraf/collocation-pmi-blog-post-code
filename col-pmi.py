import spacy
import math

from nltk import bigrams

# download Language model for spacy
# python -m spacy download en_core_web_sm
model = spacy.load('en_core_web_sm')

# read the contents of the text file


def load_corpus():
    with open('wiki-en-flower.txt') as text_file:
        text = text_file.read()

        return text


doc = model(load_corpus())

# count the tokens in the doc
n_tokens = len(doc)
print(f"Number of tokens in the corpus: {n_tokens}\n")


def generate_bigrams(doc):
    token_list = [str(tok) for tok in doc]

    bgs = bigrams(token_list)
    return [str(bg) for bg in bgs]


bgs = generate_bigrams(doc=doc)


def count_word_frequency(word, doc):
    freq = 0
    for tok in doc:
        if tok.text == word:
            freq = freq + 1

    return freq


n_sunflower = count_word_frequency(word='sunflower', doc=doc)
n_seed = count_word_frequency(word='seed', doc=doc)


def count_bigram_frequency(k, bigrams):
    freq = 0
    for bg in bigrams:
        if bg == k:
            freq = freq + 1
    return freq


n_sunflower_seed = count_bigram_frequency(
    k="('sunflower', 'seed')", bigrams=bgs)


def probability(x, n):
    return x / n


def pmi(P_x, P_y, P_xy):
    return math.log2(P_xy / (P_x * P_y))


print(
    f"sunflower = {n_sunflower}\nseed = {n_seed}\nsunflower seed = {n_sunflower_seed}\n")

p_sunflower = probability(n_sunflower, n_tokens)
p_seed = probability(n_seed, n_tokens)
p_sunflower_seed = probability(n_sunflower_seed, len(bgs))

r = pmi(p_sunflower, p_seed, p_sunflower_seed)


print(f"pmi for sunflower seed = {r}")
