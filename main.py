import spacy
import math

from nltk import bigrams
from tabulate import tabulate


class ColPMI:
    def __init__(self):
        self.model = spacy.load('en_core_web_sm')
        self.doc = self.model(self.__load_corpus())
        self.n_tokens = len(self.doc)
        self.bgs = self.__generate_bigrams()

    def __load_corpus(self):
        with open('wiki-en-flower.txt') as text_file:
            text = text_file.read()
            return text

    def __generate_bigrams(self):
        token_list = [str(tok) for tok in self.doc]

        bgs = bigrams(token_list)
        return [str(bg) for bg in bgs]

    def __count_word_frequency(self, word):
        freq = 0
        for tok in self.doc:
            if tok.text == word:
                freq = freq + 1
        return freq

    def __count_bigram_frequency(self, k):
        freq = 0
        for bg in self.bgs:
            if bg == k:
                freq = freq + 1
        return freq

    def __probability(self, x, n):
        return x / n

    def __pmi(self, P_x, P_y, P_xy):
        try:
            return math.log2(P_xy / (P_x * P_y))
        except:
            return 0

    def PMI(self, x, y):
        bg = f"('{x}', '{y}')"

        # frequency
        n_x = self.__count_word_frequency(word=x)
        n_y = self.__count_word_frequency(word=y)
        n_bg = self.__count_bigram_frequency(k=bg)

        # probability
        p_x = self.__probability(n_x, self.n_tokens)
        p_y = self.__probability(n_y, self.n_tokens)
        p_bg = self.__probability(n_bg, len(self.bgs))

        # pmi
        pmi = self.__pmi(p_x, p_y, p_bg)

        return [x, y, n_x, n_y, n_bg, p_x, p_y, p_bg, pmi]


col_pmi = ColPMI()
results = []

results.append(col_pmi.PMI('sunflower', 'seed'))
results.append(col_pmi.PMI('sunflower', 'oil'))
results.append(col_pmi.PMI('sunflower', 'field'))


print(tabulate(results, headers=[
      'x', 'y', 'C(x)', 'C(y)', 'C(x, y)', 'P(x)', 'P(y)', 'P(x, y)', 'PMI'], tablefmt='orgtbl'))
