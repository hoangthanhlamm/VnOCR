import re

from libs.word_beam_search.prefix_tree import PrefixTree


class LanguageModel:
    """unigrams/bigrams LM, add-k smoothing"""

    def __init__(self, corpus, chars, wordChars):
        """read text from filename, specify chars which are contained in dataset, specify chars which form words"""
        # read from file
        self.wordCharPattern = '[' + wordChars + ']'
        self.wordPattern = self.wordCharPattern + '+'
        words = re.findall(self.wordPattern, corpus)
        uniqueWords = list(set(words))  # make unique
        self.numWords = len(words)
        self.numUniqueWords = len(uniqueWords)
        self.smoothing = True
        self.addK = 1.0 if self.smoothing else 0.0

        # create unigrams
        self.unigrams = {}
        for w in words:
            w = w.lower()
            if w not in self.unigrams:
                self.unigrams[w] = 0
            self.unigrams[w] += 1 / self.numWords

        # create unnormalized bigrams
        bigrams = {}
        for i in range(len(words) - 1):
            w1 = words[i].lower()
            w2 = words[i + 1].lower()
            if w1 not in bigrams:
                bigrams[w1] = {}
            if w2 not in bigrams[w1]:
                bigrams[w1][w2] = self.addK  # add-K
            bigrams[w1][w2] += 1

        # normalize bigrams
        for w1 in bigrams.keys():
            # sum up
            probSum = self.numUniqueWords * self.addK  # add-K smoothing
            for w2 in bigrams[w1].keys():
                probSum += bigrams[w1][w2]
            # and divide
            for w2 in bigrams[w1].keys():
                bigrams[w1][w2] /= probSum
        self.bigrams = bigrams

        # create prefix tree
        self.tree = PrefixTree()  # create empty tree
        self.tree.add_words(uniqueWords)  # add all unique words to tree

        # list of all chars, word chars and non-word chars
        self.allChars = chars
        self.wordChars = wordChars
        self.nonWordChars = str().join(
            set(chars) - set(re.findall(self.wordCharPattern, chars)))  # else calculate those chars

    def get_next_words(self, text):
        """text must be prefix of a word"""
        return self.tree.get_next_words(text)

    def get_next_chars(self, text):
        """text must be prefix of a word"""
        nextChars = str().join(self.tree.get_next_chars(text))

        # if in between two words or if word ends, add non-word chars
        if (text == '') or (self.is_word(text)):
            nextChars += self.get_non_word_chars()

        return nextChars

    def get_word_chars(self):
        return self.wordChars

    def get_non_word_chars(self):
        return self.nonWordChars

    def get_all_chars(self):
        return self.allChars

    def is_word(self, text):
        return self.tree.is_word(text)

    def get_unigram_prob(self, w):
        """prob of seeing word w."""
        w = w.lower()
        val = self.unigrams.get(w)
        if val is not None:
            return val
        return 0

    def get_bigram_prob(self, w1, w2):
        """prob of seeing words w1 w2 next to each other."""
        w1 = w1.lower()
        w2 = w2.lower()
        val1 = self.bigrams.get(w1)
        if val1 is not None:
            val2 = val1.get(w2)
            if val2 is not None:
                return val2
            return self.addK / (self.get_unigram_prob(w1) * self.numUniqueWords + self.numUniqueWords)
        return 0
