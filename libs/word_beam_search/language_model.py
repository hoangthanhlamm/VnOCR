import re

from libs.word_beam_search.prefix_tree import PrefixTree


class LanguageModel:
    """unigrams/bigrams LM, add-k smoothing"""

    def __init__(self, corpus, chars, wordChars):
        """read text from filename, specify chars which are contained in dataset, specify chars which form words"""
        self.wordCharPattern = '[' + wordChars + ']'
        self.wordPattern = self.wordCharPattern + '+'
        words = re.findall(self.wordPattern, corpus)
        uniqueWords = list(set(words))  # make unique

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
