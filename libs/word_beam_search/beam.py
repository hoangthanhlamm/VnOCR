import copy


class Optical:
    """optical score of beam"""

    def __init__(self, prBlank=0.0, prNonBlank=0.0):
        self.prBlank = prBlank  # prob of ending with a blank
        self.prNonBlank = prNonBlank  # prob of ending with a non-blank


class Textual:
    """textual score of beam"""

    def __init__(self, text=''):
        self.text = text
        self.wordHist = []  # history of words so far
        self.wordDev = ''  # developing word
        self.prUnnormalized = 1.0
        self.prTotal = 1.0


class Beam:
    """beam with text, optical and textual score"""

    def __init__(self, lm, useNGrams):
        """creates genesis beam"""
        self.optical = Optical(1.0, 0.0)
        self.textual = Textual('')
        self.lm = lm
        self.useNGrams = useNGrams

    def merge_beam(self, beam):
        """merge probabilities of two beams with same text"""

        if self.get_text() != beam.get_text():
            raise Exception('mergeBeam: texts differ')

        self.optical.prNonBlank += beam.get_pr_non_blank()
        self.optical.prBlank += beam.get_pr_blank()

    def get_text(self):
        return self.textual.text

    def get_pr_blank(self):
        return self.optical.prBlank

    def get_pr_non_blank(self):
        return self.optical.prNonBlank

    def get_pr_total(self):
        return self.get_pr_blank() + self.get_pr_non_blank()

    def get_pr_textual(self):
        return self.textual.prTotal

    def get_score(self):
        return self.get_pr_total() * self.get_pr_textual()

    def get_next_chars(self):
        return self.lm.get_next_chars(self.textual.wordDev)

    def create_child_beam(self, newChar, prBlank, prNonBlank):
        """extend beam by new character and set optical score"""
        beam = Beam(self.lm, self.useNGrams)

        # copy textual information
        beam.textual = copy.deepcopy(self.textual)
        beam.textual.text += newChar

        # do textual calculations only if beam gets extended
        if newChar != '':
            if self.useNGrams:  # use unigrams and bigrams

                # if new char occurs inside a word
                if newChar in beam.lm.get_word_chars():
                    beam.textual.wordDev += newChar
                    nextWords = beam.lm.get_next_words(beam.textual.wordDev)

                    # no complete word in text, then use unigrams of all possible next words
                    numWords = len(beam.textual.wordHist)
                    prSum = 0
                    if numWords == 0:
                        for w in nextWords:
                            prSum += beam.lm.get_unigram_prob(w)
                    # take last complete word and sum up bigrams of all possible next words
                    else:
                        lastWord = beam.textual.wordHist[-1]
                        for w in nextWords:
                            prSum += beam.lm.get_bigram_prob(lastWord, w)
                    beam.textual.prTotal = beam.textual.prUnnormalized * prSum
                    beam.textual.prTotal = beam.textual.prTotal ** (
                            1 / (numWords + 1)) if numWords >= 1 else beam.textual.prTotal

                # if new char does not occur inside a word
                else:
                    # if current word is not empty, add it to history
                    if beam.textual.wordDev != '':
                        beam.textual.wordHist.append(beam.textual.wordDev)
                        beam.textual.wordDev = ''

                        # score with unigram (first word) or bigram (all other words) probability
                        numWords = len(beam.textual.wordHist)
                        if numWords == 1:
                            beam.textual.prUnnormalized *= beam.lm.get_unigram_prob(beam.textual.wordHist[-1])
                            beam.textual.prTotal = beam.textual.prUnnormalized
                        elif numWords >= 2:
                            beam.textual.prUnnormalized *= beam.lm.get_bigram_prob(beam.textual.wordHist[-2],
                                                                                   beam.textual.wordHist[-1])
                            beam.textual.prTotal = beam.textual.prUnnormalized ** (1 / numWords)

            else:  # don't use unigrams and bigrams, just keep wordDev up to date
                if newChar in beam.lm.get_word_chars():
                    beam.textual.wordDev += newChar
                else:
                    beam.textual.wordDev = ''

        # set optical information
        beam.optical.prBlank = prBlank
        beam.optical.prNonBlank = prNonBlank
        return beam

    def __str__(self):
        return '"' + self.get_text() + '"' + ';' + str(self.get_pr_total()) + ';' + str(self.get_pr_textual()) + ';' + str(
            self.textual.prUnnormalized)


class BeamList:
    """list of beams at specific time-step"""

    def __init__(self):
        self.beams = {}

    def add_beam(self, beam):
        """add or merge new beam into list"""
        # add if text not yet known
        if beam.get_text() not in self.beams:
            self.beams[beam.get_text()] = beam
        # otherwise merge with existing beam
        else:
            self.beams[beam.get_text()].merge_beam(beam)

    def get_best_beams(self, num):
        """return best beams, specify the max. number of beams to be returned (beam width)"""
        u = [v for (_, v) in self.beams.items()]
        lmWeight = 1
        return sorted(u, reverse=True, key=lambda x: x.get_pr_total() * (x.get_pr_textual() ** lmWeight))[:num]

    def delete_partial_beams(self, lm):
        """delete beams for which last word is not finished"""
        for (k, v) in self.beams.items():
            lastWord = v.textual.wordDev
            if (lastWord != '') and (not lm.is_word(lastWord)):
                del self.beams[k]

    def complete_beams(self, lm):
        """complete beams such that last word is complete word"""
        for (_, v) in self.beams.items():
            lastPrefix = v.textual.wordDev
            if lastPrefix == '' or lm.is_word(lastPrefix):
                continue

            # get word candidates for this prefix
            words = lm.get_next_words(lastPrefix)
            # if there is just one candidate, then the last prefix can be extended to
            if len(words) == 1:
                word = words[0]
                v.textual.text += word[len(lastPrefix) - len(word):]
