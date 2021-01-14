from libs.word_beam_search.beam import Beam, BeamList


def word_beam_search(mat, beamWidth, lm):
    """decode matrix, use given beam width and language model"""
    chars = lm.get_all_chars()
    blankIdx = len(chars)  # blank label is supposed to be last label in RNN output
    maxT, _ = mat.shape  # shape of RNN output: TxC

    genesisBeam = Beam(lm)  # empty string
    last = BeamList()  # list of beams at time-step before beginning of RNN output
    last.add_beam(genesisBeam)  # start with genesis beam

    # go over all time-steps
    for t in range(maxT):
        curr = BeamList()  # list of beams at current time-step

        # go over best beams
        bestBeams = last.get_best_beams(beamWidth)  # get best beams
        for beam in bestBeams:
            # calc probability that beam ends with non-blank
            prNonBlank = 0
            if beam.get_text() != '':
                # char at time-step t must also occur at t-1
                labelIdx = chars.index(beam.get_text()[-1])
                prNonBlank = beam.get_pr_non_blank() * mat[t, labelIdx]

            # calc probability that beam ends with blank
            prBlank = beam.get_pr_total() * mat[t, blankIdx]

            # save result
            curr.add_beam(beam.create_child_beam('', prBlank, prNonBlank))

            # extend current beam with characters according to language model
            nextChars = beam.get_next_chars()
            for c in nextChars:
                # extend current beam with new character
                labelIdx = chars.index(c)
                if beam.get_text() != '' and beam.get_text()[-1] == c:
                    prNonBlank = mat[t, labelIdx] * beam.get_pr_blank()  # same chars must be separated by blank
                else:
                    prNonBlank = mat[t, labelIdx] * beam.get_pr_total()  # different chars can be neighbours

                # save result
                curr.add_beam(beam.create_child_beam(c, 0, prNonBlank))

        # move current beams to next time-step
        last = curr

    # return most probable beam
    last.complete_beams(lm)
    bestBeams = last.get_best_beams(1)  # sort by probability
    return bestBeams[0].get_text()
