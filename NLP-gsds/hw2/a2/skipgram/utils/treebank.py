#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random
class RTE_dataset:
    
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            path = "utils/datasets/RTE"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        """ Method to extract tokens from sentences """ 
        
        # If tokens are already computed and stored, return them
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        # Initialize dictionaries and variables for tokenization
        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        # Iterate through sentences in the dataset
        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                # If the word is not in the tokens dictionary, add it
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1 # Initialize word frequency to 1
                    idx += 1
                else:
                    tokenfreq[w] += 1 # Increment word frequency

        # Add an "UNK" token for unknown words
        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        # Store computed tokens and related information
        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens
    
    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/test.tsv", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower() for w in splitted]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[w for w in s
            if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
            for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences

        return self._allsentences
        
    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)
    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb
    
    def sampleTokenIdx(self, outsideWordIdx, K):
        
        """
        Method to sample a token index based on reweighted token frequencies
        """
        
        if not (hasattr(self, "_samplingFreq") and self._samplingFreq is not None):
            
            nTokens = len(self.tokens()) # initialize self._tokenfreq, self._tokens, self._revtokens and get num tokens
            self._samplingFreq = np.zeros((nTokens,)) # initialize sampling frequency

            i = 0
            for w in range(nTokens):
                w = self._revtokens[i]
                if w in self._tokenfreq: # if w in token frequency dict
                    freq = 1.0 * self._tokenfreq[w] 
                    ## ToDo: Reweigh the frequency by taking 0.75th power of frequncy. (1 line)
                    freq = freq ** 0.75
                    ##
                else: # if w is not in token frequency dict
                    freq = 0.0
                self._samplingFreq[i] = freq
                i += 1

            # convert frequency to probability by dividing by the total frequency
            self._samplingFreq /= np.sum(self._samplingFreq)
        
        tmpSamplingFreq = np.copy(self._samplingFreq)
        # make the outsideWordIdx's sampling frequency zero
        tmpSamplingFreq[outsideWordIdx] = 0.0
        # convert modified tmpSamplingFreq back to probability by dividing by the sum (probability must sum to 1.)
        tmpSamplingFreq /= tmpSamplingFreq.sum()
        
        ## ToDo: Sample K indices according to tmpSamplingFreq (1~2lines) (hint: you can use np.random.choice)
        idx = None
        idx = list(np.random.choice(len(tmpSamplingFreq), size = K, replace = False, p = tmpSamplingFreq))
        ## 
        return idx