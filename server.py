#!/usr/bin/env python

import collections

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

# Other stemmers
from nltk import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Clustering Algorithms
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn import metrics


def native_language_to_iso639_2(native_language):
    if native_language is 'english':
        return 'eng'
    elif native_language is 'finnish':
        return 'fin'
    elif native_language is 'portuguese':
        return 'por'
    elif native_language is 'italian':
        return 'ita'
    elif native_language is 'danish':
        return 'dan'
    elif native_language is 'dutch':
        return 'eng' # not supported for synset analysis
    elif native_language is 'french':
        return 'fra'
    elif native_language is 'german':
        return 'eng' # not supported for synset analysis
    elif native_language is 'norwegian':
        return 'nno'
    elif native_language is 'turkish':
        return 'tur'
    elif native_language is 'swedish':
        return 'swe'
    else:
        return 'eng'

class Nltk(object):
    @classmethod
    def word_tokenize(cls, s, language):
        return nltk.tokenize.word_tokenize(s.decode('utf-8'), language=language)

    @classmethod
    def my_word_tokenize(cls, s, language):
        return nltk.tokenize.word_tokenize(s, language=language)

    @classmethod
    def tweet_tokenize(cls, s, preserve_case, reduce_len, strip_handles):  # tweet_tokenize originale
        return nltk.tokenize.casual.casual_tokenize(s.decode('utf-8'), preserve_case=preserve_case,
                                                    reduce_len=reduce_len, strip_handles=strip_handles)

    @classmethod
    def synsets(cls, s, pos, lang):
        # print("synset s is", s)
        if lang == 'tur':  # turkish is not supported by WordNet (synset and stemmer)
            lang = 'eng'
        return wn.synsets(s, pos=pos, lang=lang)

    @classmethod
    def senti_synset(cls, synset):
        return swn.senti_synset(synset.name())

    # def remove_stopwords(self, words, lang='english'):
    @classmethod
    def remove_stopwords(cls, words, lang):
        sw = stopwords.words(lang)
        #         for index in range(len(words)):
        filtered_words = [w for w in words if w.lower() not in sw]
        return filtered_words
        # return [w for w in words if w.lower() not in sw]

    @classmethod
    def remove_stopwords_for_clustering(cls, text, lang):
        # print "Words before removing sw are", text
        stops = set(stopwords.words(lang))  # same as "sw = stopwords.words(lang)" but may improve performance
        filtered_words = [w for w in text.split(' ') if w.lower() not in stops]
        filtered_string = filtered_words[0]
        for index in range(len(filtered_words) - 1):
            filtered_string = filtered_string + " " + filtered_words[index + 1]
        # print "filtered string 2 is", filtered_string
        return filtered_string


class Clusterization(object):
    def __init__(self):
        self.nltk = Nltk()

    def make_clusterization(self, message_list, language_list):
        ###############################################################################
        # Vectorize texts

        # message_list[index] is an ordinary string (str)
        for index in range(len(message_list)):
            #print "Message number", index, "is=", message_list[index], "\tand its language is", native_language_to_iso639_2(language_list[index])

            message_list[index] = self.nltk.remove_stopwords_for_clustering(message_list[index], language_list[index]) # remove stopwords using identified language
            #print "Message number", index, "after removing stopwords is=", message_list[index]

            if language_list[index] == 'turkish': # turkish is not supported by WordNet (synset and stemmer)
                pass
            else:
                message_list[index] = unicode(message_list[index], 'utf-8') # prevents special characters from triggering exceptions
                tokens = self.nltk.my_word_tokenize(message_list[index], language_list[index])
                #print 'Tokens are', tokens, "and language is", language_list[index]
                #message_list[index] = [WordNetLemmatizer().lemmatize(t.lower()) for t in message_list[index]] # not useful!
                tokens = [SnowballStemmer(language_list[index]).stem(t.lower()) for t in tokens] # SnowballStemmer allows to give language as parameter

                #               ###############################################################################
                #               # This section of code has the purpose of replacing each word into the message with the list of all lemmas of its synsets
                #               ###############################################################################
                #               token_ind=0
                #               #print "Initial tokens is=", tokens
                #               for token_ind in range(len(tokens)):
                #                 #print "Initial token number", token_ind, "is=", tokens[token_ind].encode('utf-8')
                #                 tokens_to_add=""
                #                 synset_number=0
                #                 for s in wn.synsets(tokens[token_ind], pos=None, lang=native_language_to_iso639_2(language_list[index])):
                #                   #print "synset number", synset_number, "of\t", tokens[token_ind].encode('utf-8'), "\tis=", s
                #                   lemmas=s.lemmas()
                #                   #print "lemmas are=\t", lemmas
                #                   for lemma in lemmas:
                #                     #print "lemma name is=\t", lemma.name()
                #                     tokens_to_add = tokens_to_add + " " + lemma.name()
                #                   synset_number +=1
                #                 tokens[token_ind] = tokens[token_ind] + " " + tokens_to_add
                #                 #print "Final token number", token_ind, "is=", tokens[token_ind].encode('utf-8')
                #                 token_ind +=1
                #               #print "Final tokens is=", tokens
                #               ###############################################################################
                #               # End of lemmas section
                #               ###############################################################################

                message_list[index] = tokens[0].encode('utf-8')
                for ind in range(len(tokens)-1):
                    message_list[index] = message_list[index] + " " + tokens[ind+1].encode('utf-8')

                    #print "Message number", index, "after removing stopwords, stemming and synset is=", message_list[index]

        #End of FOR loop

        vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.1, lowercase=True)  # TfidfVectorizer measures the frequency of a term (TF) normalized to its "common usage" (Inverse Document Frequency), using an internal dictionary

        #print "message_list is ", message_list
        tfidf_model = vectorizer.fit_transform(message_list) # returns a (likely sparse) matrix in the form: (input tweets) X (tokens i.e. words)
        #print "\ntfidf_model is ", tfidf_model

        ###############################################################################
        # Compute clustering with MeanShift

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(tfidf_model.toarray(), quantile=0.4,)

        if bandwidth > 0:
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        else:
            ms = MeanShift()

        ms.fit(tfidf_model.toarray())
        ms_labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(ms_labels)
        num_clusters_ = len(labels_unique)

        print "number of clusters estimated by Mean Shift algorithm : %d" % num_clusters_
        if 1 < num_clusters_ < len(message_list):
            print "Silhouette Coefficient for MS: %0.3f"      % metrics.silhouette_score(tfidf_model.toarray(), ms_labels, metric='sqeuclidean')
        #print 'Labels=', ms_labels, '\n'
        ms_clustering = collections.defaultdict(list)

        for idx, label in enumerate(ms_labels):
            ms_clustering[label].append(idx)

        print "Mean Shift results: ", dict(ms_clustering), "\n"

        ###############################################################################
        # Plot result
        #          import matplotlib.pyplot as plt
        #          from itertools import cycle
        #
        #          plt.figure(1)
        #          plt.clf()
        #
        #          colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        #          for k, col in zip(range(num_clusters_), colors):
        #              my_members = ms_labels == k
        #              cluster_center = cluster_centers[k]
        #              plt.plot(tfidf_model.toarray()[my_members, 0], tfidf_model.toarray()[my_members, 1], col + '.')
        #              plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #                       markeredgecolor='k', markersize=14)
        #          plt.title('Estimated number of clusters: %d' % num_clusters_)
        #          plt.show()
        # End of plot result


        ###############################################################################
        # Compute clustering with Affinity Propagation
        #          af = AffinityPropagation().fit(tfidf_model.toarray())
        #          cluster_centers_indices = af.cluster_centers_indices_
        #          af_labels = af.labels_
        #
        #          n_clusters_ = len(cluster_centers_indices)
        #
        #          print 'Estimated number of clusters by AP: %d' % n_clusters_
        #          print "Silhouette Coefficient for AP: %0.3f"      % metrics.silhouette_score(tfidf_model.toarray(), af_labels, metric='sqeuclidean')

        ###############################################################################
        # Compute clustering with K-means
        #          km_model = KMeans(n_clusters=3)  # implements the KMeans clustering algorithm, which requires to specify in advance the final number of clusters
        #          km_model.fit(tfidf_model)
        #
        #          km_clustering = collections.defaultdict(list)
        #
        #          for idx, label in enumerate(km_model.labels_):
        #           km_clustering[label].append(idx)
        #
        #          print "\nK-means results: ", dict(km_clustering)

        print "\nEnd of clusterization script"

        return num_clusters_, ms_labels #returns number of clusters and labels identified by Mean Shift algorithmns number of clusters and labels identified by Mean Shift algorithm