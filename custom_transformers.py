from sklearn.datasets import fetch_20newsgroups
from sklearn.base import BaseEstimator, TransformerMixin
# The lemmatizer is actually pretty complicated, it needs Parts of Speech (POS) tags
import nltk
from nltk import pos_tag
# nltk.download('punkt')#, if you need "tokenizers/punkt/english.pickle", choose it
# nltk.download('averaged_perceptron_tagger')


class Importer(BaseEstimator, TransformerMixin):

    def __init__(self, include_headers=True, shuffle=True, random_state=42, categories=None):
        self.include_headers = include_headers
        self.shuffle = shuffle
        self.random_state = random_state
        if categories is not None:
            self.categories = categories
        else:
            self.categories = ['comp.graphics', 'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']

    def transform(self, dummy, *_):
        if self.include_headers:
            raw_documents = fetch_20newsgroups(subset='train',
                                               categories=self.categories,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state)
        else:
            raw_documents = fetch_20newsgroups(subset='train',
                                               categories=self.categories,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state,
                                               remove=('headers',))
        return raw_documents.data

    def fit(self, *_):
        return self

class Lemmatizer(BaseEstimator, TransformerMixin):

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.wnl = nltk.wordnet.WordNetLemmatizer()

    def transform(self, raw_documents, *_):
        if self.enabled:
            return self._lemmatize_and_filter(raw_documents)
        else:
            return raw_documents

    def fit(self, *_):
        return self

    ##################################################################
    # supporting functions:
    def _penn2morphy(self, penntag):
        """ Converts Penn Treebank tags to WordNet. """
        morphy_tag = {'NN': 'n', 'JJ': 'a',
                      'VB': 'v', 'RB': 'r'}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return 'n'

    # def lemmatize_sent(list_word, wnl):
    #     # Text input is string, returns array of lowercased strings(words).
    #     return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
    #             for word, tag in pos_tag(list_word)]

    def _lemmatize_training(self, text):
        # Text input is string, returns array of lowercased strings(words).
        return [self.wnl.lemmatize(word.lower(), pos=self._penn2morphy(tag))
                for word, tag in pos_tag(nltk.word_tokenize(text))]

    def _filter_numbers(self, text_array):
        # Filter out any numbers found in the array of strings
        output = []
        for s in text_array:
            if not s.isdigit():
                # if not a digit...
                try:
                    # if a float, filter out
                    float(s)
                except ValueError:
                    # if not a float, add to output
                    output.append(s)
            else:
                # if a digit, filter out
                pass
        return output

    def _array_to_string(self, text_array, delimeter=""):
        # Converts an array back into a string of words using the provided delimeter to add between each word
        output = ""
        for s in text_array:
            output = output + delimeter + s
        return output

    def _lemmatize_and_filter(self, documents):
        # Performs lemmatization, and number filtering on the given documents
        lemmatized_data = []
        for i in documents:
            # lemmatize the document:
            training_tagged = pos_tag(nltk.word_tokenize(i))
            lemmatized_array = self._lemmatize_training(i)

            # remove numbers from document:
            filtered_array = self._filter_numbers(lemmatized_array)

            # reassemble back to string:
            lemmatized_string = self._array_to_string(filtered_array, ' ')

            # add to final data list
            # print(lemmatized_string)
            lemmatized_data.append(lemmatized_string)

        return lemmatized_data