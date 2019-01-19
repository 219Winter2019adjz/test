from sklearn.datasets import fetch_20newsgroups
from sklearn.base import BaseEstimator, TransformerMixin
import re
# The lemmatizer is actually pretty complicated, it needs Parts of Speech (POS) tags
import nltk
from nltk import pos_tag
# nltk.download('punkt')#, if you need "tokenizers/punkt/english.pickle", choose it
# nltk.download('averaged_perceptron_tagger')


class Importer(BaseEstimator, TransformerMixin):

    def __init__(self, remove=None):
        if remove is None:
            remove = []
        self.remove = remove

    def transform(self, raw_documents, *_):
        if 'headers' in self.remove:
            raw_documents = [self.strip_newsgroup_header(text) for text in raw_documents]
        if 'footers' in self.remove:
            raw_documents = [self.strip_newsgroup_footer(text) for text in raw_documents]

        return raw_documents

    def fit(self, *_):
        return self

    ####################################################################################################################
    ## Taken from twenty_newsgroups.py
    @staticmethod
    def strip_newsgroup_header(text):
        """
        Given text in "news" format, strip the headers, by removing everything
        before the first blank line.

        Parameters
        ----------
        text : string
            The text from which to remove the signature block.
        """
        _before, _blankline, after = text.partition('\n\n')
        return after

    _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                           r'|^In article|^Quoted from|^\||^>)')

    @staticmethod
    def strip_newsgroup_footer(text):
        """
        Given text in "news" format, attempt to remove a signature block.

        As a rough heuristic, we assume that signatures are set apart by either
        a blank line or a line made of hyphens, and that it is the last such line
        in the file (disregarding blank lines at the end).

        Parameters
        ----------
        text : string
            The text from which to remove the signature block.
        """
        lines = text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break

        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text

    ##
    ####################################################################################################################

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