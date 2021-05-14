"""
HOw to process all the text, using yield

how to find all the interesting words?
- easily can count the highest number of words of each type in document
    - how to count only certainly kinds of words
- then can easily iterate through actually finding where each is located and storing the sentence
    - probably loop through the sentences
    - could also use concordance, although in instructions it says sentence

remove all punctuation inc from stop words then remove stop words then count them
- but if the interesting words have punctuation how do we do that?
- but it doesn't have to be those words...
- if i have another interesting words criteriait can not be those words

the moment you're saving stuff in a variable it's liable to get a bit crazy memorywise

"""
import os
from collections import defaultdict
from string import punctuation

from nltk import word_tokenize, sent_tokenize, pos_tag, FreqDist, TweetTokenizer
from nltk.corpus import stopwords


class WordContextFinder:
    # TODO: must also find document...
    #   - tag words with document?
    #   - tag words with sentence index
    #   - how easy would it be to map each word to its sentences
    #   - perhaps what we need to do is to map each sentence to a document
    #   - what about words that appear in a sentence which starts on one doc but the word is in the other doc

    def __init__(self, sentences, words):
        self.contexts = self.get_word_contexts(sentences, words)

    @staticmethod
    def get_word_contexts(sentences, words):
        # context_dict = {k: [] for k in set(words)}
        context_dict = defaultdict(list)
        for sentence in sentences:
            intersection = set(words).intersection(set(sentence.split()))
            for key in intersection:
                context_dict[key].append(sentence)
        return context_dict


class DocumentTextExtractor:

    @staticmethod
    def get_string_from_document(document_name):
        with open(document_name) as file:
            return file.read()

    def get_string_from_directory(self, directory_name):
        # TODO: error handling on opening files
        pass


class WordTokenizer:

    def __init__(self, text, word_tokenizer, document_name=None):
        self.sentences = self.tokenize_into_sentences(text)
        self.words = self.tokenize_into_words(text, word_tokenizer)

    @staticmethod
    def tokenize_into_sentences(text):
        return sent_tokenize(text)

    @staticmethod
    def tokenize_into_words(text, word_tokenizer):
        tweet_tokenizer = word_tokenizer()
        return tweet_tokenizer.tokenize(text)


class WordNormalizer:

    @staticmethod
    def remove_from_tokens(tokens, remove_list):
        return [token for token in tokens if token not in remove_list]


class WordCounter:

    @staticmethod
    def most_common_words(words, number):
        return FreqDist(words).most_common(number)

    def filter(self):
        pass


if __name__ == '__main__':
    wc = WordCounter()
    non_stop_words = wc.get_all_non_stop_words('documents')
