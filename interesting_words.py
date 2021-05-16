import os
from collections import defaultdict
from string import punctuation

import pandas as pd
from nltk import sent_tokenize, pos_tag, FreqDist, TweetTokenizer, bigrams
from nltk.corpus import stopwords


class WordContextFinder:

    def __init__(self, sentences, words):
        self.contexts = self.get_word_contexts(sentences, words)

    @staticmethod
    def get_word_contexts(sentences, words):
        context_dict = defaultdict(list)
        for (document, sentence) in sentences:
            intersection = set(words).intersection(set(sentence.split()))
            for word in intersection:
                context_dict[word].append(f'{document}: {sentence}')
        return context_dict


# TODO: error handling on opening files
class DocumentTextExtractor:

    def __init__(self):
        self.word_tokenizer = CustomTokenizer
        self.word_tokens = []
        self.sentence_tokens = []
        self.tagged_normalized_words = []

    def print_interesting_word_table(self, directory_name, number_following):
        self.get_sentence_and_work_tokens(directory_name)
        self.normalize_words()
        interesting_words = self.get_interesting_words(number_following=number_following)
        most_common_10 = WordCounter.most_common_words(self.word_tokens, interesting_words, 10)
        wcf = WordContextFinder(self.sentence_tokens, most_common_10)
        data_tabulate = self.convert_to_csv_form(wcf.contexts)
        self.export_csv(data_tabulate)

    @staticmethod
    def get_string_from_document(document_name):
        with open(document_name) as file:
            return file.read()

    # TODO: directory in directories? what then?
    # TODO: could change this so the sentences are the keys and the documents names are the values
    def get_sentence_and_work_tokens(self, directory_name):
        for file_name in os.listdir(directory_name):
            document_string = self.get_string_from_document(f'{directory_name}/{file_name}')
            tokenizer = CustomTokenizer(document_string)
            self.word_tokens += tokenizer.words
            self.sentence_tokens += [(file_name, sent) for sent in tokenizer.sentences]

    def normalize_words(self):
        word_normalizer = WordNormalizer()
        self.tagged_normalized_words = word_normalizer.normalize_words(self.word_tokens)

    def get_interesting_words(self, number_following=4):
        # create word_following_dict
        following_dict = self.create_word_type_following_dict(self.tagged_normalized_words)
        # get all words that have three or more different followers
        interesting_words = self.find_number_follow_types(following_dict, number_following)
        # strip out the stop words
        return WordNormalizer.remove_from_tokens(interesting_words, stopwords.words('english'))

    @staticmethod
    def create_word_type_following_dict(bigram_tokens):
        words_following_dict = defaultdict(set)
        # TODO: here you could do yield
        for ((word, tag), (following_word, following_tag)) in bigram_tokens:
            words_following_dict[word].add(following_tag)
        return words_following_dict

    @staticmethod
    def find_number_follow_types(words_following_dict, number):
        return [word for word, follow_types in words_following_dict.items() if len(follow_types) >= number]

    @staticmethod
    def convert_to_csv_form(word_context_dict):
        csv_format = []
        for word, sentences in word_context_dict.items():
            for i, sentence in enumerate(sentences):
                csv_format.append([word, sentence] if i == 0 else ['', sentence])
        return csv_format

    @staticmethod
    def export_csv(csv_data):
        df = pd.DataFrame(csv_data)
        df.to_csv('test.csv', index=False, header=['word', 'context'])


# TODO: make these more generic
#  allow it to be called a bit more genrically
class WordNormalizer:

    @staticmethod
    def remove_from_tokens(tokens, remove_list):
        """
        :param tokens: List of String objects
        :param remove_list: List of String objects - tokens to remove - should be in lower case
        :return: List of String objects corresponding to tokens - remove_list
        """
        # TODO: is there anything that can't be stringed?
        # TODO: no sure makes sense to also turn it all to strings here
        return [token for token in tokens if token.lower() not in remove_list]

    @staticmethod
    def word_tagger(tokens):
        return pos_tag(tokens, tagset='universal')

    @staticmethod
    def create_bigrams(tokens):
        return bigrams(tokens)

    def normalize_words(self, word_tokens):
        no_punct_tokens = self.remove_from_tokens(word_tokens, list(punctuation) + ['’', '—'])
        tagged_tokens = self.word_tagger(no_punct_tokens)
        return self.create_bigrams(tagged_tokens)


# TODO: maybe make this follow the same pattern as normal tokenizers
#  - maybe even get rid of it entirely
class CustomTokenizer:

    def __init__(self, text):
        word_tokenizer = TweetTokenizer()
        self.sentences = self.tokenize_into_sentences(text)
        self.words = self.tokenize(text, word_tokenizer)

    @staticmethod
    def tokenize_into_sentences(text):
        return sent_tokenize(text)

    @staticmethod
    def tokenize(text, word_tokenizer):
        return word_tokenizer.tokenize(text)


class WordCounter:

    @staticmethod
    def most_common_words(all_tokens, interesting_words, number):
        freq_dist = FreqDist([t for t in all_tokens if t in interesting_words]).most_common(number)
        return sorted(w for (w, freq) in freq_dist)




if __name__ == '__main__':
    wc = WordCounter()
    print('ALL WORKING')
    # non_stop_words = wc.get_all_non_stop_words('documents')
