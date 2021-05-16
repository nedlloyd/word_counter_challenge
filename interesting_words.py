import os
from collections import defaultdict
import ssl
from string import punctuation

import pandas as pd
from nltk import download, sent_tokenize, pos_tag, FreqDist, TweetTokenizer, bigrams
from nltk.corpus import stopwords


def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    download('punkt')
    download('stopwords')
    download('universal_tagset')
    download('averaged_perceptron_tagger')


class DocumentTextExtractor:

    def __init__(self, directory_name, number_following, most_common_number):
        """
        Reads all documents in specified directory.
        Finds words with >= number_following types of following words.
            i.e. If a word is only followed by verbs and number_following > 1 then it will not be returned
                If a word is followed by nouns and very and number_following = 2 then it will be returned
        Option to export result to csv.
        :param directory_name: String - name of directory where files to be read are
        :param number_following: Int - Int - minimum number required of following word kinds (Nouns, verbs etc.)
            to meet criteria.
        :param most_common_number: Int - number of results returned.
        """
        self._directory_name = directory_name
        self._number_following = number_following
        self._most_common_number = most_common_number
        self._word_tokens = []
        self._sentence_tokens = []
        self.tokenizer = TweetTokenizer

    def export_interesting_words_as_csv(self):
        interesting_words = self.get_interesting_words(number_following=self._number_following)
        most_common_10 = WordCounter.most_common_words(self._word_tokens, interesting_words, self._most_common_number)
        contexts = WordContextFinder.get_word_contexts(self._sentence_tokens, most_common_10, self.tokenizer)
        data_tabulate = self._convert_to_csv_form(contexts)
        self._export_csv(data_tabulate)

    def get_interesting_words(self, number_following=4):
        """
        Finds words with >= number_following types (Nouns, Verbs etc.) of following words.
        :param number_following: Int - minimum number required of following word kinds.
        :return: List of Strings representing interesting words.
        """
        self._extract_sentence_and_work_tokens(self._directory_name)
        tagged_normalized_words = WordNormalizer().normalize_words(self._word_tokens)
        following_dict = self._create_word_type_following_dict(tagged_normalized_words)
        interesting_words = self._find_number_follow_types(following_dict, number_following)
        return WordNormalizer.remove_from_tokens(interesting_words, stopwords.words('english'))

    @staticmethod
    def _get_string_from_document(document_path):
        """
        Gets document as String.
        :param document_path: String representing path to document.
        :return: String - representing file content.
        """
        with open(document_path) as file:
            return file.read()

    def _extract_sentence_and_work_tokens(self, directory_name):
        """
        Sets _word_tokens variable as list of word Strings in lower case.
        Sets _sentence_tokens variable as list of sentence Strings.
        :param directory_name: String representing directory name.
        """
        for file_name in os.listdir(directory_name):
            document_string = self._get_string_from_document(f'{directory_name}/{file_name}')
            self._word_tokens += [w.lower() for w in self.tokenizer().tokenize(document_string)]
            self._sentence_tokens += [(file_name, sent) for sent in sent_tokenize(document_string)]

    @staticmethod
    def _create_word_type_following_dict(bigram_tokens):
        """
        :param bigram_tokens: List of bigrams of words tagged by word type
            e.g. [((hammer, NOUN), (hard, ADJECTIVE))]
        :return: Dict with of key of words and values of a set of following type words
            e.g. {'phone': {'NOUN', 'VERB'}}
        """
        words_following_dict = defaultdict(set)
        for ((word, tag), (following_word, following_tag)) in bigram_tokens:
            words_following_dict[word].add(following_tag)
        return words_following_dict

    @staticmethod
    def _find_number_follow_types(words_following_dict, number):
        """
        :param words_following_dict: Dict with of key of words and values of a set of following type words
            e.g. {'phone': {'NOUN', 'VERB'}}
        :param number: Int - minimum number of follow types
        :return: list  - of word Strings corresponding to words with >= number of following type words
        """
        return [word for word, follow_types in words_following_dict.items() if len(follow_types) >= number]

    @staticmethod
    def _convert_to_csv_form(word_context_dict):
        """
        :param word_context_dict: dict of word contexts in form: {word - String: ['sentence contexts']}
        :return: list of word contexts in form:
            [['word', 'sentence context 1'], ['', 'sentence context 2'], , ['word 2', 'sentence context 1']]
        """
        csv_format = []
        for word, sentences in word_context_dict.items():
            for i, sentence in enumerate(sentences):
                csv_format.append([word, sentence] if i == 0 else ['', sentence])
        return csv_format

    @staticmethod
    def _export_csv(csv_data):
        """
        export data to csv
        """
        df = pd.DataFrame(csv_data)
        df.to_csv('test.csv', index=False, header=['word', 'context'])


class WordContextFinder:

    @staticmethod
    def get_word_contexts(sentences, words, word_tokenizer):
        """
        Get sentence context for each word in sentence
        :param sentences: list of tuples of form (document_name, sentence)
        :param words: list of words Strings
        :param word_tokenizer: Tokenizer class - tokenizer class used to tokenize sentence
        :return: dict of form {word - String: ['sentence contexts']}
        """
        context_dict = defaultdict(list)
        lower_case_words = [w.lower() for w in words]
        for (document, sentence) in sentences:
            intersection = set(lower_case_words).intersection(
                set([w.lower() for w in word_tokenizer().tokenize(sentence)])
            )
            for word in intersection:
                context_dict[word].append(f'{document}: {sentence}')
        return context_dict


class WordNormalizer:

    @staticmethod
    def remove_from_tokens(tokens, remove_list):
        """
        :param tokens: List of String objects
        :param remove_list: List of String objects - tokens to remove - should be in lower case
        :return: List of String objects corresponding to tokens - remove_list
        """
        return [token for token in tokens if token.lower() not in remove_list]

    @staticmethod
    def word_tagger(tokens):
        """
        :param tokens: List words Strings
        :return: List of tuples in form (word, word type)
        """
        return pos_tag(tokens, tagset='universal')

    def normalize_words(self, word_tokens):
        """
        - removes punctuation from words
        - tags words with word types
        - create bigram of (word, word_type) tuples
        :param word_tokens: List words Strings
        :return: list bigrams of (word, word_type) e.g. [((hammer, NOUN), (hard, ADJECTIVE))]
        """
        no_punct_tokens = self.remove_from_tokens(word_tokens, list(punctuation) + ['’', '—'])
        tagged_tokens = self.word_tagger(no_punct_tokens)
        return bigrams(tagged_tokens)


class WordCounter:

    @staticmethod
    def most_common_words(all_tokens, interesting_words, number):
        """
        :param all_tokens: list of word String objects representing all tokens
        :param interesting_words: list of word String objects words to be counted
        :param number: int the number of most common words to be returned
        :return: list - most common words ordered alphabetically
        """
        freq_dist = FreqDist([t for t in all_tokens if t in interesting_words]).most_common(number)
        return sorted(w for (w, freq) in freq_dist)


if __name__ == '__main__':
    print('ALL WORKING')
