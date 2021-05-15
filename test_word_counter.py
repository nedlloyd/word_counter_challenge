from string import punctuation
from unittest import main, TestCase

from nltk import TweetTokenizer

from interesting_words import WordCounter, WordNormalizer, WordTokenizer, WordContextFinder, DocumentTextExtractor


class TestWordTokenizer(TestCase):

    def test_keeps_words_with_apostrophes(self):
        text = "Just one line - of text doesn't matter."
        self.assertEqual(
            WordTokenizer(text, TweetTokenizer).words,
            ["Just", "one", "line", "-", "of", "text", "doesn't", "matter", "."]
        )

    def test_tokenize_words(self):
        text = "Just one line of text."
        self.assertEqual(
            WordTokenizer(text, TweetTokenizer).words,
            ["Just", "one", "line", "of", "text", "."]
        )

    def test_tokenize_single_sentence(self):
        text = "Just one line of text."
        self.assertEqual(
            WordTokenizer(text, TweetTokenizer).sentences,
            ["Just one line of text."]
        )

    def test_tokenize_sentences_test_text_1(self):
        with open('tests/test_directory/test_text_1.txt') as file:
            text = file.read()
            sentence_tokens = [
                "he doesn't even know.",
                "and a list of movies: '2001', 'Computer Chess', 'The Red Shoes'.",
                "The schools' movie is nice.",
                'Too nice even.',
                'way-too-nice.'
            ]
            self.assertEqual(
                WordTokenizer(text, TweetTokenizer).sentences,
                sentence_tokens
            )

    def test_tokenize_sentences_test_text_2(self):
        with open('tests/test_directory/test_text_2.txt') as file:
            text = file.read()
            sentence_tokens = [
                "quite simply the second document.", "and that, for now, is all you're getting - YES!"
            ]
            self.assertEqual(
                WordTokenizer(text, TweetTokenizer).sentences,
                sentence_tokens
            )


class TestWordNormalisation(TestCase):

    def test_remove_punctuation(self):
        tokens = ["Just", "one", "line", "-", "of", "text", "doesn't", "matter", "."]
        remove_list = list(punctuation)
        self.assertEqual(
            WordNormalizer.remove_from_tokens(tokens, remove_list),
            ["Just", "one", "line", "of", "text", "doesn't", "matter"]
        )

    def test_remove_stop_words_and_punctuation(self):
        tokens = ["Just", "one", "line", "-", "of", "text", "doesn't", "matter", "."]
        remove_list = ['of', "doesn't"] + list(punctuation)
        self.assertEqual(
            WordNormalizer.remove_from_tokens(tokens, remove_list),
            ["Just", "one", "line", "text", "matter"]
        )

    def test_remove_tokens_upper_case(self):
        tokens = ["Just", "one", "line", "-", "Of", "text", "Doesn't", "matter", "."]
        remove_list = ['of', "doesn't"] + list(punctuation)
        self.assertEqual(
            WordNormalizer.remove_from_tokens(tokens, remove_list),
            ["Just", "one", "line", "text", "matter"]
        )

    def test_tag_word_types(self):
        tokens = ['car']
        self.assertEqual(
            WordNormalizer.word_tagger(tokens),
            [('car', 'NOUN')]
        )

    def test_tag_word_types_sentence(self):
        tokens = ["Just", "one", "line", "of", "text", "is", "all"]
        tagged_words = [
            ('Just', 'ADV'),
            ('one', 'NUM'),
            ('line', 'NOUN'),
            ('of', 'ADP'),
            ('text', 'NOUN'),
            ("is", 'VERB'),
            ('all', 'DET')
        ]
        self.assertEqual(
            WordNormalizer.word_tagger(tokens),
            tagged_words
        )

    def test_create_bigrams(self):
        tokens = ['just', 'three', 'words']
        token_bigrams = [("just", "three"), ("three", "words")]
        self.assertEqual(
            list(WordNormalizer.create_bigrams(tokens)),
            token_bigrams
        )

    def test_create_bigrams_sentence(self):
        tokens = ["Just", "one", "line", "of", "text", "is", "all"]
        token_bigrams = [
            ("Just", "one"), ("one", "line"), ("line", "of"), ("of", "text"), ("text", "is"), ("is", "all")
        ]
        self.assertEqual(
            list(WordNormalizer.create_bigrams(tokens)),
            token_bigrams
        )

    def test_create_bigrams_word_type_pairs(self):
        tokens = [('just', 'ADV'), ('three', 'NUM'), ('words', 'NOUN')]
        token_bigrams = [(('just', 'ADV'), ('three', 'NUM')), (('three', 'NUM'), ('words', 'NOUN'))]
        self.assertEqual(
            list(WordNormalizer.create_bigrams(tokens)),
            token_bigrams
        )

    def test_create_word_type_following_dict(self):
        tokens = [(('just', 'ADV'), ('three', 'NUM')), (('three', 'NUM'), ('words', 'NOUN'))]
        following_dict = {'just': {'NUM'}, 'three': {'NOUN'}}
        self.assertEqual(
            WordNormalizer.create_word_type_following_dict(tokens),
            following_dict
        )

    def test_create_word_type_following_dict_multiple_following(self):
        """
        Test when a word is followed by more than one different kinds of words
        """
        tokens = [
            (('just', 'ADV'), ('three', 'NUM')),
            (('three', 'NUM'), ('words', 'NOUN')),
            (('words', 'NOUN'), ('just', 'ADV')),
            (('just', 'ADV'), ('say', 'VERB'))
        ]
        following_dict = {'just': {'NUM', 'VERB'}, 'three': {'NOUN'}, 'words': {'ADV'}}
        self.assertEqual(
            WordNormalizer.create_word_type_following_dict(tokens),
            following_dict
        )

    def test_find_words_with_two_different_following_word_types(self):
        following_dict = {'just': {'NUM', 'VERB'}, 'three': {'NOUN'}, 'words': {'ADV'}}
        self.assertEqual(
            WordNormalizer.find_number_follow_types(following_dict, 2),
            ['just']
        )

    def test_find_words_with_three_different_following_word_types(self):
        following_dict = {
            'just': {'NUM', 'VERB', 'NOUN'},
            'three': {'NOUN'},
            'words': {'ADV', 'VERB'},
            'follow': {'PRON', 'NOUN', 'ADJ', 'DET'}
        }
        self.assertEqual(
            WordNormalizer.find_number_follow_types(following_dict, 3),
            ['just', 'follow']
        )



class TestWordCounter(TestCase):

    def test_most_common_word(self):
        tokens = ["Just", "one", "line", "text", "matter", 'matter']
        self.assertEqual(WordCounter.most_common_words(tokens, 1), [('matter', 2)])

    def test_most_common_word_multiple(self):
        tokens = [
            "Just", "one", "line", "text", "matter", 'matter', 'words', 'words', 'matter',
            'this', 'and', 'that', 'words', 'more', 'keep', 'going', 'text', 'line', 'line'
        ]
        self.assertEqual(
            WordCounter.most_common_words(tokens, 3),
            [('line', 3), ('matter', 3), ('words', 3),]
        )




class TestWordContextFinder(TestCase):

    def test_gets_context_of_word(self):
        sentences = {'text_1': ['let us go then', 'you and i', 'as the evening is spread out against the sky']}
        words = ['evening']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(wcf.contexts, {'evening': {'as the evening is spread out against the sky'}})

    def test_multiple_sentences(self):
        sentences = [
            'let us go then', 'you and i', 'as the evening is spread out against the sky',
            'like a patient etherised upon a table', 'let us go', 'through half deserted streets'
        ]
        words = ['us']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            wcf.contexts,
            {'us': ['let us go then', 'let us go']}
        )

    def test_duplicate_sentences(self):
        sentences = [
            'let us go', 'you and i', 'as the evening is spread out against the sky',
            'like a patient etherised upon a table', 'let us go', 'through half deserted streets'
        ]
        words = ['us']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            wcf.contexts,
            {'us': ['let us go', 'let us go']}
        )

    def test_multiple_words(self):
        sentences = [
            'let us go then', 'you and i', 'as the evening is spread out against the sky',
            'like a patient etherised upon a table', 'let us go', 'through half deserted streets'
        ]
        words = ['us', 'evening']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            wcf.contexts,
            {
                'us': ['let us go then', 'let us go'],
                'evening': ['as the evening is spread out against the sky']
            }
        )


class TestDocumentTextExtractor(TestCase):
    maxDiff = None

    def test_single_document_get_text(self):
        # TODO this could be accomplished by writting temporary file
        document_string = DocumentTextExtractor.get_string_from_document('tests/test_directory/test_text_2.txt')
        self.assertEqual(
            document_string, "quite simply the second document.  and that, for now, is all you're getting - YES!"
        )

    # def test_single_document_no_file(self):
    #     # TODO this could be accomplished by writting temporary file
    #     document_string = DocumentTextExtractor.get_string_from_document('tests/test_directory/no_file.txt')

    def test_multiple_documents_get_text(self):
        document_string = DocumentTextExtractor().get_string_from_directory('tests/test_directory')
        # print(f'1: {document_string}')
        # print()
        # with open('tests/test_all_text.txt') as file:
        #     # print(f'2: {file.read()}')
        #     file_string = file.read()
        self.assertEqual(
            document_string,
            "quite simply the second document.  "
            "and that, for now, is all you're getting - YES! he doesn't even know. "
            "and a list of movies: '2001', 'Computer Chess', 'The Red Shoes'. "
            "The schools' movie is nice.  Too nice even. way-too-nice."
        )


    # def test_normalize_text(self):
    #     with open('tests/test_directory/test_text_1.txt') as file:
    #         text = file.read()
    #         wn = WordNormalisation()
    #         tokenized_sentences = [
    #             "he doesn't even know.",
    #             "and a list of movies: '2001', 'Computer Chess', 'The Red Shoes'.",
    #             "The schools' movie is nice.",
    #             'Too nice even.',
    #             'way-too-nice.'
    #         ]
    #         self.assertEqual(
    #             wn.normalize_text(text),
    #             tokenized_sentences
    #         )




if __name__ == '__main__':
    main()
