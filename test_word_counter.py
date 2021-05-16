from string import punctuation
from unittest import main, TestCase

from nltk import TweetTokenizer

from interesting_words import WordCounter, WordNormalizer, CustomTokenizer, WordContextFinder, DocumentTextExtractor


class TestCustomTokenizer(TestCase):

    def test_keeps_words_with_apostrophes(self):
        text = "Just one line - of text doesn't matter."
        self.assertEqual(
            CustomTokenizer(text).words,
            ["Just", "one", "line", "-", "of", "text", "doesn't", "matter", "."]
        )

    def test_tokenize_words(self):
        text = "Just one line of text."
        self.assertEqual(
            CustomTokenizer(text).words,
            ["Just", "one", "line", "of", "text", "."]
        )

    def test_tokenize_single_sentence(self):
        text = "Just one line of text."
        self.assertEqual(
            CustomTokenizer(text).sentences,
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
                CustomTokenizer(text).sentences,
                sentence_tokens
            )

    def test_tokenize_sentences_test_text_2(self):
        with open('tests/test_directory/test_text_2.txt') as file:
            text = file.read()
            sentence_tokens = [
                "quite simply the second document.", "and that, for now, is all you're getting - YES!"
            ]
            self.assertEqual(
                CustomTokenizer(text).sentences,
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

    def test_remove_tokens_contains_digits(self):
        tokens = ["Just", 1, "line", "-", "Of", "text", "Doesn't", "matter", "."]
        remove_list = ['of', "doesn't"] + list(punctuation)
        self.assertEqual(
            WordNormalizer.remove_from_tokens(tokens, remove_list),
            ["Just", 1, "line", "text", "matter"]
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


class TestWordCounter(TestCase):

    def test_most_common_word(self):
        tokens = ["Just", "one", "line", "text", "matter", 'matter']
        self.assertEqual(WordCounter.most_common_words(tokens, tokens, 1), ['matter'])

    def test_most_common_word_multiple(self):
        tokens = [
            "Just", "one", "line", "text", "matter", 'matter', 'words', 'words', 'matter',
            'this', 'and', 'that', 'words', 'more', 'keep', 'going', 'text', 'line', 'line'
        ]
        self.assertEqual(
            WordCounter.most_common_words(tokens, tokens, 3),
            ['line', 'matter', 'words']
        )

    def test_most_common_of_set(self):
        tokens = [
            "Just", "one", "line", "text", "matter", 'matter', 'words', 'words', 'matter',
            'this', 'and', 'that', 'words', 'more', 'keep', 'going', 'text', 'line', 'line',
        ]
        self.assertEqual(
            WordCounter.most_common_words(tokens, ['words', 'matter'], 3),
            ['matter', 'words']
        )

    def test_filter_out_most_common(self):
        tokens = [
            "Just", "one", "line", "text", "matter", 'matter', 'words', 'words', 'matter',
            'this', 'and', 'that', 'words', 'more', 'keep', 'going', 'text', 'line', 'line',
            'words', 'words', 'matter', 'matter', 'just', 'keep', 'keep'
        ]
        self.assertEqual(
            WordCounter.most_common_words(tokens, ['just'], 2),
            ['just']
        )


class TestWordContextFinder(TestCase):

    def test_gets_context_of_word(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky')
        ]
        words = ['evening']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(dict(wcf.contexts), {'evening': 'text_1: as the evening is spread out against the sky\n\n'})

    def test_multiple_sentences(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_1', 'like a patient etherised upon a table'), ('text_1', 'let us go'),
            ('text_1', 'through half deserted streets')
        ]
        words = ['us']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(dict(wcf.contexts), {'us': 'text_1: let us go then\n\ntext_1: let us go\n\n'})

    def test_duplicate_sentences(self):
        sentences = [
            ('text_1', 'let us go'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_1', 'like a patient etherised upon a table'), ('text_1', 'let us go'),
            ('text_1', 'through half deserted streets')
        ]
        words = ['us']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            dict(wcf.contexts),
            {'us': 'text_1: let us go\n\ntext_1: let us go\n\n'}
        )

    def test_multiple_documents(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_2', 'like a patient etherised upon a table'), ('text_2', 'let us go'),
            ('text_2', 'through half deserted streets')
        ]
        words = ['us']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(dict(wcf.contexts), {'us': 'text_1: let us go then\n\ntext_2: let us go\n\n'})

    def test_multiple_documents_same_sentence(self):
        sentences = [
            ('text_1', 'let us go'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_2', 'like a patient etherised upon a table'), ('text_2', 'let us go'),
            ('text_2', 'through half deserted streets')
        ]
        words = ['us']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            dict(wcf.contexts),
            {'us': 'text_1: let us go\n\ntext_2: let us go\n\n'}
        )

    def test_multiple_words(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_1', 'like a patient etherised upon a table'),
            ('text_1', 'let us go'), ('text_1', 'through half deserted streets')
        ]
        words = ['us', 'evening']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            dict(wcf.contexts),
            {
                'us': 'text_1: let us go then\n\ntext_1: let us go\n\n',
                'evening': 'text_1: as the evening is spread out against the sky\n\n'
            }
        )

    def test_multiple_words_diff_docs(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_2', 'like a patient etherised upon a table'), ('text_2', 'let us go'),
            ('text_2', 'through half deserted streets'),
        ]
        words = ['us', 'evening']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(
            dict(wcf.contexts),
            {
                'us': 'text_1: let us go then\n\ntext_2: let us go\n\n',
                'evening': 'text_1: as the evening is spread out against the sky\n\n'
            }
        )




class TestDocumentTextExtractor(TestCase):

    def test_single_document_get_text(self):
        # TODO this could be accomplished by writting temporary file
        document_string = DocumentTextExtractor.get_string_from_document('tests/test_directory/test_text_2.txt')
        self.assertEqual(
            document_string, "quite simply the second document.  and that, for now, is all you're getting - YES!"
        )

    # def test_single_document_no_file(self):
    #     # TODO this could be accomplished by writting temporary file
    #     document_string = DocumentTextExtractor.get_string_from_document('tests/test_directory/no_file.txt')

    def test_multiple_documents_get_word_tokens(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_directory')
        self.assertEqual(len(text_extractor.word_tokens), 58)

    def test_document_get_word_tokens(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_directory_single_file')
        self.assertEqual(len(text_extractor.word_tokens), 7)

    def test_splits_sentences_by_document(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_directory_single_file')
        self.assertEqual(text_extractor.sentence_tokens, [('single_file.txt', 'one, two, three and four')])

    def test_gets_num_of_sentences(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_directory_single_file')
        self.assertEqual(len(text_extractor.sentence_tokens), 1)

    def test_splits_sentences_by_multiple_docs(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_directory')
        self.assertEqual(text_extractor.sentence_tokens[0], ('test_text_2.txt', 'quite simply the second document.'))
        self.assertEqual(text_extractor.sentence_tokens[6], ('test_text_1.txt', 'way-too-nice.'))

    def test_gets_num_of_sentences_multiple_docs(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_directory')
        self.assertEqual(len(text_extractor.sentence_tokens), 7)

    def test_create_word_type_following_dict(self):
        tokens = [(('just', 'ADV'), ('three', 'NUM')), (('three', 'NUM'), ('words', 'NOUN'))]
        following_dict = {'just': {'NUM'}, 'three': {'NOUN'}}
        self.assertEqual(
            DocumentTextExtractor().create_word_type_following_dict(tokens),
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
            DocumentTextExtractor().create_word_type_following_dict(tokens),
            following_dict
        )

    def test_find_words_with_two_different_following_word_types(self):
        following_dict = {'just': {'NUM', 'VERB'}, 'three': {'NOUN'}, 'words': {'ADV'}}
        self.assertEqual(
            DocumentTextExtractor().find_number_follow_types(following_dict, 2),
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
            DocumentTextExtractor().find_number_follow_types(following_dict, 3),
            ['just', 'follow']
        )

    def test_get_interesting_words(self):
        text_extractor = DocumentTextExtractor()
        text_extractor.get_sentence_and_work_tokens('tests/test_extractor')
        text_extractor.normalize_words()
        self.assertEqual(
            text_extractor.get_interesting_words(number_following=2), ['take', 'nothing', 'time', 'get']
        )

    def test_convert_to_tabulate_format(self):
        words_dict = {
            'us': 'text_1: let us go then\n\ntext_2: let us go\n\n',
            'evening': 'text_1: as the evening is spread out against the sky\n\n'
        }
        self.assertEqual(
            DocumentTextExtractor.convert_to_tabulate_form(words_dict),
            [
                ['us', 'text_1: let us go then\n\ntext_2: let us go\n\n'],
                ['evening', 'text_1: as the evening is spread out against the sky\n\n']
            ]
        )

    def test_convert_to_csv_format(self):
        context_dict = {'us': ['text_1: let us go then', 'text_1: let us go']}
        csv_format = DocumentTextExtractor.convert_to_csv_form(context_dict)
        self.assertEqual(csv_format, [['us', 'text_1: let us go then'], ['', 'text_1: let us go']])

    def test_convert_to_csv_format_multi_word(self):
        context_dict = {
            'us': ['text_1: let us go then', 'text_1: let us go'],
            'hotter': ['text_1: i get hotter.', 'text_1: Did you say hotter?'],
        }
        csv_format = DocumentTextExtractor.convert_to_csv_form(context_dict)
        self.assertEqual(
            csv_format,
            [
                ['us', 'text_1: let us go then'],
                ['', 'text_1: let us go'],
                ['hotter', 'text_1: i get hotter.'],
                ['', 'text_1: Did you say hotter?']
            ]
        )



if __name__ == '__main__':
    main()
