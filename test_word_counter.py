from string import punctuation
from unittest import main, TestCase

from nltk import TweetTokenizer

from interesting_words import WordCounter, WordNormalizer, WordContextFinder, DocumentTextExtractor


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

    def test_get_context_of_word(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky')
        ]
        words = ['evening']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(dict(contexts), {'evening': ['text_1: as the evening is spread out against the sky']})

    def test_get_context_of_word_different_case(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky')
        ]
        words = ['Evening']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(dict(contexts), {'evening': ['text_1: as the evening is spread out against the sky']})

    def test_get_context_of_word_ending_with_punctuation(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky.')
        ]
        words = ['sky']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(dict(contexts), {'sky': ['text_1: as the evening is spread out against the sky.']})

    def test_get_context_of_word_with_comma(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky, seas, beasts and trees.')
        ]
        words = ['sky']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(
            dict(contexts), {'sky': ['text_1: as the evening is spread out against the sky, seas, beasts and trees.']}
        )

    def test_get_context_sentence_different_case(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the Evening is spread out against the sky')
        ]
        words = ['evening']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(dict(contexts), {'evening': ['text_1: as the Evening is spread out against the sky']})

    def test_multiple_sentences(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_1', 'like a patient etherised upon a table'), ('text_1', 'let us go'),
            ('text_1', 'through half deserted streets')
        ]
        words = ['us']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(dict(contexts), {'us': ['text_1: let us go then', 'text_1: let us go']})

    def test_duplicate_sentences(self):
        sentences = [
            ('text_1', 'let us go'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_1', 'like a patient etherised upon a table'), ('text_1', 'let us go'),
            ('text_1', 'through half deserted streets')
        ]
        words = ['us']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(
            dict(contexts),
            {'us': ['text_1: let us go', 'text_1: let us go']}
        )

    def test_multiple_documents(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_2', 'like a patient etherised upon a table'), ('text_2', 'let us go'),
            ('text_2', 'through half deserted streets')
        ]
        words = ['us']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(dict(contexts), {'us': ['text_1: let us go then', 'text_2: let us go']})

    def test_multiple_documents_same_sentence(self):
        sentences = [
            ('text_1', 'let us go'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_2', 'like a patient etherised upon a table'), ('text_2', 'let us go'),
            ('text_2', 'through half deserted streets')
        ]
        words = ['us']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(
            dict(contexts),
            {'us': ['text_1: let us go', 'text_2: let us go']}
        )

    def test_multiple_words(self):
        sentences = [
            ('text_1', 'let us go then'), ('text_1', 'you and i'),
            ('text_1', 'as the evening is spread out against the sky'),
            ('text_1', 'like a patient etherised upon a table'),
            ('text_1', 'let us go'), ('text_1', 'through half deserted streets')
        ]
        words = ['us', 'evening']
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(
            dict(contexts),
            {
                'us': ['text_1: let us go then', 'text_1: let us go'],
                'evening': ['text_1: as the evening is spread out against the sky']
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
        contexts = WordContextFinder.get_word_contexts(sentences, words, TweetTokenizer)
        self.assertEqual(
            dict(contexts),
            {
                'us': ['text_1: let us go then', 'text_2: let us go'],
                'evening': ['text_1: as the evening is spread out against the sky']
            }
        )


class TestDocumentTextExtractor(TestCase):

    def test_single_document_get_text(self):
        document_string = DocumentTextExtractor._get_string_from_document('tests_files/test_directory/test_text_2.txt')
        self.assertEqual(
            document_string, "quite simply the second document.  and that, for now, is all you're getting - YES!"
        )

    def test_multiple_documents_get_word_tokens(self):
        text_extractor = DocumentTextExtractor('test_directory', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_directory')
        self.assertEqual(len(text_extractor._word_tokens), 58)

    def test_document_get_word_tokens(self):
        text_extractor = DocumentTextExtractor('test_directory', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_directory_single_file')
        self.assertEqual(len(text_extractor._word_tokens), 7)

    def test_splits_sentences_by_document(self):
        text_extractor = DocumentTextExtractor('test_directory', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_directory_single_file')
        self.assertEqual(text_extractor._sentence_tokens, [('single_file.txt', 'one, two, three and four')])

    def test_gets_num_of_sentences(self):
        text_extractor = DocumentTextExtractor('test_directory', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_directory_single_file')
        self.assertEqual(len(text_extractor._sentence_tokens), 1)

    def test_splits_sentences_by_multiple_docs(self):
        text_extractor = DocumentTextExtractor('test_directory', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_directory')
        self.assertEqual(text_extractor._sentence_tokens[0], ('test_text_2.txt', 'quite simply the second document.'))
        self.assertEqual(text_extractor._sentence_tokens[6], ('test_text_1.txt', 'way-too-nice.'))

    def test_gets_num_of_sentences_multiple_docs(self):
        text_extractor = DocumentTextExtractor('test_directory', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_directory')
        self.assertEqual(len(text_extractor._sentence_tokens), 7)

    def test_create_word_type_following_dict(self):
        tokens = [(('just', 'ADV'), ('three', 'NUM')), (('three', 'NUM'), ('words', 'NOUN'))]
        following_dict = {'just': {'NUM'}, 'three': {'NOUN'}}
        self.assertEqual(
            DocumentTextExtractor('test_directory', 5, 10)._create_word_type_following_dict(tokens),
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
            DocumentTextExtractor('test_directory', 5, 10)._create_word_type_following_dict(tokens),
            following_dict
        )

    def test_find_words_with_two_different_following_word_types(self):
        following_dict = {'just': {'NUM', 'VERB'}, 'three': {'NOUN'}, 'words': {'ADV'}}
        self.assertEqual(
            DocumentTextExtractor('test_directory', 5, 10)._find_number_follow_types(following_dict, 2),
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
            DocumentTextExtractor('test_directory', 5, 10)._find_number_follow_types(following_dict, 3),
            ['just', 'follow']
        )

    def test_get_interesting_words(self):
        text_extractor = DocumentTextExtractor('tests_files/test_extractor', 5, 10)
        text_extractor._extract_sentence_and_work_tokens('tests_files/test_extractor')
        self.assertEqual(
            text_extractor.get_interesting_words(number_following=2), ['take', 'nothing', 'time', 'whenever', 'get']
        )

    def test_convert_to_csv_format(self):
        context_dict = {'us': ['text_1: let us go then', 'text_1: let us go']}
        csv_format = DocumentTextExtractor._convert_to_csv_form(context_dict)
        self.assertEqual(csv_format, [['us', 'text_1: let us go then'], ['', 'text_1: let us go']])

    def test_convert_to_csv_format_multi_word(self):
        context_dict = {
            'us': ['text_1: let us go then', 'text_1: let us go'],
            'hotter': ['text_1: i get hotter.', 'text_1: Did you say hotter?'],
        }
        csv_format = DocumentTextExtractor._convert_to_csv_form(context_dict)
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
