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

    def test_tokenize_sentences(self):
        with open('tests/test_documents/test_text_1.txt') as file:
            text = file.read()
            tokenized_sentences = [
                "he doesn't even know.",
                "and a list of movies: '2001', 'Computer Chess', 'The Red Shoes'.",
                "The schools' movie is nice.",
                'Too nice even.',
                'way-too-nice.'
            ]
            self.assertEqual(
                WordTokenizer(text, TweetTokenizer).sentences,
                tokenized_sentences
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
        sentences = ['let us go then', 'you and i', 'as the evening is spread out against the sky']
        words = ['evening']
        wcf = WordContextFinder(sentences, words)
        self.assertEqual(wcf.contexts, {'evening': ['as the evening is spread out against the sky']})

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

    def test_single_document_get_text(self):
        # TODO this could be accomplished by writting temporary file
        document_string = DocumentTextExtractor.get_string_from_document('tests/test_documents/test_text_2.txt')
        self.assertEqual(
            document_string, "quite simply the second document.  and that, for now, is all you're getting - YES!"
        )

    # def test_single_document_no_file(self):
    #     # TODO this could be accomplished by writting temporary file
    #     document_string = DocumentTextExtractor.get_string_from_document('tests/test_documents/no_file.txt')

    def test_multiple_documents_get_text(self):
        document_string = DocumentTextExtractor().get_string_from_directory('test/test_documents')
        self.assertEqual(
            document_string,
            "he doesn't even know. and a list of movies: '2001', 'Computer Chess', 'The Red Shoes'. "
            "The schools' movie is nice.  Too nice even. way-too-nice.  "
            "quite simply the second document.  and that, for now, is all you're getting - YES!"
        )


    # def test_normalize_text(self):
    #     with open('tests/test_documents/test_text_1.txt') as file:
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
