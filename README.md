# word_counter_challenge
word_counter_challenge


Interesting words: 
I have taken this to mean ambiguous words. These i have defined as words with a high number of different types (nouns, verbs etc.) of following words.

If a word is always followed a noun it is not interesting.  If it is followed in one case by a noun in another by an adverb in another by a verb etc. it is interesting. 

I don’t know if this is linguistically significant but it was interesting to me.  

The program outputs a table of the most common of these words and their sentence contexts as a csv.  


Set up instructions:
# clone repo
git clone https://github.com/nedlloyd/word_counter_challenge.git
# create virtual env
python3 -m venv /path/to/new/virtual/environment
# start virtual env
. path/to/env/bin/activate
# install requirements
pip install -r path/to/requirements/file/in/project
# start_shell
ipython
# download interesting words are csv (in shell). 
from interesting_words import *
# 6 is the number of following word types. 10 means the most common 10 interesting words.
dtx = DocumentTextExtractor(‘documents’, 6, 10)
dtx.export_interesting_words_as_csv()

