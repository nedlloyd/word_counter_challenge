# Interesting Word Counter


##Interesting words: 
I have taken this to mean ambiguous words. These i have defined as words with a high number of different types (nouns, verbs etc.) of following words.

If a word is always followed a noun it is not interesting.  If it is followed in one case by a noun in another by an adverb in another by a verb etc. it is interesting. 

I don’t know if this is linguistically significant but it was interesting to me.  

The program outputs a table of the most common of these words and their sentence contexts as a csv.  


## Set up instructions:
Using python 3.9
### clone repo
git clone https://github.com/nedlloyd/word_counter_challenge.git
### create virtual env
python3 -m venv /path/to/new/virtual/environment
### start virtual env
. path/to/env/bin/activate
### install requirements - from inside project directory
pip install -r requirements.txt
### start shell
ipython
#### 'documents' is where the directory containing documents is
#### 6 is the number of following word types. 
#### 10 means the most common 10 interesting words.
from interesting_words import DocumentTextExtractor, download_nltk_data  
download_nltk_data()  
extractor = DocumentTextExtractor('documents', 6, 10)  
extractor.export_interesting_words_as_csv()  

