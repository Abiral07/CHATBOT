import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')
stemmer = PorterStemmer()
stop_words = list(stopwords.words('english'))
# print (stop_words)

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    #stem each word
    tokenized_sentence= [stem(w) for w in tokenized_sentence]
    #initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
       if word in tokenized_sentence:
          bag[index]=1.0
    return bag

'''sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello" , "how", "I", "you", "bye", "thank", "cool"]
bag   = bag_of_words(sentence,words)
print(bag) #[0. 1. 1. 0. 1. 0. 0. 0.] '''
