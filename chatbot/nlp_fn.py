import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()

#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.metrics.pairwise import cosine_similarity

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# lemmer = WordNetLemmatizer()
# def LemTokens(tokens):
#     return [lemmer.lemmatize(token) for token in tokens]

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    #stem each word
    tokenized_sentence= [stem(w) for w in tokenized_sentence]
    #initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, w in enumerate(all_words):
       if w in tokenized_sentence:
          bag[index]=1.0
    return bag

'''sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello" , "how", "I", "you", "bye", "thank", "cool"]
bag   = bag_of_words(sentence,words)
print(bag)'''
