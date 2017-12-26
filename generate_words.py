import nltk
from nltk import FreqDist, sent_tokenize, word_tokenize
from nltk.corpus import brown
import ssl
import re
import numpy as np

sentence_corpus = " ".join(brown.words())

def getWords(min_len):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('brown')
    nltk.download('punkt')

    source = FreqDist(i.lower() for i in brown.words())
    source = np.array(source.most_common())[:, :1]

    # the Brown corpus contains duplicates and contains
    # words with weird punctuation and digits
    word_list = np.unique(np.char.lower(source))
    p = np.random.permutation(word_list.shape[0])
    word_list = word_list[p]

    words = [word for word in word_list if len(word) == len(set(word)) and re.search("[^A-Za-z\ ]", word) == None]

    output = [word for word in words if len(word) >= min_len and len(word) <= 26 and word[-1:] != 's']
    return output

def getSentence(word):
    global sentence_corpus

    sentence = [i for i in sent_tokenize(sentence_corpus) if word in word_tokenize(i)]
    sentence.sort(key=len)
    if len(sentence) > 0:
        return sentence[0]
    else:
        return ''


def write_to_csv(file, word, sentence):
    with open('corpus/' + file + '.csv', 'a') as f:
        f.write(word + ',"' + sentence + '"\n')

words = getWords(3)
for word in words:
    sentence = getSentence(word)
    if sentence != '':
        print(word)
        write_to_csv('words2', word, sentence)
