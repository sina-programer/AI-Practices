import numpy as np
import pandas as pd


def bag_of_words(texts: list[str]):
    """ For each word, save the number of word usage in each sentence (word-count) """

    words = []
    for sentence in texts:
        words.extend(sentence.split())

    matrix = pd.DataFrame(columns=words, index=range(len(texts)))
    for i, sentence in enumerate(texts):
        sentence_words = sentence.split()

        for word in words:
            matrix.loc[i, word] = sentence_words.count(word)

    return matrix


def term_frequency(texts: list[str]):
    """ For each word, save the density of word (word-usage / all-words) """

    words = []
    for sentence in texts:
        words.extend(sentence.split())

    matrix = pd.DataFrame(columns=words, index=range(len(texts)))
    for i, sentence in enumerate(texts):
        sentence_words = sentence.split()

        for word in words:
            matrix.loc[i, word] = sentence_words.count(word) / len(sentence_words)

    return matrix


def invert_document_frequency(texts: list[str]):
    """ For each word, save (all-sentences / consist-sentences) """

    def count_in_texts(clause):
        return sum(clause in sentence.split() for sentence in texts)

    words = []
    for sentence in texts:
        words.extend(sentence.split())

    matrix = pd.DataFrame(columns=words, index=range(len(texts)))
    for i, sentence in enumerate(texts):
        sentence_words = sentence.split()

        for word in words:
            matrix.loc[i, word] = len(texts) / count_in_texts(word)

    return matrix


def TF_IDF(texts):
    return term_frequency(texts) * invert_document_frequency(texts)


texts = [
    'learning is my best entertainment',
    'she has some beautiful antique furniture',
    'I am not really depressed',
    'I would like to play tennis with you some day',
    'may I have another one, please',
    'why you did not come to the class yesterday',
    'did I play in your new tennis gym',
    'would you like to give me some suace',
    'how beautiful is your new furniture'
]

print("The TF-IDF calculated for sentences:")
print(TF_IDF(texts))
