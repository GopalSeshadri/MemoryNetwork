import numpy as np
import pandas as pd
import re
import nltk

from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class Preprocess:
    def tokenize(sent):
        '''
        This function takes in a sentence and returns a list of tokens.

        Parameters:
        sent (str) : The sentence to tokenize

        Returns:
        tokens (list) : A list of tokens
        '''
        tokens = word_tokenize(sent)
        return tokens

    def createTriples(file):
        '''
        This function takes in a file and returns a list of triples.
        A triple contains a tuple of story, question and answer.

        Parameters:
        file (file) : The tarfile

        Results:
        triples (list) : A list of triples of story, question and answer.
        '''
        triples, story = [], []
        for line in file:
            line = line.decode('utf-8').strip()
            # print(line)
            id, line = line.split(' ', 1)

            if int(id) == 1:                    #Whenever ecounter 1, it means a new story is starting
                story = []

            if '\t' in line:                    #Whenever encouter \t, that means it is a question and answer line
                question, answer, _  = line.split('\t')
                question = Preprocess.tokenize(question)

                story_indexed = [[str(idx)] + sent for idx, sent in enumerate(story) if sent]
                triples.append((story_indexed, question, answer))
            else:
                story.append(Preprocess.tokenize(line))

        return triples

    def should_flatten(input):
        '''
        This function takes in input and returns a boolean whether the type of input is not string or bytes.

        Parameters:
        input (obj) : It can be list or a string

        Returns:
        flag (boolean) : If the type of the input is not string or bytes
        '''
        return not isinstance(input, (str, bytes))

    def flatten(input):
        '''
        This function takes in list of lists as input and returns a flattened list.

        Parameters:
        input (list) : A list to be flattened
        '''
        for each in input:
            if Preprocess.should_flatten(each):
                yield from Preprocess.flatten(each)                # yield from yields each item of the iterable, but yield yields the iterable itself.
            else:
                yield each                              # This will be an individual string, hence yield the item itself

    def vectorizeTriples(triples, word2idx, story_maxsents, story_maxlen, question_maxlen):
        '''
        This function takes as input triples, word2idx and maximum lengths of
        story_maxsents, story_maxlen and question_maxlen

        Parameters:
        triples (list) : A list of triples of story, question and answer.
        word2idx (dict) : A dictionary of word to indices
        story_maxsents (int) : The maximum number of sentences in the story
        story_maxlen (int) : The maximum number of words in sentences in the story
        question_maxlen (int) : The maximum number of question

        Returns:
        stories (numpy array) : A list of list of padded indices of the story
        question (numpy array) : A list of padded indices of the question
        answer (numpy array) : A list of indices of answers
        '''
        stories, questions, answers = [], [], []
        for each in triples:
            stories.append([[word2idx[word] for word in sent] for sent in each[0]])
            questions.append([word2idx[word] for word in each[1]])
            answers.append([word2idx[each[2]]])

        #padding each sentences in story to length story_maxlen
        stories = [pad_sequences(each, maxlen = story_maxlen) for each in stories]

        #padding stories with sentences of zeros to match story_maxsents.
        for idx, story in enumerate(stories):
            stories[idx] = np.concatenate(
                [
                    story,
                    np.zeros((story_maxsents - story.shape[0], story_maxlen), 'int')
                ]
            )
        stories = np.stack(stories)
        #padding each questions to length question_maxlen
        questions = pad_sequences(questions, maxlen = question_maxlen)
        answers = np.array(answers)

        return stories, questions, answers

    def getData(challenge, tar):
        '''
        This function takes in the challenge path string and a tar object. Extrated the input files using the
        tar object and created triples from the file and created vectors on the same.

        Parameters:
        challenge (str) : THe challenge path name
        tar (object) : A tarfile object

        Returns:
        train_stories (list) : A list of triples of training set
        test_stories (list) : A list of triples of testing set
        stories_train, questions_train, answers_train (numpy array) : A list of padded and vectorized stories, question and answers of training set
        stories_test, questions_test, answers_test (numpy array) : A list of padded and vectorized stories, question and answers of testing set
        story_maxsents (int) : The maximum number of sentences in the story
        story_maxlen (int) : The maximum number of words in sentences in the story
        question_maxlen (int) : The maximum number of question
        vocab (list) : The list of unique words
        vocab_size (int) : The size of the vocabulary
        word2idx (dict) : The dictionary of words to indices
        '''
        train_stories = Preprocess.createTriples(tar.extractfile(challenge.format('train')))
        test_stories = Preprocess.createTriples(tar.extractfile(challenge.format('test')))
        all_stories = train_stories + test_stories

        story_maxlen = max([len(sent) for story, _, _ in all_stories for sent in story])
        story_maxsents = max([len(story) for story, _, _ in all_stories])
        question_maxlen = max([len(question) for _, question, _ in all_stories])

        vocab = sorted(set(Preprocess.flatten(all_stories)))
        vocab.insert(0, '<pad>')
        vocab_size = len(vocab)

        word2idx = {word : idx for idx, word in enumerate(vocab)}

        stories_train, questions_train, answers_train = \
            Preprocess.vectorizeTriples(train_stories, word2idx, story_maxsents, story_maxlen, question_maxlen)

        stories_test, questions_test, answers_test = \
            Preprocess.vectorizeTriples(test_stories, word2idx, story_maxsents, story_maxlen, question_maxlen)

        return train_stories, test_stories, \
            stories_train, questions_train, answers_train, \
            stories_test, questions_test, answers_test, \
            story_maxlen, story_maxsents, question_maxlen, \
            vocab, vocab_size, word2idx
