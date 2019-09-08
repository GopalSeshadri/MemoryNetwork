import numpy as np
import pandas as pd
import re

from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

class Preprocess:
    def tokenize(sent):
        return word_tokenize(sent)

    def createTriples(file):
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
        return not isinstance(input, (str, bytes))

    def flatten(input):
        for each in input:
            if Preprocess.should_flatten(each):
                yield from Preprocess.flatten(each)                # yield from yields each item of the iterable, but yield yields the iterable itself.
            else:
                yield each                              # This will be an individual string, hence yield the item itself

    def vectorizeTriples(triples, word2idx, story_maxsents, story_maxlen, question_maxlen):
        stories, questions, answers = [], [], []
        for each in triples:
            stories.append([[word2idx[word] for word in sent] for sent in each[0]])
            questions.append([word2idx[word] for word in each[1]])
            answers.append(each[2])

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
