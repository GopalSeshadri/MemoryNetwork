import numpy as np
import pandas as pd
import keras
import tarfile
from preprocess import Preprocess
from models import Models
from utilities import Utilities

EMBEDDING_DIM = 32
NUM_EPOCHS = 10
NUM_EPOCHS_2 = 50
BATCH_SIZE = 32

class Main:
    def saveModels():
        tar = tarfile.open('Data/babi_tasks_1-20_v1-2.tar.gz')

        challenges = {
          # QA1 with 10,000 samples
          'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
          # QA2 with 10,000 samples
          'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
        }

        ## Single Supporting Fact Challenge
        ss_train_stories, ss_test_stories, \
            ss_stories_train, ss_questions_train, ss_answers_train, \
            ss_stories_test, ss_questions_test, ss_answers_test, \
            ss_story_maxlen, ss_story_maxsents, ss_question_maxlen, \
            ss_vocab, ss_vocab_size, ss_word2idx = \
            Preprocess.getData(challenges['single_supporting_fact_10k'], tar)

        ss_idx2word = {value : key for key, value in ss_word2idx.items()}

        single_model, single_debug_model = \
            Models.singleModel(ss_story_maxlen, ss_story_maxsents, ss_question_maxlen, ss_vocab_size, \
                            ss_stories_train, ss_questions_train, ss_answers_train, \
                            ss_stories_test, ss_questions_test, ss_answers_test, \
                            EMBEDDING_DIM, NUM_EPOCHS, BATCH_SIZE)

        Utilities.saveModel(single_model, 'single_model')
        Utilities.saveModel(single_debug_model, 'single_debug_model')

        ## Two Supporting Fact challenge
        ts_train_stories, ts_test_stories, \
            ts_stories_train, ts_questions_train, ts_answers_train, \
            ts_stories_test, ts_questions_test, ts_answers_test, \
            ts_story_maxlen, ts_story_maxsents, ts_question_maxlen, \
            ts_vocab, ts_vocab_size, ts_word2idx = \
            Preprocess.getData(challenges['two_supporting_facts_10k'], tar)

        ts_idx2word = {value : key for key, value in ts_word2idx.items()}

        double_model, double_debug_model = \
            Models.doubleModel(ts_story_maxlen, ts_story_maxsents, ts_question_maxlen, ts_vocab_size, \
                            ts_stories_train, ts_questions_train, ts_answers_train, \
                            ts_stories_test, ts_questions_test, ts_answers_test, \
                            EMBEDDING_DIM, NUM_EPOCHS_2, BATCH_SIZE)

        Utilities.saveModel(double_model, 'double_model')
        Utilities.saveModel(double_debug_model, 'double_debug_model')

    def generateAnswer(choice):
        tar = tarfile.open('Data/babi_tasks_1-20_v1-2.tar.gz')

        challenges = {
          # QA1 with 10,000 samples
          'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
          # QA2 with 10,000 samples
          'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
        }

        if choice == 'single':
            ## Single Supporting Fact Challenge
            ss_train_stories, ss_test_stories, \
                ss_stories_train, ss_questions_train, ss_answers_train, \
                ss_stories_test, ss_questions_test, ss_answers_test, \
                ss_story_maxlen, ss_story_maxsents, ss_question_maxlen, \
                ss_vocab, ss_vocab_size, ss_word2idx = \
                Preprocess.getData(challenges['single_supporting_fact_10k'], tar)

            ss_idx2word = {value : key for key, value in ss_word2idx.items()}

            single_model = Utilities.loadModel('single_model')
            single_debug_model = Utilities.loadModel('single_debug_model')

            story, question, correct_answer, weights2, predicted_answer = Models.predictSingleModelAnswer(ss_test_stories, ss_stories_test, ss_questions_test, ss_idx2word, single_model, single_debug_model)

        else:
            ## Two Supporting Fact challenge
            ts_train_stories, ts_test_stories, \
                ts_stories_train, ts_questions_train, ts_answers_train, \
                ts_stories_test, ts_questions_test, ts_answers_test, \
                ts_story_maxlen, ts_story_maxsents, ts_question_maxlen, \
                ts_vocab, ts_vocab_size, ts_word2idx = \
                Preprocess.getData(challenges['two_supporting_facts_10k'], tar)

            ts_idx2word = {value : key for key, value in ts_word2idx.items()}

            double_model = Utilities.loadModel('double_model')
            double_debug_model = Utilities.loadModel('double_debug_model')

            story, question, correct_answer, weights1, weights2, predicted_answer = Models.predictDoubleModelAnswer(ts_test_stories, ts_stories_test, ts_questions_test, ts_idx2word, double_model, double_debug_model)


Main.saveModels()
