import numpy as np
import pandas as pd
import keras
import tarfile
from preprocess import Preprocess

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
