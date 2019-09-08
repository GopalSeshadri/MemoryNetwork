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
train_stories, test_stories, \
    stories_train, questions_train, answers_train, \
    stories_test, questions_test, answers_test, \
    story_maxlen, story_maxsents, question_maxlen, \
    vocab, vocab_size, word2idx = Preprocess.getData(challenges['single_supporting_fact_10k'], tar)
