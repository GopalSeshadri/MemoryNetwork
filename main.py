import numpy as np
import pandas as pd
import keras
import tarfile
from preprocess import Preprocess
from models import Models

EMBEDDING_DIM = 32
NUM_EPOCHS = 10
BATCH_SIZE = 32

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
    Models.singleModel(ss_story_maxlen, ss_story_maxsents, ss_question_maxlen, ss_vocab_size, EMBEDDING_DIM)

print(ss_idx2word)
single_model.fit([ss_stories_train, ss_questions_train], ss_answers_train, \
    epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, validation_data = ([ss_stories_test, ss_questions_test], ss_answers_test))

ss_random_idx = np.random.choice(len(ss_stories_test))
ss_random_story = ss_stories_test[ss_random_idx : ss_random_idx + 1]
ss_random_question = ss_questions_test[ss_random_idx : ss_random_idx + 1]
ss_random_answer = ss_idx2word[np.argmax(single_model.predict([ss_random_story, ss_random_question]))]
ss_random_weights = single_debug_model.predict([ss_random_story, ss_random_question]).flatten()

s, q, a = ss_test_stories[ss_random_idx]
for idx, sent in enumerate(s):
    print('{:.2f}\t{}'.format(ss_random_weights[idx], ' '.join(sent)))
print('The question is : {}'.format(' '.join(q)))
print('The predicted anwer is : {}'.format(ss_random_answer))
print('The correct answer is : {}'.format(a))
