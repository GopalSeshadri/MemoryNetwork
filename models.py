import numpy as np
from keras.layers import Input, Embedding, Lambda, Dot
from keras.layers import Embedding, Dense, Activation, Reshape, Dropout
from keras.optimizers import Adam, RMSprop
from keras.models import Model
import keras.backend as K

class Models:
    def singleModel(ss_story_maxlen, ss_story_maxsents, ss_question_maxlen, ss_vocab_size,
                    ss_stories_train, ss_questions_train, ss_answers_train,
                    ss_stories_test, ss_questions_test, ss_answers_test,
                    embedding_dim, num_epochs, batch_size):

        '''
        This function takes in training data and testing data for stories, question and answers, max lengths of story and question,
        maximum sentence in story, vocab size, embedding dimension, number of epochs and batch size. Returns the models and debugging
        models for single fact problem.

        Parameters:
        ss_story_maxlen (int) : The maximum number of words in sentences in the story
        ss_story_maxsents (int) : The maximum number of sentences in the story
        ss_question_maxlen (int) : The maximum number of question
        ss_vocab_size (int) : The size of the vocabulary
        ss_stories_train, ss_questions_train, ss_answers_train (numpy array) : A list of padded and vectorized stories, question and answers of training set
        ss_stories_test, ss_questions_test, ss_answers_test (numpy array) : A list of padded and vectorized stories, question and answers of testing set
        embedding_dim (int) : The size of embedding
        num_epochs (int) : The number of epochs
        batch_size (int) : The size of mini batches

        Returns:
        single_model (keras model) : The model trained on Single Fact Dataset
        single_debug_model (keras model) : The debug model for Single Fact Dataset
        '''
        input_story = Input(shape = (ss_story_maxsents, ss_story_maxlen))
        embedded_story = Embedding(ss_vocab_size, embedding_dim)(input_story)
        summed_across_words_story = Lambda(lambda x: K.sum(x, axis = 2))(embedded_story)
        # print(summed_across_words_story.shape)

        input_question = Input(shape = (ss_question_maxlen,))
        embedded_question = Embedding(ss_vocab_size, embedding_dim)(input_question)
        # print(embedded_question.shape)
        summed_across_words_question = Lambda(lambda x: K.sum(x, axis = 1))(embedded_question)
        # print(summed_across_words_question.shape)
        summed_across_words_question = Reshape((1, embedding_dim))(summed_across_words_question)
        # print(summed_across_words_question.shape)

        x = Dot(axes = 2)([summed_across_words_story, summed_across_words_question])
        # print(x.shape)
        x = Reshape((ss_story_maxsents,))(x)
        # print(x.shape)
        x = Activation('softmax')(x)
        sent_weights = Reshape((ss_story_maxsents, 1))(x)
        # print(sent_weights.shape)

        x = Dot(axes = 1)([sent_weights, summed_across_words_story])
        # print(x.shape)
        x = Reshape((embedding_dim,))(x)
        # print(x.shape)
        out = Dense(ss_vocab_size, activation = 'softmax')(x)
        # print(out.shape)

        single_model = Model([input_story, input_question], out)

        single_model.compile(optimizer = RMSprop(lr = 1e-3),
                            loss = 'sparse_categorical_crossentropy',
                            metrics = ['accuracy'])


        single_model.fit([ss_stories_train, ss_questions_train], ss_answers_train, \
            epochs = num_epochs, batch_size = batch_size, validation_data = ([ss_stories_test, ss_questions_test], ss_answers_test))

        single_debug_model = Model([input_story, input_question], sent_weights)

        return single_model, single_debug_model

    def doubleModel(ts_story_maxlen, ts_story_maxsents, ts_question_maxlen, ts_vocab_size,
                    ts_stories_train, ts_questions_train, ts_answers_train,
                    ts_stories_test, ts_questions_test, ts_answers_test,
                    embedding_dim, num_epochs, batch_size):
        '''
        This function takes in training data and testing data for stories, question and answers, max lengths of story and question,
        maximum sentence in story, vocab size, embedding dimension, number of epochs and batch size. Returns the models and debugging
        models for two fact problem.

        Parameters:
        ss_story_maxlen (int) : The maximum number of words in sentences in the story
        ss_story_maxsents (int) : The maximum number of sentences in the story
        ss_question_maxlen (int) : The maximum number of question
        ss_vocab_size (int) : The size of the vocabulary
        ss_stories_train, ss_questions_train, ss_answers_train (numpy array) : A list of padded and vectorized stories, question and answers of training set
        ss_stories_test, ss_questions_test, ss_answers_test (numpy array) : A list of padded and vectorized stories, question and answers of testing set
        embedding_dim (int) : The size of embedding
        num_epochs (int) : The number of epochs
        batch_size (int) : The size of mini batches

        Returns:
        double_model (keras model) : The model trained on two fact dataset
        double_debug_model (keras model) : The debug model for two fact dataset
        '''
        input_story = Input(shape = (ts_story_maxsents, ts_story_maxlen))
        embedded_story = Embedding(ts_vocab_size, embedding_dim)(input_story)
        summed_across_words_story = Lambda(lambda x: K.sum(x, axis = 2))(embedded_story)

        input_question = Input(shape = (ts_question_maxlen,))
        embedded_question = Embedding(ts_vocab_size, embedding_dim)(input_question)
        summed_across_words_question = Lambda(lambda x : K.sum(x, axis = 1))(embedded_question)

        def hop(story, query):
            # here query can be the question or the answer of the first hop
            x = Reshape((1, embedding_dim))(query)
            x = Dot(axes = 2)([story, x])
            x = Reshape((ts_story_maxsents,))(x)
            x = Activation('softmax')(x)
            sent_weights = Reshape((ts_story_maxsents, 1))(x)

            story_embedding2 = Embedding(ts_vocab_size, embedding_dim)(input_story)
            summed_across_words_story2 = Lambda(lambda x : K.sum(x, axis = 2))(story_embedding2)
            x = Dot(axes = 1)([sent_weights, summed_across_words_story2])
            x = Reshape((embedding_dim, ))(x)
            x = Dropout(0.1)(x)
            out = Dense(embedding_dim, activation = 'elu')(x)

            return out, summed_across_words_story2, sent_weights

        answer_1, summed_across_words_story, sent_weights_1 = hop(summed_across_words_story, summed_across_words_question)
        answer_2, _, sent_weights_2 = hop(summed_across_words_story, answer_1)

        answer = Dense(ts_vocab_size, activation = 'softmax')(answer_2)

        double_model = Model([input_story, input_question], answer)

        double_model.compile(optimizer = RMSprop(lr = 1e-3),
                            loss = 'sparse_categorical_crossentropy',
                            metrics = ['accuracy'])

        double_model.fit([ts_stories_train, ts_questions_train], ts_answers_train, \
            epochs = num_epochs, batch_size = batch_size, validation_data = ([ts_stories_test, ts_questions_test], ts_answers_test))

        double_debug_model = Model([input_story, input_question], [sent_weights_1, sent_weights_2])

        return double_model, double_debug_model

    def predictSingleModelAnswer(test_stories, stories_test, questions_test, idx2word, pred_model, debug_model):
        '''
        This function takes as input test story data, tokenized stories and question from test set, idx2word, model and debug model
        of single fact problem and returns the text of story, question, correct answer, weights and predicted answer.

        Parameters:
        test_stories (list) : The list of list of story text from test set where each entity is a sentence in a story
        stories_test (list) : The tokenized list of stories from test set
        questions_test (list) : The tokenized list of questions from test set
        idx2word (dict) : A dictionary of indices to words in the vocab
        pred_model (keras model) : The prediction model for single fact problem
        debug_model (keras model) : The debug model for single fact problem

        Returns:
        s (str) : A random story picked from test set, with words of tokens merged into single text
        q (str) : The question of the corresponding random story picked from test set, with words of tokens merged into single text
        a (str) : The answer for the corresponding story and question
        random_weights (numpy array) : The sentence weights from the debug model
        random_answer (str) : The predicted answer
        '''
        random_idx = np.random.choice(len(stories_test))
        random_story = stories_test[random_idx : random_idx + 1]
        random_question = questions_test[random_idx : random_idx + 1]
        random_answer = idx2word[np.argmax(pred_model.predict([random_story, random_question]))]
        random_weights = debug_model.predict([random_story, random_question]).flatten()

        s, q, a = test_stories[random_idx]
        for idx, sent in enumerate(s):
            print('{:.2f}\t{}'.format(random_weights[idx], ' '.join(sent)))
            s[idx] = ' '.join(sent)
        print('The question is : {}'.format(' '.join(q)))
        print('The predicted anwer is : {}'.format(random_answer))
        print('The correct answer is : {}'.format(a))
        q = ' '.join(q)
        return s, q, a, random_weights, random_answer

    def predictDoubleModelAnswer(test_stories, stories_test, questions_test, idx2word, pred_model, debug_model):
        '''
        This function takes as input test story data, tokenized stories and question from test set, idx2word, model and debug model
        of two fact problem and returns the text of story, question, correct answer, weights and predicted answer.

        Parameters:
        test_stories (list) : The list of list of story text from test set where each entity is a sentence in a story
        stories_test (list) : The tokenized list of stories from test set
        questions_test (list) : The tokenized list of questions from test set
        idx2word (dict) : A dictionary of indices to words in the vocab
        pred_model (keras model) : The prediction model for two fact problem
        debug_model (keras model) : The debug model for two fact problem

        Returns:
        s (str) : A random story picked from test set, with words of tokens merged into single text
        q (str) : The question of the corresponding random story picked from test set, with words of tokens merged into single text
        a (str) : The answer for the corresponding story and question
        random_weights1 (numpy array) : The sentence weights of the first hop from the debug model
        random_weights2 (numpy array) : The sentence weights of the second hop from the debug model
        random_answer (str) : The predicted answer
        '''
        random_idx = np.random.choice(len(stories_test))
        random_story = stories_test[random_idx : random_idx + 1]
        random_question = questions_test[random_idx : random_idx + 1]
        random_answer = idx2word[np.argmax(pred_model.predict([random_story, random_question]))]
        random_weights1, random_weights2 = debug_model.predict([random_story, random_question])
        random_weights1 = random_weights1.flatten()
        random_weights2 = random_weights2.flatten()

        s, q, a = test_stories[random_idx]
        for idx, sent in enumerate(s):
            print('{:.2f}\t{:.2f}\t{}'.format(random_weights1[idx], random_weights2[idx],' '.join(sent)))
            s[idx] = ' '.join(sent)
        print('The question is : {}'.format(' '.join(q)))
        print('The predicted anwer is : {}'.format(random_answer))
        print('The correct answer is : {}'.format(a))
        q = ' '.join(q)
        return s, q, a, random_weights1, random_weights2, random_answer
