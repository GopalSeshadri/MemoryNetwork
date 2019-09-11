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

        print('were')
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
