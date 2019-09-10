from keras.layers import Input, Embedding, Lambda, Dot
from keras.layers import Embedding, Dense, Activation, Reshape
from keras.optimizers import Adam, RMSprop
from keras.models import Model
import keras.backend as K

class Models:
    def singleModel(ss_story_maxlen, ss_story_maxsents, ss_question_maxlen, ss_vocab_size, embedding_dim):
        input_story = Input(shape = (ss_story_maxsents, ss_story_maxlen))
        embedded_story = Embedding(ss_vocab_size, embedding_dim)(input_story)
        summed_across_words_story = Lambda(lambda x: K.sum(x, axis = 2))(embedded_story)
        print(summed_across_words_story.shape)

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

        single_debug_model = Model([input_story, input_question], sent_weights)

        return single_model, single_debug_model
