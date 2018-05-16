from comet_ml import Experiment
from keras.backend import set_session
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Masking, TimeDistributed, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.backend import sparse_categorical_crossentropy
from keras.callbacks import TensorBoard, EarlyStopping
from functools import partial
from sys import exit
from os.path import exists
from os import environ
import tensorflow as tf
import keras
import numpy as np
import tbvaccine
tbvaccine.add_hook(isolate=True, show_vars=False)

mode = "train" # "train" or "inference"

WEIGHTS_FILE = "weights/iranaiyo11.h5"
tensorboard_logdir = None #"tfboard_logs/weights_512_s_10000_w_10000_l2_2nd"
SAVE_WEIGHTS = True
ONLY_CPU = True

if mode == "train" and SAVE_WEIGHTS and exists(WEIGHTS_FILE):
    print("Error: Specified WEIGHTS_FILE already exists.")
    exit(1)

if ONLY_CPU:
    NUM_CORES = 8
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,\
        inter_op_parallelism_threads=NUM_CORES, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 0})
    session = tf.Session(config=config)
    set_session(session)

START = "startsentence"
END   = "endsentence"

hidden_size = 512
batch_size  = 1
max_epochs  = 300
num_samples = 20 # None = use all samples
num_words   = 10000

with open("./data_set/cornell movie-dialogs corpus/train.enc") as ftrain_enc:
    train_enc_input = ftrain_enc.readlines()[:num_samples]

with open("./data_set/cornell movie-dialogs corpus/train.dec") as ftrain_dec:
    train_dec        = ftrain_dec.readlines()[:num_samples]
    train_dec_input  = [START + " " + sentence for sentence in train_dec]
    train_dec_target = [sentence + " " + END   for sentence in train_dec] # Teacher Forcing
"""
train_enc_input  = ["How are you?",  "How's the weather today?",      "You seems so happy today.",       "How about go out for lunch?", "Congraturation on your marriage!"]
train_dec        = ["Fine, thanks.", "It's snow. Very cold outside.", "My mother recoverd from a sick!", "Sounds good. Let's go.",      "Thank you! Today is my best day in my life."]
train_dec_input  = [START + " " + sentense for sentense in train_dec]
train_dec_target = [sentense + " " + END   for sentense in train_dec]
"""

tokenizer = Tokenizer(num_words=num_words) # 範囲外の消えた単語の影響が気になる。UNKにすべきでは
tokenizer.fit_on_texts(train_enc_input + train_dec_input + train_dec_target)

train_enc_input  = tokenizer.texts_to_sequences(train_enc_input)
train_dec_input  = tokenizer.texts_to_sequences(train_dec_input)
train_dec_target = tokenizer.texts_to_sequences(train_dec_target)

train_enc_input  = np.array(pad_sequences(train_enc_input,  padding="post"))
train_dec_input  = np.array(pad_sequences(train_dec_input,  padding="post"))
train_dec_target = np.array(pad_sequences(train_dec_target, padding="post"))

# データの次元をLSTMレイヤーの入力次元を合わせる。(batch_size, timesteps) => (batch_size, timesteps, feature_dim)
train_enc_input  = train_enc_input.reshape(train_enc_input.shape[0], train_enc_input.shape[1], 1)
train_dec_input  = train_dec_input.reshape(train_dec_input.shape[0], train_dec_input.shape[1], 1)
train_dec_target = train_dec_target.reshape(train_dec_target.shape[0], train_dec_target.shape[1], 1)

# layer declaration
encoder_inputs = Input(shape=(train_enc_input.shape[1], train_enc_input.shape[2])) # (timestep, feature_dim)
decoder_inputs = Input(shape=(train_dec_input.shape[1], train_dec_input.shape[2]))
encoder_lstm   = CuDNNLSTM(hidden_size, go_backwards=True, return_sequences=True, name="encoder_lstm")
encoder_lstm2  = CuDNNLSTM(hidden_size, return_state=True, name="encoder_lstm2")
decoder_lstm   = CuDNNLSTM(hidden_size, return_sequences=True, name="decoder_lstm") # LSTM = 382s 42ms/step CuDNNLSTM = 107s 12ms/step
decoder_lstm2  = CuDNNLSTM(hidden_size, return_sequences=True, name="decoder_lstm2")
decoder_dense  = Dense(num_words, activation="softmax", name="output_dence")
if ONLY_CPU:
    encoder_lstm  = LSTM(hidden_size, go_backwards=True, return_state=True, return_sequences=True, name="encoder_lstm")
    encoder_lstm2 = LSTM(hidden_size, return_state=True, name="encoder_lstm2")
    decoder_lstm  = LSTM(hidden_size, return_sequences=True, name="decoder_lstm")
    decoder_lstm2 = LSTM(hidden_size, return_sequences=True, name="decoder_lstm2")

def train():
    # if SAVE_WEIGHTS:
    #     experiment = Experiment(api_key=environ["COMET_ML_API_KEY"])

    layer1_out          = encoder_lstm(encoder_inputs)
    _, state_h, state_c = encoder_lstm2(layer1_out)

    encoder_states = [state_h, state_c]

    outputs1        = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    outputs2        = decoder_lstm2(outputs1)
    decoder_outputs = decoder_dense(outputs2)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # loss = partial(sparse_categorical_crossentropy, from_logits=True)
    loss = "sparse_categorical_crossentropy"
    callbacks = [EarlyStopping(monitor="loss", patience=5, verbose=1)]
    if tensorboard_logdir is not None:
        callbacks.append(TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1))

    model.compile(optimizer="adam", loss=loss)
    model.fit([train_enc_input, train_dec_input], train_dec_target,
              batch_size=batch_size,
              epochs=max_epochs,
              # validation_split=0.1,
              callbacks=callbacks
             )

    if SAVE_WEIGHTS:
        print("Save weights...")
        model.save_weights(WEIGHTS_FILE)

def inference():
    encoder_inputs = Input(shape=(None, 1))

    layer1_out = encoder_lstm(encoder_inputs)
    _, h, c    = encoder_lstm2(layer1_out)

    encoder_states = [h, c]

    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.load_weights(WEIGHTS_FILE, by_name=True)

    # 学習時はencoder_statesをそのまま入力するだけだったのでstateのInputは不要だったが、
    # 推論時は生成の各ステップでdecoderのstateを入力するので、Inputが必要。
    decoder_inputs = Input(shape=(1,1))
    state_input_h1 = Input(shape=(hidden_size,))
    state_input_c1 = Input(shape=(hidden_size,))
    state_input_h2 = Input(shape=(hidden_size,))
    state_input_c2 = Input(shape=(hidden_size,))
    states_inputs  = [state_input_h1, state_input_c1, state_input_h2, state_input_c2]

    # return states only in inferences
    decoder_lstm.return_state  = True
    decoder_lstm2.return_state = True

    decoder_out,  h1, c1 = decoder_lstm(decoder_inputs, initial_state=states_inputs[:2])
    decoder_out2, h2, c2 = decoder_lstm2(decoder_out,  initial_state=states_inputs[2:])
    decoder_outputs       = decoder_dense(lstm_outputs2)
    states_outputs = [h1, c1, h2, c2]
    decoder_model  = Model([decoder_inputs] + states_inputs, [decoder_outputs] + states_outputs)
    decoder_model.load_weights(WEIGHTS_FILE, by_name=True)

    input_seq = tokenizer.texts_to_sequences(["You have my word.  As a gentleman"])
    input_seq = np.array(input_seq)
    # input_seq = input_seq[:,:,np.newaxis]

    # with open("./data_set/cornell movie-dialogs corpus/train.enc") as f:
    #     test_enc_input = f.readlines()[:10000]
    # test_enc_input = tokenizer.texts_to_sequences(test_enc_input)
    # test_enc_input = np.array([np.array(test) for test in test_enc_input])
    # test_enc_input = [test[:,np.newaxis] for test in test_enc_input] # (entire_size, timesteps) => (entire, timesteps, feature_dim)

    # input_seqs = train_enc_input
    input_seqs = [input_seq]

    results = []
    """ encode """
    for i, input_seq in enumerate(input_seqs):
        print(i)
        states_values = encoder_model.predict(input_seq[:,:,np.newaxis]) # (timesteps, feature_dim) => (batch_size, timesteps, feature_dim)
        # 1st layer is initialized by encoder states
        # 2nd layer is initialized by zeros
        # Input expects (batch_size, hidden_size)
        # For this time, batch_size's actual number does not care
        # but it must have 2 dimmensions. So add 1 to dim 1.
        second_layer_init_states = [np.zeros((1,hidden_size)), np.zeros((1,hidden_size))]
        states_values            = states_values + second_layer_init_states

        target_seq = np.array([[[tokenizer.word_index[START]]]]) # (1, 1, 1)

        index_to_word      = {v:k for k,v in tokenizer.word_index.items()}
        index_to_word[0]   = ""
        generated_sentense = ""
        MAX_GEN_WORD       = 1000
        """ decode """
        for j in range(MAX_GEN_WORD):
            output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq] + states_values)

            sampled_index = np.argmax(output_tokens[0, 0, :])
            sampled_word = index_to_word[sampled_index]
            if (sampled_word == END):
                break

            print_debug(j, output_tokens, sampled_index, sampled_word, generated_sentense)
            generated_sentense += str(sampled_word) + " "


            target_seq = np.array([[[sampled_index]]])
            states_values = [h1, c1, h2, c2]

        results.append(generated_sentense)

    return results

def print_debug(count, output_tokens, sampled_index, sampled_word, decoded_sentense):
    print("-----")
    print("count = " + str(count))
    print("raw outputs = ")
    print(output_tokens)
    print("index = ")
    print(sampled_index)
    print("word = ")
    print(sampled_word)
    print("sentence = ")
    print(decoded_sentense)
    print("-----")

if __name__ == "__main__":
    if   mode == "train":
        train()
    elif mode == "inference":
        decoded_sentenses = inference()
        print(decoded_sentenses)
        # with open("iranaiyo", "w") as f:
        #     for sentense in decoded_sentenses:
        #             f.write(sentense + "\n")
