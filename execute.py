from comet_ml import Experiment
from keras.backend import set_session
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Masking, TimeDistributed, Bidirectional, Embedding
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

mode = "inference" # "train" or "inference"

WEIGHTS_FILE = "weights/iranaiyo12.h5"
PRETRAINED_VEC_FILE = "data_set/GloVe/glove.6B.300d.txt"
tensorboard_logdir = None #"tfboard_logs/weights_512_s_10000_w_10000_l2_2nd"
SAVE_WEIGHTS = False
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

hidden_size   = 512
batch_size    = 5
max_epochs    = 300
num_samples   = 30 # None = use all samples
num_words     = 10000
embedding_dim = 300

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
# Embedding Layerを使う場合は内部で3次元に変換されるため必要ない
# train_enc_input  = train_enc_input.reshape(train_enc_input.shape[0], train_enc_input.shape[1], 1)
# train_dec_input  = train_dec_input.reshape(train_dec_input.shape[0], train_dec_input.shape[1], 1)
# train_dec_target = train_dec_target.reshape(train_dec_target.shape[0], train_dec_target.shape[1], 1)

# Embedding Matrixをつくる
embedding_index = {}
with open(PRETRAINED_VEC_FILE, "r") as f:
    for line in f:
        _line = line.split()
        word, vec = _line[0], np.asarray(_line[1:])
        embedding_index[word] = vec

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# layer declaration
# encoder_inputs = Input(shape=(train_enc_input.shape[1], train_enc_input.shape[2])) # (timestep, feature_dim)
# decoder_inputs = Input(shape=(train_dec_input.shape[1], train_dec_input.shape[2]))
encoder_inputs = Input(shape=(train_enc_input.shape[1],))
decoder_inputs = Input(shape=(train_dec_input.shape[1],))
embedding      = Embedding(len(tokenizer.word_index) + 1, embedding_dim, weights=[embedding_matrix], trainable=False)
encoder_lstm   = CuDNNLSTM(hidden_size, go_backwards=True, return_sequences=True, name="encoder_lstm")
encoder_lstm2  = CuDNNLSTM(hidden_size, return_state=True, name="encoder_lstm2")
decoder_lstm   = CuDNNLSTM(hidden_size, return_sequences=True, name="decoder_lstm") # LSTM = 382s 42ms/step CuDNNLSTM = 107s 12ms/step
decoder_lstm2  = CuDNNLSTM(hidden_size, return_sequences=True, name="decoder_lstm2")
decoder_dense  = Dense(num_words, activation="linear", name="output_dence")
if ONLY_CPU:
    encoder_lstm  = LSTM(hidden_size, go_backwards=True, return_state=True, return_sequences=True, name="encoder_lstm")
    encoder_lstm2 = LSTM(hidden_size, return_state=True, name="encoder_lstm2")
    decoder_lstm  = LSTM(hidden_size, return_sequences=True, name="decoder_lstm")
    decoder_lstm2 = LSTM(hidden_size, return_sequences=True, name="decoder_lstm2")

def train():
    # if SAVE_WEIGHTS:
    #     experiment = Experiment(api_key=environ["COMET_ML_API_KEY"])

    enc_emb    = embedding(encoder_inputs)
    layer1_out = encoder_lstm(enc_emb)
    _, h, c    = encoder_lstm2(layer1_out)
    encoder_states = [h, c]

    dec_emb        = embedding(decoder_inputs)
    output1        = decoder_lstm(dec_emb, initial_state=encoder_states)
    output2        = decoder_lstm2(output1)
    decoder_output = decoder_dense(output2)

    model = Model([encoder_inputs, decoder_inputs], decoder_output)

    # loss = partial(sparse_categorical_crossentropy, from_logits=True)
    # loss = "sparse_categorical_crossentropy"
    # workaround of https://github.com/tensorflow/tensorflow/issues/17150
    def sparse_loss(y_true, y_pred):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss_mean = tf.reduce_mean(loss)
        return loss_mean
    loss = sparse_loss

    callbacks = [EarlyStopping(monitor="loss", patience=5, verbose=1)]
    if tensorboard_logdir is not None:
        callbacks.append(TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1))

    decoder_target = tf.placeholder(dtype="int32", shape=(None, train_dec_target.shape[1]))
    model.compile(optimizer="adam", loss=loss, target_tensors=[decoder_target])
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
    encoder_inputs = Input(shape=(None,))

    enc_emb    = embedding(encoder_inputs)
    layer1_out = encoder_lstm(enc_emb)
    _, h, c    = encoder_lstm2(layer1_out)

    encoder_states = [h, c]

    encoder_model = Model(encoder_inputs, encoder_states)
    #print(encoder_model.summary()); exit()
    encoder_model.load_weights(WEIGHTS_FILE, by_name=True)

    # 学習時はencoder_statesをそのまま入力するだけだったのでstateのInputは不要だったが、
    # 推論時は生成の各ステップでdecoderのstateを入力するので、Inputが必要。
    decoder_inputs = Input(shape=(1,))
    state_input_h1 = Input(shape=(hidden_size,))
    state_input_c1 = Input(shape=(hidden_size,))
    state_input_h2 = Input(shape=(hidden_size,))
    state_input_c2 = Input(shape=(hidden_size,))
    states_inputs  = [state_input_h1, state_input_c1, state_input_h2, state_input_c2]

    # return states only in inferences
    decoder_lstm.return_state  = True
    decoder_lstm2.return_state = True

    dec_emb              = embedding(decoder_inputs)
    decoder_out,  h1, c1 = decoder_lstm(dec_emb, initial_state=states_inputs[:2])
    decoder_out2, h2, c2 = decoder_lstm2(decoder_out,  initial_state=states_inputs[2:])
    decoder_output       = decoder_dense(decoder_out2)
    states_outputs = [h1, c1, h2, c2]
    decoder_model  = Model([decoder_inputs] + states_inputs, [decoder_output] + states_outputs)
    decoder_model.load_weights(WEIGHTS_FILE, by_name=True)

    # input_seq = tokenizer.texts_to_sequences(["You have my word.  As a gentleman"])
    input_seq = tokenizer.texts_to_sequences(["The thing is, Cameron -- I'm at the mercy of a particularly hideous breed of loser."])
    # input_seq = tokenizer.texts_to_sequences(["Why?"])
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
    """--- INFERENCE PROCESS ---"""
    """ encode """
    for i, input_seq in enumerate(input_seqs):
        print(i)
        # print(input_seq.shape); exit()
        states_values = encoder_model.predict(input_seq) # (timesteps, feature_dim) => (batch_size, timesteps, feature_dim)
        # 1st layer is initialized by encoder states
        # 2nd layer is initialized by zeros
        # Input expects (batch_size, hidden_size)
        # For this time, batch_size's actual number does not care
        # but it must have 2 dimmensions. So add 1 to dim 1.
        second_layer_init_states = [np.zeros((1,hidden_size)), np.zeros((1,hidden_size))]
        states_values            = states_values + second_layer_init_states

        target_seq = np.array([[tokenizer.word_index[START]]]) # (1, 1)

        index_to_word      = {v:k for k,v in tokenizer.word_index.items()}
        index_to_word[0]   = ""
        generated_sentense = ""
        MAX_GEN_WORD       = 1000
        """ decode """
        for j in range(MAX_GEN_WORD):
            output_token, h1, c1, h2, c2 = decoder_model.predict([target_seq] + states_values)

            sampled_index = np.argmax(output_token[0, 0, :])
            sampled_word = index_to_word[sampled_index]
            print_debug(j, output_token, sampled_index, sampled_word, generated_sentense)
            if (sampled_word == END):
                break

            generated_sentense += str(sampled_word) + " "

            target_seq = np.array([[sampled_index]])
            states_values = [h1, c1, h2, c2]

        results.append(generated_sentense)

    return results

def print_debug(count, output_token, sampled_index, sampled_word, decoded_sentense):
    print("-----")
    print("count = " + str(count))
    print("raw outputs = ")
    print(output_token)
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
