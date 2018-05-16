class ChatbotModel:

    def __init__(self):
        self.encoder_lstm   = CuDNNLSTM(hidden_size, go_backwards=True, return_state=True, name="encoder_lstm")
        self.encoder_inputs = Input(shape=(train_enc_input.shape[1], train_enc_input.shape[2])) # (timestep, feature_dim)

        self.decoder_inputs = Input(shape=(train_dec_input.shape[1], train_dec_input.shape[2]))
        self.decoder_lstm   = CuDNNLSTM(hidden_size, return_sequences=True, return_state=True, name="decoder_lstm") # LSTM = 382s 42ms/step CuDNNLSTM = 107s 12ms/step
        self.decoder_lstm2  = CuDNNLSTM(hidden_size, return_sequences=True, return_state=True, name="decoder_lstm2")
        self.decoder_dense  = Dense(num_words, activation="softmax", name="output_dence")

    def train(self):
        _, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]

        lstm_outputs1, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        lstm_outputs2, _, _ = decoder_lstm2(lstm_outputs1)
        decoder_outputs = decoder_dense(lstm_outputs2)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # loss = partial(sparse_categorical_crossentropy, from_logits=True)
        loss = "sparse_categorical_crossentropy"

        # sparse_categorical_crossentropy = ラベルをone-hotにしなくていい + 最後の層の出力をsoftmaxしなくていい
        model.compile(optimizer="adam", loss=loss)
        model.fit([train_enc_input, train_dec_input], train_dec_target,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1)

        if SAVE_WEIGHTS:
            model.save_weights(WEIGHTS_FILE)
