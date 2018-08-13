"""
Multi-head-attn based sentiment analysis model with IMDB dataset.
The code comes from https://kexue.fm/archives/4765
"""
from __future__ import print_function

from keras.datasets import imdb
from keras.preprocessing import sequence

from layers.bojone_attention_keras import MultiHeadAttn, PositionEncoding

max_features = 20000  # vocab size
max_seq_len = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Padding sequences...')
x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_seq_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Model
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dropout, Dense

S_inputs = Input(shape=(max_seq_len,), dtype='int32')
emb_seq = Embedding(max_features + 1, 128)(S_inputs)
emb_seq = PositionEncoding()(emb_seq)  # 增加pos enc能轻微提高准确率
O_seq = MultiHeadAttn(8, 16)([emb_seq, emb_seq, emb_seq])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test),
          verbose=2)
