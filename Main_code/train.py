import json
import time
import os
import numpy as np
from keras.utils.data_utils import pad_sequences
from keras.layers import Dense, Dropout, GlobalMaxPooling1D, LSTM, Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras import Sequential
import util


BASE_PATH = "/content/drive/MyDrive/meme"
MODEL_NAME = 'meme_text_gen'
MODEL_PATH = util.get_model_path(BASE_PATH, MODEL_NAME)
os.mkdir(MODEL_PATH)


SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 16
ROWS_TO_SCAN = 2000000
NUM_EPOCHS = 30
BATCH_SIZE = 256


print('loading json data...')
t = time.time()

# tu ide dataset cez ktorý trénujeme
training_data = json.load(open(BASE_PATH + '/merged_dataset.json'))

print('loading json took %ds' % round(time.time() - t))
util.print_memory()


print('scanning %d of %d json rows...' % (min(ROWS_TO_SCAN, len(training_data)), len(training_data)))
t = time.time()

texts = []  # list of text samples

# labels_index dictionary pismeno a jeho hodnota, urobene podla citania json textu
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
label_id_counter = 0

#toto prejde text, po písmene a vytvorí taketo veci
# ["000000061533  0  make", "|"],
# ["000000061533  1  make|a", "l"]
# ["000000061533  1  make|al", "l"],
# ["000000061533  1  make|all the memes", "|"],

#menenie jsona a jeho textu do vlastnej doby, kde sa riešia labels a ich indexy, plus koľko riadkov ma memecko
for i, row in enumerate(training_data):
    template_id = str(row[0]).zfill(12)
    text = row[1].lower()
    start_index = len(template_id) + 2 + 1 + 2  # template_id, spaces, box_index, spaces
    box_index = 0
    #z textu, berie postupne pismena, priradi im cislo
    #takto pretvori postupnost pismien na postupnost cisel
    #z textu vznikaju ciselne vety, ulozene v labels
    for j in range(0, len(text)):
        char = text[j]
        # note: it is critical that the number of spaces plus len(box_index) is >= the convolution width
        texts.append(template_id + '  ' + str(box_index) + '  ' + text[0:j])
        if char in labels_index:                 #ak pismeno je v label_index
            label_id = labels_index[char]        #
        else:
            label_id = label_id_counter         #
            labels_index[char] = label_id       #
            label_id_counter += 1              # increment label id counter
        labels.append(label_id)
        if char == '|':
            box_index += 1

    if i >= ROWS_TO_SCAN:
        break


print('tokenizing %d texts...' % len(texts))
del training_data  # free memory

#char to int pretvori znaky z textu na ciselne hodnoty, najpouzivanejsie zoradi a da im hodnotu
char_to_int = util.map_char_to_int(texts)

#text obsahuje už formu# ["000000061533  1  make|all the memes", "|"],, teraz všetko v hranatých zatvorkach prerobí
#do číselnej podoby a ulozi do sequences
sequences = util.texts_to_sequences(texts, char_to_int)
del texts  # free memory

print('saving tokenizer and labels to file...')

# save tokenizer, label indexes, and parameters so they can be used for predicting later
#ulozí parametre do modela na neskoršie predikovanie
with open(MODEL_PATH + '/params.json', 'w') as handle:
    json.dump({
        'sequence_length': SEQUENCE_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'num_rows_used': len(sequences),
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'char_to_int': char_to_int,
        'labels_index': labels_index
    }, handle)

print('found %s unique tokens.' % len(char_to_int))
print('padding sequences...')
data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)   #doplnenie cisel do maximalnej dlzky ciselnych viet
del sequences  # free memory
labels = np.asarray(labels)

util.print_memory()

print('data:', data)
print('labels:', labels)
print('shape of data tensor:', data.shape)
print('shape of label tensor:', labels.shape)

# split data into training and validation
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# validation set can be much smaller if we use a lot of data (source: andrew ng on coursera video)
validation_ratio = 0.2 if data.shape[0] < 1000000 else 0.02
num_validation_samples = int(validation_ratio * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
del data, labels  # free memory

util.print_memory()


# model.add(Dense(1024, activation='tanh'))
# model.add(BatchNormalization())

#MODEL KTORY SI VIEME UPRAVOVAT PODLA VELKOSTI DATASETU
print('training model...')

model = Sequential()
model.add(Embedding(len(char_to_int) + 1, EMBEDDING_DIM, input_length=SEQUENCE_LENGTH))
model.add(LSTM(512, activation='tanh', dropout=0.25))
model.add(BatchNormalization())
model.add(Dense(512, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(len(labels_index), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

print('model summary: ')
model.summary()

# save model summary in model folder so we can reference it later when comparing models
with open(MODEL_PATH + '/summary.txt', 'w') as handle:
    model.summary(print_fn=lambda x: handle.write(x + '\n'))

# make sure we only keep the weights from the epoch with the best accuracy, rather than the last set of weights
checkpointer = ModelCheckpoint(filepath=MODEL_PATH + '/model.h5', verbose=1, save_best_only=True)
history_checkpointer = util.SaveHistoryCheckpoint(model_path=MODEL_PATH)

#trenovanie modela
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpointer, history_checkpointer])

util.copy_model_to_latest(BASE_PATH, MODEL_PATH, MODEL_NAME)

print('total time: %ds' % round(util.total_time()))
