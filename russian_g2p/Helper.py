import itertools
import os

import numpy as np
import spacy
import tensorflow as tf
import torchaudio
from transformers import Wav2Vec2FeatureExtractor

from russian_g2p.Accentor import Accentor
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme

accentor = Accentor()
transcriptor = Grapheme2Phoneme()

bad_words = {'+б+а+н+к+о+м+а+т+': 'банкома+т',
             "+к+о+л+е+с+а+х+": "коле+сах",
             "+ч+е+т+в+е+р+т+а+я+": "четверта+я",
             "+а+к+т+е+р+о+м+": "акте+ром",
             "+п+ю+р+е+": "пюре+"}

characters = [x for x in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "]

phonems = ['A', 'A0', 'A0l', 'Al', 'B', 'B0', 'B0l', 'Bl', 'D', 'D0', 'D0l',
           'DZ', 'DZ0', 'DZ0l', 'DZH', 'DZH0', 'DZH0l', 'DZHl', 'DZl', 'Dl',
           'E', 'E0', 'E0l', 'El', 'F', 'F0', 'F0l', 'Fl', 'G', 'G0', 'G0l',
           'GH', 'GH0', 'GH0l', 'GHl', 'Gl', 'I', 'I0', 'I0l', 'Il', 'J0',
           'J0l', 'K', 'K0', 'K0l', 'KH', 'KH0', 'KH0l', 'KHl', 'Kl', 'L',
           'L0', 'L0l', 'Ll', 'M', 'M0', 'M0l', 'Ml', 'N', 'N0', 'N0l', 'Nl',
           'O', 'O0', 'O0l', 'Ol', 'P', 'P0', 'P0l', 'Pl', 'R', 'R0', 'R0l',
           'Rl', 'S', 'S0', 'S0l', 'SH', 'SH0', 'SH0l', 'SHl', 'Sl', 'T', 'T0',
           'T0l', 'TS', 'TS0', 'TS0l', 'TSH', 'TSH0', 'TSH0l', 'TSHl', 'TSl',
           'Tl', 'U', 'U0', 'U0l', 'Ul', 'V', 'V0', 'V0l', 'Vl', 'Y', 'Y0',
           'Y0l', 'Yl', 'Z', 'Z0', 'Z0l', 'ZH', 'ZH0', 'ZH0l', 'ZHl', 'Zl', 'sil']

pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ",
            "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
            "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
            "SPACE"]

char_to_num = tf.keras.layers.StringLookup(
    vocabulary=characters, oov_token="")

num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

phn_to_num = tf.keras.layers.StringLookup(
    vocabulary=phonems, oov_token="")

num_to_phn = tf.keras.layers.StringLookup(
    vocabulary=phn_to_num.get_vocabulary(), oov_token="", invert=True)

tags_to_num = tf.keras.layers.StringLookup(
    vocabulary=pos_tags, oov_token="")

num_to_pos = tf.keras.layers.StringLookup(
    vocabulary=tags_to_num.get_vocabulary(), oov_token="", invert=True)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)

nlp = spacy.load("ru_core_news_lg")
DATA_PATH = r"C:\Users\LimpWinter\Documents\Projects\multitask_wav2vec\data"



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            df,
            batch_size,
            train=True,
            padding=None
    ):

        self.df = df.sort_values('duration')  # сортирока чтобы  при паддинге занимать меньше места
        self.batch_size = batch_size
        self.train = train
        self.SAMPLE_RATE = 16_000
        self.n = len(self.df)
        self.padding = padding
        self.MAX_INPUT_BATCH_SIZE = None
        self.set_max_batch_size()

    def set_max_batch_size(self):
        wav_file, sr = torchaudio.load(os.path.join(DATA_PATH, 'train', self.df['audio_filepath'].iloc[-1]))
        self.MAX_INPUT_BATCH_SIZE = len(np.array(wav_file).squeeze())
        assert isinstance(self.MAX_INPUT_BATCH_SIZE, int)

    def __get_input(self, wav_batch):
        processed_batch = feature_extractor(
            wav_batch,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors='pt',
            padding='longest',
            padding_value=0.0
            # max_length=self.MAX_INPUT_BATCH_SIZE
        )  # wav (batch_size x seq_len), mask (batch_size x seq_len)

        processed_waves = processed_batch.input_values
        attention_masks = processed_batch.attention_mask

        return processed_waves, attention_masks

    def __get_wav(self, wav_path):
        if self.train:
            # wav_file, sr = torchaudio.load(os.path.join(DATA_PATH, "train", wav_path))
            audio, sr = torchaudio.load(os.path.join(DATA_PATH, "train", wav_path))  # shape(channels,seq_len)
        else:
            audio, sr = torchaudio.load(os.path.join(DATA_PATH, 'test', 'crowd', wav_path))
        assert sr == self.SAMPLE_RATE
        return np.array(audio.squeeze(0))  # shape(seq_len,)

    def get_max_padding_size(self):
        return self.MAX_INPUT_BATCH_SIZE

    @staticmethod
    def get_pos_emb(text: str):
        doc = nlp(text.replace(" ", "  "))
        tagged = [w.pos_ for w in doc]
        return tags_to_num(tagged)

    @staticmethod
    def get_phn_emb(text: str):
        # разбиваем на слова
        text = [[word] for word in text.split(" ")]

        # ставим ударения и заменяем "сломанные" слова
        accented = accentor.do_accents(text)[0]
        accented = [bad_words.get(word) if word in bad_words else word for word in accented]
        # создаем и кодируем транскрипции
        transcription = [transcriptor.word_to_phonemes(elem) for elem in accented]
        transcription = np.array(transcription, dtype=np.ndarray)
        transcription = [np.append(lst, ['PAD']) for lst in transcription]
        transcription[-1] = transcription[-1][:-1]
        merged = list(itertools.chain.from_iterable(transcription))
        return phn_to_num(merged)

    @staticmethod
    def get_text_emb(text: str):
        emb = list(text)
        return np.array(char_to_num(emb)).tolist()  # !!!! добавил np.array

    def __pad_batch_arrays(self, batch, padding=False):
        if not padding:
            max_vec_len = [len(elem) for elem in batch]
            max_vec_len = max(max_vec_len)
        else:
            max_vec_len = self.MAX_INPUT_BATCH_SIZE
        padded_batch = []

        for elem in batch:
            elem = list(elem) + [0] * (max_vec_len - len(elem))
            padded_batch.append(elem)
        return padded_batch

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        paths_batch = batches['audio_filepath']

        text_batch = batches['text']

        x_batch = [self.__get_wav(X) for X in paths_batch]
        # print(x_batch[0].shape)
        # print(x_batch[1].shape)
        # print(np.array(x_batch).shape)
        # print(f'X_batch shape:{np.array(x_batch).shape}')
        x_data, x_mask = self.__get_input(x_batch)
        # assert len(x_batch) == 2
        # print('Processing of input complete.')
        # if self.padding == 'longest':
        # x_batch = self.__pad_batch_arrays(x_batch, padding=True)  # Padded

        y_text_batch = np.array([self.get_text_emb(y) for y in text_batch])
        # if self.padding == 'longest':
        # self.set_max_batch_size()
        y_text_batch = self.__pad_batch_arrays(y_text_batch, padding=False)

        # y_phn_batch = np.array([self.get_phn_emb(y) for y in text_batch])
        # y_phn_batch = self.__pad_batch_arrays(y_phn_batch)

        # y_pos_tags_batch = np.array([self.get_text_emb(y) for y in text_batch])
        # y_pos_tags_batch = self.__pad_batch_arrays(y_pos_tags_batch)

        # return [x_data, x_mask], y_text_batch  # tuple([y_text_batch, y_phn_batch, y_pos_tags_batch])
        return (x_data, x_mask), y_text_batch  # tuple([y_text_batch, y_phn_batch, y_pos_tags_batch])

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__get_data(batches)

    def __len__(self):
        return self.n // self.batch_size


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode('utf-8')
        output_text.append(result)
    return output_text

