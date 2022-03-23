import itertools
import json
import os

import numpy as np
import spacy
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from russian_g2p.Accentor import Accentor
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme

accentor = Accentor()
transcriptor = Grapheme2Phoneme()

characters = [x for x in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "]

phonems = ["A", "A0", "A0l", "Al", "B", "B0", "B0l", "Bl", "D", "D0", "D0l",
           "DZ", "DZ0", "DZ0l", "DZH", "DZH0", "DZH0l", "DZHl", "DZl", "Dl",
           "E", "E0", "E0l", "El", "F", "F0", "F0l", "Fl", "G", "G0", "G0l",
           "GH", "GH0", "GH0l", "GHl", "Gl", "I", "I0", "I0l", "Il", "J0",
           "J0l", "K", "K0", "K0l", "KH", "KH0", "KH0l", "KHl", "Kl", "L",
           "L0", "L0l", "Ll", "M", "M0", "M0l", "Ml", "N", "N0", "N0l", "Nl",
           "O", "O0", "O0l", "Ol", "P", "P0", "P0l", "Pl", "R", "R0", "R0l",
           "Rl", "S", "S0", "S0l", "SH", "SH0", "SH0l", "SHl", "Sl", "T", "T0",
           "T0l", "TS", "TS0", "TS0l", "TSH", "TSH0", "TSH0l", "TSHl", "TSl",
           "Tl", "U", "U0", "U0l", "Ul", "V", "V0", "V0l", "Vl", "Y", "Y0",
           "Y0l", "Yl", "Z", "Z0", "Z0l", "ZH", "ZH0", "ZH0l", "ZHl", "Zl", "sil"]

pos_tags = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ",
            "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
            "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
            "SPACE"]

nlp = spacy.load("ru_core_news_lg")
DATA_PATH = r"C:\Users\LimpWinter\Documents\Projects\multitask_wav2vec\data"


class DataProcessor:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        self.characters = characters
        self.phonems = phonems
        self.pos_tags = pos_tags

        self.characters_char_map = {v: k for k, v in enumerate(self.characters + ['[UNK]', '[PAD]'])}
        self.characters_index_map = {v: k for k, v in self.characters_char_map.items()}
        self.char_pad_token_id = self.characters_char_map['[PAD]']
        print("char pad token id:", self.char_pad_token_id)

        self.phonems_phn_map = {v: k for k, v in enumerate(self.phonems + ['[UNK]', '[PAD]'])}
        self.phonems_index_map = {v: k for k, v in self.phonems_phn_map.items()}
        self.phn_pad_token_id = self.phonems_phn_map['[PAD]']
        print("phn pad token id", self.phn_pad_token_id)

        self.postags_tag_map = {v: k for k, v in enumerate(self.pos_tags + ['[UNK]', '[PAD]'])}
        self.postags_index_map = {k: v for k, v in self.postags_tag_map.items()}
        self.postag_pad_token_id = self.postags_tag_map['[PAD]']
        print("postag pad token id", self.postag_pad_token_id)

        with open('vocabs/char_vocab.json', 'w') as vocab_file:
            json.dump(self.characters_char_map, vocab_file)
        with open('vocabs/phonems_vocab.json', 'w') as vocab_file:
            json.dump(self.phonems_phn_map, vocab_file)
        with open('vocabs/postags_vocab.json', 'w') as vocab_file:
            json.dump(self.postags_tag_map, vocab_file)

        self.char_tokenizer = Wav2Vec2CTCTokenizer("vocabs/char_vocab.json",
                                                   unk_token="[UNK]",
                                                   pad_token="[PAD]",
                                                   word_delimiter_token=" ")
        self.phonems_tokenizer = Wav2Vec2CTCTokenizer("vocabs/phonems_vocab.json",
                                                      unk_token="[UNK]",
                                                      pad_token="[PAD]",
                                                      word_delimiter_token="sil")
        self.postags_tokenizer = Wav2Vec2CTCTokenizer("vocabs/postags_vocab.json",
                                                      unk_token="[UNK]",
                                                      pad_token="[PAD]",
                                                      word_delimiter_token="SPACE")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                          sampling_rate=16000,
                                                          padding_value=0.0,
                                                          do_normalize=True,
                                                          return_attention_mask=True)

    def decode_batch_predictions(self, pred):
        return self.tokenizer.batch_decode(pred)


def pad_batch_arrays(batch):  # Устаревший
    max_vec_len = [len(elem) for elem in batch]
    max_vec_len = max(max_vec_len)
    padded_batch = []
    for elem in batch:
        elem = list(elem) + [-100] * (max_vec_len - len(elem))
        padded_batch.append(elem)
    return torch.Tensor(padded_batch)


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
            self,
            df,
            batch_size,
            train=True,
            padding=True
    ):

        self.df = df.sort_values('duration')  # сортирока чтобы  при паддинге занимать меньше места
        self.batch_size = batch_size
        self.train = train  # Потом лучше вместо train передавать путь к файлам
        self.SAMPLE_RATE = 16_000
        self.data_processor = DataProcessor()
        self.padding = padding
        self.n = len(self.df)

    def __get_input(self, wav_batch):
        processed_batch = self.data_processor.feature_extractor(
            wav_batch,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors='pt',
            padding='longest',
            padding_value=0.0
        )  # wav (batch_size x seq_len), mask (batch_size x seq_len)

        processed_waves = processed_batch.input_values
        attention_masks = processed_batch.attention_mask

        return processed_waves, attention_masks

    def __get_wav(self, wav_path):
        if self.train:
            audio, sr = torchaudio.load(os.path.join(DATA_PATH, "train", wav_path))  # shape(channels=1,seq_len)
        else:
            audio, sr = torchaudio.load(os.path.join(DATA_PATH, 'test', 'crowd', wav_path))
        assert sr == self.SAMPLE_RATE
        return np.array(audio.squeeze(0))  # shape(seq_len,)

    @staticmethod
    def get_pos_tags(text: str):
        doc = nlp(text.replace(" ", "  "))
        tagged = [w.pos_ for w in doc]
        return tagged

    @staticmethod
    def get_phonems(text: str):
        # разбиваем на слова
        text = [[word] for word in text.split(" ")]

        # ставим ударения и заменяем "сломанные" слова
        accented = accentor.do_accents(text)[0]
        # accented = [bad_words.get(word) if word in bad_words else word for word in accented]
        accented = [word.replace('+', '') if word.count('+') > 1 else word for word in accented]

        # создаем и кодируем транскрипции
        transcription = [transcriptor.word_to_phonemes(elem) for elem in accented]
        transcription = np.array(transcription, dtype=np.ndarray)
        transcription = [np.append(lst, ['sil']) for lst in transcription]
        transcription[-1] = transcription[-1][:-1]
        merged = list(itertools.chain.from_iterable(transcription))
        # return  self.emb_proc.phn_to_int(merged)
        print(''.join(merged))
        return ''.join(merged)
        # np.array(merged).tofile('merged.txt', sep=' ', format='%s')
        # return self.data_processor.phonems_tokenizer(merged).input_ids

    def __get_data(self, batches):
        # Generates data containing batch_size samples

        paths_batch = batches['audio_filepath']
        text_batch = batches['text']

        x_batch = [self.__get_wav(X) for X in paths_batch]
        x_data, x_mask = self.__get_input(x_batch)  # padding with Wav2Vec2FeatureExtractor

        y_text_batch = self.data_processor.char_tokenizer(list(text_batch),
                                                          padding=self.padding,
                                                          return_tensors='pt')
        y_text_batch = y_text_batch['input_ids'].masked_fill(y_text_batch.attention_mask.ne(1), -100)

        y_phn_batch = [self.get_phonems(y) for y in text_batch]  # Using Accentor и Grapheme2Phoneme
        y_phn_batch = self.data_processor.phonems_tokenizer(y_phn_batch,
                                                            is_split_into_words=False,
                                                            padding=True,
                                                            return_tensors='pt')  # Encoding with tokenizer
        # y_phn_batch = pad_batch_arrays(y_phn_batch)
        y_phn_batch = y_phn_batch['input_ids'].masked_fill(y_phn_batch.attention_mask.ne(1), -100)

        y_pos_tags_batch = [self.get_pos_tags(y) for y in text_batch]
        y_pos_tags_batch = self.data_processor.postags_tokenizer(y_pos_tags_batch,
                                                                 is_split_into_words=True,
                                                                 padding=True,
                                                                 return_tensors='pt')
        y_pos_tags_batch = y_pos_tags_batch['input_ids'].masked_fill(y_pos_tags_batch.attention_mask.ne(1), -100)

        return (x_data, x_mask), (y_text_batch, y_phn_batch, y_pos_tags_batch)

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__get_data(batches)

    def __len__(self):
        return self.n // self.batch_size
