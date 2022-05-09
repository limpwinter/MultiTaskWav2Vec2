import argparse
import itertools
import logging
import os
import pathlib
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import spacy
import torch
import torchaudio

from russian_g2p.Accentor import Accentor
from russian_g2p.DataHandler import DataProcessor
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme

accentor = Accentor()
transcriptor = Grapheme2Phoneme()
# 03.05.2022 (12:03)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

nlp = spacy.load("models/ru_core_news_lg")


# examples_vocab = Counter()

def get_pos_tags(text: str):
    doc = nlp(text.strip().replace(" ", "  "))
    tagged = [w.pos_ for w in doc]
    return tagged


def get_phonems(text: str):
    # разбиваем на слова
    text = [[word] for word in text.strip().split(" ")]

    # ставим ударения и заменяем "сломанные" слова
    accented = accentor.do_accents(text)[0]
    accented = [word.replace('+', '') if word.count('+') > 1 else word for word in accented]

    # создаем и кодируем транскрипции
    transcription = [transcriptor.word_to_phonemes(elem) for elem in accented]
    transcription = np.array(transcription, dtype=np.ndarray)
    transcription = [np.append(lst, ['sil']) for lst in transcription]
    transcription[-1] = transcription[-1][:-1]
    merged = list(itertools.chain.from_iterable(transcription))
    return ''.join(merged)


def preprocess_data(data_folder, new_folder_path, data_processor, data: List):
    _, path, text, duration = data
    wav_abs_path = pathlib.Path(data_folder, path)
    pkl_abs_path = Path(new_folder_path, path)
    pkl_abs_path.parent.mkdir(parents=True, exist_ok=True)
    new_data_path = pathlib.Path(pkl_abs_path).with_suffix('.pkl')
    # examples_vocab[pkl_abs_path.resolve()] += 1

    wav_vile, sr = torchaudio.load(wav_abs_path)
    wav_vile = wav_vile.squeeze(0)
    preprocessed_wav = data_processor.processor(
        wav_vile,
        sampling_rate=sr,
        return_tensors='pt'
    ).input_values[0]
    assert len(preprocessed_wav.shape) == 1

    encoded_phonems = data_processor.phonems_tokenizer(
        get_phonems(text),
        is_split_into_words=False,
        return_tensors='pt'
    ).input_ids[0]
    assert len(encoded_phonems.shape) == 1

    encoded_pos_tags = data_processor.postags_tokenizer(
        get_pos_tags(text),
        is_split_into_words=True,
        return_tensors='pt'
    ).input_ids[0]
    assert len(encoded_pos_tags.shape) == 1

    def mean_pooling(model_output):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = torch.ones_like(token_embeddings)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    encoded_input = data_processor.sencence_tokenizer(
        text,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = data_processor.sentence_encoding_model(**encoded_input)
    sentence_embeddings = mean_pooling(output)[0]

    char_labels = data_processor.char_tokenizer(
        text,
        return_tensors='pt'
    ).input_ids[0]
    # Perform pooling. In this case, mean pooling
    result = {
        'path_to_save': new_data_path,
        'input_values': preprocessed_wav,
        'text': text,
        'characters': char_labels,
        'phonems': encoded_phonems,
        'pos_tags': encoded_pos_tags,
        'sentence_embedding': sentence_embeddings,
        'duration': duration
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)

    args = parser.parse_args()
    train_data = pd.read_json(os.path.join(args.data_folder, 'train', '10min.jsonl'), lines=True)
    test_data = pd.read_json(os.path.join(args.data_folder, 'test', 'crowd', 'manifest.jsonl'), lines=True)

    train_data_path = Path(args.data_folder, 'train')
    new_train_data_path = Path(args.data_folder, 'train_pkl')
    test_data_path = Path(args.data_folder, 'test', 'crowd')
    new_test_data_path = Path(args.data_folder, 'test_pkl', 'crowd')

    new_test_data_path.parent.mkdir(parents=True, exist_ok=True)
    new_train_data_path.mkdir(parents=True, exist_ok=True)

    data_processor = DataProcessor()
    logger.info('Start train data preprocessing')
    for i, row in enumerate(train_data.values):
        if i % 50 == 0:
            info_msg = f'Train preprocessing is {(float(i) / len(train_data)) * 100:.1f}% complete.'
            logger.info(info_msg)
        preprocessed_data = preprocess_data(train_data_path, new_train_data_path, data_processor, row)
        with open(preprocessed_data['path_to_save'], 'wb') as f:
            pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # save_data(preprocessed_data)

    # for i, row in enumerate(test_data.values):
    #     if i % 50 == 0:
    #         info_msg = f'Test preprocessing is {(float(i) / len(test_data)) * 100:.1f}% complete.'
    #         logger.info(info_msg)
    #     preprocessed_data = preprocess_data(test_data_path, new_test_data_path, data_processor, row)
    # with open(preprocessed_data['path_to_save'], 'wb') as f:
    #     pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # save_data(preprocessed_data)
    # vocab_path = Path(args.data_folder, 'vocabs')
    # vocab_path.mkdir(parents=True, exist_ok=True)
    # vocab_file = vocab_path / Path('examples_vocab.pkl')
    # with open(vocab_file, 'wb') as f:
    #     pickle.dump(examples_vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('Preprocessing_data.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main()
