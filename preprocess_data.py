import argparse
import logging
import os
import pathlib
import pickle
import sys
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torchaudio

from russian_g2p.DataHandler import DataProcessor
# 29.04.2022 (17:52)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def preprocess_data(data_folder, new_data_path, data_processor, data: List):
    _, path, text, _ = data

    wav_abs_path = pathlib.Path(data_folder, path)
    pkl_abs_path = Path(new_data_path, path)
    pkl_abs_path.parent.mkdir(parents=True, exist_ok=True)

    wav_vile, sr = torchaudio.load(wav_abs_path)
    wav_vile = wav_vile.squeeze(0)
    preprocessed_wav = data_processor.processor(
        wav_vile,
        sampling_rate=sr,
        return_tensors='pt'
    ).input_values[0]
    new_data_path = pathlib.Path(pkl_abs_path).with_suffix('.pkl')

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
    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(output)[0]
    result = {
        'path_to_save': new_data_path,
        'data': preprocessed_wav,
        'embedding': sentence_embeddings}
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

    for i, row in enumerate(test_data.values):
        if i % 50 == 0:
            info_msg = f'Test preprocessing is {(float(i) / len(test_data)) * 100:.1f}% complete.'
            logger.info(info_msg)
        preprocessed_data = preprocess_data(test_data_path, new_test_data_path, data_processor, row)
        with open(preprocessed_data['path_to_save'], 'wb') as f:
            pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # save_data(preprocessed_data)


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
