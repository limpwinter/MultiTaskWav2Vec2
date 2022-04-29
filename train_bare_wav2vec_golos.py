import os.path
import random
import warnings
from pathlib import Path

import argparse
import logging
import pandas as pd
import torch
import sys
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from tqdm import trange
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import List
import numpy as np
import random
import time
# 20.04.2022(22:50)

from russian_g2p.DataHandler import (DataGenerator,
                                     DataProcessor,
                                     DataCollator,
                                     GolosDataset
                                     )

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
tb = SummaryWriter()


def levenshtein(seq1: List[str], seq2: List[str]) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.int32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


def wer(predicted, target):
    error = []
    for i in range(len(target)):
        y_true = target[i].split()
        y_pred = predicted[i].split()
        error.append(levenshtein(y_true, y_pred) / len(y_true))
    res = np.array(error).mean()
    return res


def cer(predicted, target):
    error = []
    for i in range(len(target)):
        y_true = target[i]
        y_pred = predicted[i]
        error.append(levenshtein(y_true, y_pred) / len(y_true))
    res = np.array(error).mean()
    return res


def train(model, device, train_data_generator, epoch, optimizer):
    model.train()
    running_loss = 0
    running_forward_time = 0
    running_data_gen_time = 0
    for i in range(len(train_data_generator)):
        start_gen = time.time()
        (data, mask), labels = train_data_generator[i]
        end_gen = time.time()
        data, mask, labels = data.to(device), mask.to(device), labels.to(device)
        start_forward = time.time()
        loss = model(
            data,
            attention_mask=mask,
            labels=labels
        ).loss
        end_forward = time.time()

        del data, mask, labels
        tb.add_scalar('Loss/train', loss.item(), epoch)
        tb.add_scalar('Time/train/forward', float(end_forward - start_forward), epoch)
        tb.add_scalar('Time/train/data gen', float(end_gen - start_gen), epoch)
        running_loss += loss.item()
        running_data_gen_time += (end_gen - start_gen)
        running_forward_time += (end_forward - start_forward)
        info_msg = f'Epoch: {epoch} (Iteration: {i} with loss: {loss.item()}'
        logger.info(info_msg)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
    logger.info(f'Average forward time on {epoch} epoch: {running_forward_time / len(train_data_generator)}')
    logger.info(f'Average data gen time on {epoch} epoch: {running_data_gen_time / len(train_data_generator)}')
    logger.info(f'Average loss on {epoch} epoch: {running_loss / len(train_data_generator)}')


def test(model, device, data_generator, epoch):
    model.eval()
    running_wer = 0
    running_cer = 0
    running_loss = 0
    with torch.no_grad():
        printed_results = []
        for i in trange(len(data_generator)):
            (data, mask), labels = data_generator[i]
            data, mask = data.to(device), mask.to(device)

            w2v2_output = model(
                data,
                mask,
                labels=labels.to(device)
            )
            logits = w2v2_output.logits.detach().cpu()
            loss = w2v2_output.loss
            del w2v2_output, data, mask
            tb.add_scalar('Loss/test', loss.item(), epoch)
            running_loss += loss.item()
            logits = torch.argmax(logits, dim=-1)
            pred_ids = data_generator.processor.decode_batch_predictions(logits)
            labels[labels == -100] = data_generator.data_processor.characters_char_map['<pad>']
            label_str = data_generator.processor.decode_batch_predictions(labels, group_tokens=False)

            printed_results.append(
                (label_str[0],
                 pred_ids[0])
            )
            word_error_rate = wer(pred_ids, label_str)
            char_error_rate = cer(pred_ids, label_str)
            running_wer += word_error_rate
            running_cer += char_error_rate
        if len(printed_results) > 5:
            printed_results = random.sample(printed_results, k=5)
            info_msg = ''
            for ground_truth, pred_str in printed_results:
                info_msg += f'\nPREDICTED: {pred_str}\n' \
                            f'TARGET: {ground_truth}'
            logger.info(info_msg)
        # tb.add_scalar('Loss/test', running_loss / len(data_generator), epoch)
        logger.info(f'Test loss on {epoch} epoch: {running_loss / len(data_generator)}')
        tb.add_scalar('WER/test', running_wer / len(data_generator), epoch)
        tb.add_scalar('CER/test', running_cer / len(data_generator), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('pretrained_wav2vec2_path', type=str)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('checkpoint_path', type=str)
    args = parser.parse_args()

    logger.info(f'Data folder: {args.data_folder}')

    NUM_EPOCHS = 4000
    RANDOM_STATE = 256
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    logger.info('Reading Data...')

    train_data = pd.read_json(os.path.join(args.data_folder, 'train', '10hours.jsonl'), lines=True)
    test_data = pd.read_json(os.path.join(args.data_folder, 'test', 'crowd', 'manifest.jsonl'), lines=True)

    test_data = test_data[test_data['text'] != ' ']
    train_data = train_data[train_data['text'] != ' ']
    train_data = train_data[train_data['duration'] <= 6]  # removing long audio during training

    test_data_len = min(len(test_data),
                        len(train_data))

    test_data = test_data[:test_data_len].sort_values('duration')  # train_len >= test_len

    logger.info(f'Training examples: {len(train_data)}')
    logger.info(f'Testing examples: {len(test_data)}')

    data_processor = DataProcessor()
    datacollator = DataCollator(processor=data_processor.processor, padding=True)
    train_golos = GolosDataset(
        df=train_data,
        data_path=Path(args.data_folder, 'train_pkl'),
        logger=logger
    )
    test_golos = GolosDataset(
        df=test_data,
        data_path=Path(args.data_folder, 'test_pkl'),
        logger=logger
    )
    train_dataloader = DataLoader(
        train_golos,
        args.batch_size,
        shuffle=True,
        collate_fn=datacollator
    )
    test_dataloader = DataLoader(
        test_golos,
        args.batch_size,
        shuffle=False,
        collate_fn=datacollator
    )

    logger.info('Creating data generators...')
    train_data_gen = DataGenerator(
        train_data,
        data_path=args.data_folder,
        batch_size=args.batch_size,
        processor=data_processor,
        logger=logger,
        train=True,
        random_state=RANDOM_STATE
    )

    test_data_gen = DataGenerator(
        test_data,
        data_path=args.data_folder,
        batch_size=args.batch_size,
        processor=data_processor,
        train=False,
        logger=logger,
        random_state=RANDOM_STATE
    )

    logger.info('Loading tensorboard')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info(f'Device: {device}')
    logger.info('Loading Model...')

    model = Wav2Vec2ForCTC.from_pretrained(
        args.pretrained_wav2vec2_path,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=data_processor.characters_char_map['<pad>'],
        vocab_size=len(data_processor.processor.tokenizer),
        ctc_zero_infinity=True
    ).to(device)
    model.gradient_checkpointing_enable()
    # model.config.ctc_zero_infinity = True
    model.freeze_feature_extractor()

    logger.info(f'len of tokenizer: {len(data_processor.processor.tokenizer)}\n'
                f'len of vocab_size: {len(data_processor.characters_char_map)}')
    num_training_steps = NUM_EPOCHS * len(train_data_gen)
    logger.info(f'Num of training steps: {num_training_steps}')
    # progress_bar = tqdm(range(num_training_steps))

    optimizer = torch.optim.AdamW(model.parameters())  # lr=1e-3
    # 0.3162 * 0.3162 = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20, 40], gamma=0.3162)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    next_epoch = checkpoint['epoch'] + 1

    for epoch in range(next_epoch, NUM_EPOCHS + next_epoch):
        logger.info(f'Start training on {epoch} epoch.')
        train(model, device, train_data_gen, epoch, optimizer)
        logger.info(f'Start testing on {epoch} epoch.')
        test(model, device, test_data_gen, epoch)
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'models/checkpoints/baseline/baseline_{epoch}_epoch')
    tb.close()
    logger.info('Saving model')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f'models/checkpoints/baseline/baseline_final')
    logger.info('Done')


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('Multitask_Wav2Vec2_v3.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main()
