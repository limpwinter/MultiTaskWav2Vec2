import argparse
import logging
import os.path
import random
import sys
import time
import warnings
from typing import List
from pathlib import Path
from jiwer import cer, wer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from transformers import Wav2Vec2Model
from MultitaskWav2vec2 import MultitaskWav2vecModel
from russian_g2p.DataHandler import (DataGenerator,
                                     DataProcessor
                                     )

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
tb = SummaryWriter('runs/multitask-10h-multistep_lr-adamw-pkl-no_sent')


def mean_pooling(token_embeddings):
    input_mask_expanded = torch.ones_like(token_embeddings, dtype=torch.long)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# def levenshtein(seq1: List[str], seq2: List[str]) -> float:
#     size_x = len(seq1) + 1
#     size_y = len(seq2) + 1
#     matrix = np.zeros((size_x, size_y), dtype=np.int32)
#     for x in range(size_x):
#         matrix[x, 0] = x
#     for y in range(size_y):
#         matrix[0, y] = y
#     for x in range(1, size_x):
#         for y in range(1, size_y):
#             if seq1[x - 1] == seq2[y - 1]:
#                 matrix[x, y] = min(
#                     matrix[x - 1, y] + 1,
#                     matrix[x - 1, y - 1],
#                     matrix[x, y - 1] + 1
#                 )
#             else:
#                 matrix[x, y] = min(
#                     matrix[x - 1, y] + 1,
#                     matrix[x - 1, y - 1] + 1,
#                     matrix[x, y - 1] + 1
#                 )
#     return float(matrix[size_x - 1, size_y - 1])
#
#
# def wer(predicted, target):
#     error = []
#     for i in range(len(target)):
#         y_true = target[i].split()
#         y_pred = predicted[i].split()
#         error.append(levenshtein(y_true, y_pred) / len(y_true))
#     res = np.array(error).mean()
#     return res
#
#
# def cer(predicted, target):
#     error = []
#     for i in range(len(target)):
#         y_true = target[i]
#         y_pred = predicted[i]
#         error.append(levenshtein(y_true, y_pred) / len(y_true))
#     res = np.array(error).mean()
#     return res


def train(model, device, train_dataloader, epoch, lr_scheduler, optimizer):
    model.train()
    running_loss = 0
    running_char_loss = 0
    running_pos_tags_loss = 0
    running_sent_emb_loss = 0
    for i, batch in enumerate(train_dataloader):
        # start_loading_data = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}
        data = batch['input_values']
        mask = batch['attention_mask']
        # sent_emb_labels = batch['sentence_embedding']
        pos_tags_labels = batch['pos_tags_labels']
        char_labels = batch['labels']

        data, mask = data.to(device), mask.to(device)
        char_labels, pos_tags_labels, sent_emb_labels = char_labels.to(device), pos_tags_labels.to(
            device), sent_emb_labels.to(device)
        labels = char_labels, pos_tags_labels, sent_emb_labels
        start_forward = time.time()
        logits, losses = model(data,
                               mask,
                               labels
                               )
        end_forward = time.time()

        del data, mask, char_labels, pos_tags_labels, sent_emb_labels
        char_loss, pos_tags_loss, sent_emb_loss = losses
        loss = char_loss + pos_tags_loss + sent_emb_loss
        running_loss += loss.item()
        tb.add_scalar('Loss/train/overall', loss.item(), epoch)
        tb.add_scalar('Loss/train/character', char_loss.item(), epoch)
        tb.add_scalar('loss/train/pos_tags', pos_tags_loss.item(), epoch)
        tb.add_scalar('Loss/train/sent_embedding', sent_emb_loss, epoch)
        tb.add_scalar('Time/train/forward', float(end_forward - start_forward), epoch)
        running_char_loss += char_loss.item()
        running_pos_tags_loss += pos_tags_loss.item()
        running_sent_emb_loss += sent_emb_loss.item()
        info_msg = f'Epoch: {epoch} (Iteration: {i} with loss: {loss.item()})'
        logger.info(info_msg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tb.add_scalar('Learning Rate', lr_scheduler.get_lr()[-1], epoch)
    lr_scheduler.step()
    tb.add_scalar('Average Loss/train/characters', running_char_loss / len(train_dataloader), epoch)
    tb.add_scalar('Average Loss/train/POS Tags', running_pos_tags_loss / len(train_dataloader), epoch)
    tb.add_scalar('Average Loss/train/sentence_embeddings', running_sent_emb_loss / len(train_dataloader), epoch)
    tb.add_scalar('Average Loss/train/overall', running_loss / len(train_dataloader), epoch)
    logger.info(f'Average loss on {epoch} epoch: {running_loss / len(train_dataloader)}')


def test(model, device, test_dataloader, processor, epoch):
    model.eval()
    running_wer = 0
    running_cer = 0
    running_loss = 0
    running_char_loss = 0
    running_pos_tags_loss = 0
    running_sent_emb_loss = 0
    with torch.no_grad():
        printed_results = []
        for i, batch in enumerate(test_dataloader):
            # (data, mask), (char_labels, pos_tags_labels, sent_emb_labels) = test_dataloader[i]
            data = batch['input_values']
            mask = batch['attention_mask']
            sent_emb_labels = batch['sentence_embeddings']
            pos_tags_labels = batch['pos_tags']
            char_labels = batch['labels']

            data, mask = data.to(device), mask.to(device)
            char_labels, = char_labels.to(device)
            pos_tags_labels = pos_tags_labels.to(device)
            sent_emb_labels = sent_emb_labels.to(device)
            labels = char_labels, pos_tags_labels, sent_emb_labels

            logits, losses = model(
                data,
                mask,
                labels
            )

            del data, mask, pos_tags_labels, sent_emb_labels
            char_loss, pos_tags_loss, sent_emb_loss = losses
            loss = char_loss + pos_tags_loss + sent_emb_loss
            tb.add_scalar('Loss/test/overall', loss.item(), epoch)
            tb.add_scalar('Loss/test/character', char_loss.item(), epoch)
            tb.add_scalar('loss/test/pos_tags', pos_tags_loss.item(), epoch)
            tb.add_scalar('Loss/test/sent_embedding', sent_emb_loss, epoch)
            running_loss += loss.item()
            running_char_loss += char_loss.item()
            running_pos_tags_loss += pos_tags_loss.item()
            running_sent_emb_loss += sent_emb_loss.item()
            logits = torch.argmax(logits, dim=-1).cpu().detach()
            pred_ids = test_dataloader.data_processor.decode_batch_predictions(logits)
            char_labels[char_labels == -100] = test_dataloader.data_processor.characters_char_map['<pad>']
            label_str = test_dataloader.data_processor.decode_batch_predictions(char_labels, group_tokens=False)
            printed_results.append(
                (label_str,
                 pred_ids)
            )
            word_error_rate = wer(pred_ids, label_str)
            char_error_rate = cer(pred_ids, label_str)
            running_cer += char_error_rate
            running_wer += word_error_rate
        if len(printed_results) > 5:
            printed_results = random.sample(printed_results, k=5)
            info_msg = ''
            for ground_truth, pred_str in printed_results:
                info_msg += f'\nPREDICTED: {pred_str}\n' \
                            f'TARGET: {ground_truth}'
            logger.info(info_msg)
        tb.add_scalar('WER/test', running_wer / len(test_dataloader), epoch)
        tb.add_scalar('CER/test', running_cer / len(test_dataloader), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('pretrained_wav2vec2_path', type=str)
    parser.add_argument('batch_size', type=int)
    args = parser.parse_args()
    logger.info(f'Data path: {args.data_folder}')
    # TRAIN_DATA_FOLDER = Path('data/train')
    # TEST_DATA_FOLDER = Path('data/test')
    # MODEL_PATH = 'models/wav2vec2-large-xlsr-53'
    # BATCH_SIZE = 16
    NUM_EPOCHS = 4000
    RANDOM_STATE = 256
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    logger.info('Reading Data...')
    train_data = pd.read_json(os.path.join(args.data_folder, 'train', '10hours.jsonl'), lines=True)
    test_data = pd.read_json(os.path.join(args.data_folder, 'test', 'crowd', 'manifest.jsonl'), lines=True)
    test_data = test_data[test_data['text'] != ' ']
    train_data = train_data[train_data['text'] != ' ']
    train_data = train_data[train_data['duration'] <= 6]

    test_data_len = min(len(test_data), len(train_data))

    test_data = test_data[:test_data_len].sort_values('duration')  # train_len >= test_len

    logger.info(f'Training examples: {len(train_data)}')
    logger.info(f'Testing examples: {len(test_data)}')

    data_processor = DataProcessor()

    logger.info('Creating dataloaders...')

    datacollator = DataCollator(processor=data_processor.processor, padding=True)
    train_golos = GolosDataset(
        df=train_data,
        data_path=Path(args.data_folder, 'train_pkl'),
        logger=logger
    )
    test_golos = GolosDataset(
        df=test_data,
        data_path=Path(args.data_folder, 'test_pkl', 'crowd'),
        logger=logger
    )
    logger.info('Creating dataloaders...')
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

    logger.info('Loading tensorboard')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info(f'Device: {device}')
    logger.info('Loading Model...')

    multitask_wav2vec2 = MultitaskWav2vecModel(args.pretrained_wav2vec2_path,
                                               data_processor.characters_char_map,
                                               data_processor.phonems_phn_map,
                                               data_processor.pos_tags_tag_map
                                               ).to(device)
    #
    # multitask_wav2vec2.load_state_dict(torch.load('models/checkpoints/MultitaskWav2vec_35_epoch'))
    num_training_steps = NUM_EPOCHS * len(train_data_gen)
    logger.info(f'Num of training steps: {num_training_steps}')
    # progress_bar = tqdm(range(num_training_steps))
    optimizer = AdamW(multitask_wav2vec2.parameters())
    # 0.3162 * 0.3162 = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 150, 220], gamma=0.3162)
    next_epoch = 1

    for epoch in range(next_epoch, NUM_EPOCHS + next_epoch):
        logger.info(f'Start training on {epoch} epoch.')
        train(
            model=multitask_wav2vec2,
            device=device,
            train_dataloader=train_dataloader,  # TODO Change gen to loader
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            epoch=epoch,
        )
        logger.info(f'Start testing on {epoch} epoch.')
        test(
            model=model,
            device=device,
            test_dataloader=test_dataloader,
            processor=data_processor,
            epoch=epoch)
        if epoch % 10 == 0:
            Path('models/checkpoints/configuration_v0').mkdir(
                parents=True,
                exist_ok=True
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': multitask_wav2vec2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ls_scheduler_state_dict': lr_scheduler.state_dict()},
                f'models/checkpoints/configuration_v0/MultitaskWav2vec_{epoch}_epoch')
    tb.close()
    logger.info('Saving model')
    torch.save({
        'epoch': NUM_EPOCHS + next_epoch - 1,
        'model_state_dict': multitask_wav2vec2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict()},
        f'models/checkpoints/configuration_v0/MultitaskWav2vec_final')
    logger.info('Done')


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('Multitask_Wav2Vec2_v1.log')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main()
