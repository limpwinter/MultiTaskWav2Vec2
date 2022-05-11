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
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from transformers import Wav2Vec2Model
from MultitaskWav2vec2 import MultitaskWav2vecModel
from russian_g2p.DataHandler import (DataProcessor,
                                     GolosDataset,
                                     DataCollator
                                     )
# 11.05.2022 (23:24)

'''
ver 0:
no loss weight
no sent emb loss
2-phonems
10-pos tags
23-chars
'''
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
tb = SummaryWriter('runs/multitask-10min-multistep_lr-adamw-pkl-no_sent_v0')


def mean_pooling(token_embeddings):
    input_mask_expanded = torch.ones_like(token_embeddings, dtype=torch.long)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def train(model, device, train_dataloader, epoch, lr_scheduler, optimizer):
    model.train()
    running_loss = 0
    running_char_loss = 0
    running_pos_tags_loss = 0
    running_phonem_loss = 0
    # running_sent_emb_loss = 0
    iterable_dataloder = iter(train_dataloader)
    n = len(train_dataloader)
    for idx in range(len(train_dataloader)):
        start_loading_time = time.time()
        batch = next(iterable_dataloder)
        batch = {k: v.to(device) for k, v in batch.items()}
        data = batch['input_values']
        mask = batch['attention_mask']
        pos_tag_labels = batch['pos_tag_labels']
        char_labels = batch['char_labels']
        phonem_labels = batch['phonem_labels']
        # sent_emb_labels = batch['sentence_embedding']
        end_loading_time = time.time()

        labels = {'char_labels': char_labels,
                  'phonem_labels': phonem_labels,
                  'pos_tag_labels': pos_tag_labels}

        start_forward_time = time.time()
        logits, losses = model(data,
                               attention_mask=mask,
                               labels=labels
                               )
        end_forward_time = time.time()

        del data, mask, char_labels, pos_tag_labels, phonem_labels

        char_loss = losses['char_loss']
        pos_tags_loss = losses['pos_tag_loss']
        phonem_loss = losses['phonem_loss']
        loss = char_loss + pos_tags_loss + phonem_loss

        running_loss += loss.item()
        running_char_loss += char_loss.item()
        running_pos_tags_loss += pos_tags_loss.item()
        running_phonem_loss += phonem_loss.item()
        # running_sent_emb_loss += sent_emb_loss.item()
        info_msg = f'Epoch: {epoch} (Iteration: {idx} with loss: {loss.item()})'
        logger.info(info_msg)

        tb.add_scalar('Loss/overall/train', loss.item(), epoch * n + idx)
        tb.add_scalar('Loss/character/train', char_loss.item(), epoch * n + idx)
        tb.add_scalar('Loss/pos_tags/train', pos_tags_loss.item(), epoch * n + idx)
        tb.add_scalar('Loss/phonems/train', phonem_loss.item(), epoch * n + idx)
        # tb.add_scalar('Loss/sent_embedding/train', sent_emb_loss.item(), epoch * n + idx)
        tb.add_scalar('Time/forward/train', float(end_forward_time - start_forward_time), epoch * n + idx)
        tb.add_scalar('Time/data gen/train', float(end_loading_time - start_loading_time), epoch * n + idx)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tb.add_scalar('Learning Rate', lr_scheduler.get_lr()[-1], epoch)
    lr_scheduler.step()
    tb.add_scalar('Average Loss/characters/train', running_char_loss / len(train_dataloader), epoch)
    tb.add_scalar('Average Loss/POS Tags/train', running_pos_tags_loss / len(train_dataloader), epoch)
    # tb.add_scalar('Average Loss/sentence_embeddings/train', running_sent_emb_loss / len(train_dataloader), epoch)
    tb.add_scalar('Average Loss/phonems/train', running_phonem_loss / len(train_dataloader), epoch)
    tb.add_scalar('Average Loss/overall/train', running_loss / len(train_dataloader), epoch)
    logger.info(f'Average loss on {epoch} epoch: {running_loss / len(train_dataloader)}')


def test(model, device, test_dataloader, processor, epoch):
    model.eval()
    running_wer = 0
    running_cer = 0
    running_loss = 0
    running_char_loss = 0
    running_pos_tags_loss = 0
    running_phonem_loss = 0
    # running_sent_emb_loss = 0
    running_data_gen_time = 0
    running_forward_time = 0
    iterable_dataloder = iter(test_dataloader)
    n = len(test_dataloader)
    with torch.no_grad():
        printed_results = []
        for idx in range(len(test_dataloader)):
            start_loading_time = time.time()
            batch = next(iterable_dataloder)
            batch = {k: v.to(device) for k, v in batch.items()}
            data = batch['input_values']
            mask = batch['attention_mask']
            pos_tag_labels = batch['pos_tag_labels']
            char_labels = batch['char_labels']
            phonem_labels = batch['phonem_labels']
            # sent_emb_labels = batch['sentence_embedding']
            end_loading_time = time.time()

            labels = {'char_labels': char_labels,
                      'phonem_labels': phonem_labels,
                      'pos_tag_labels': pos_tag_labels}

            start_forward_time = time.time()
            logits, losses = model(data,
                                   attention_mask=mask,
                                   labels=labels
                                   )
            end_forward_time = time.time()

            del data, mask, pos_tag_labels, phonem_labels
            char_loss = losses['char_loss']
            pos_tags_loss = losses['pos_tag_loss']
            phonem_loss = losses['phonem_loss']
            loss = char_loss + pos_tags_loss + phonem_loss
            running_loss += loss.item()
            running_char_loss += char_loss.item()
            running_pos_tags_loss += pos_tags_loss.item()
            running_phonem_loss += phonem_loss.item()
            # running_sent_emb_loss += sent_emb_loss.item()

            tb.add_scalar('Time/data gen/test', float(end_loading_time - start_loading_time), epoch * n + idx)
            tb.add_scalar('Time/forward/test', float(end_forward_time - start_forward_time), epoch * n + idx)

            logits = torch.argmax(logits, dim=-1).cpu().detach()
            pred_ids = processor.decode_batch_predictions(logits)
            char_labels[char_labels == -100] = processor.characters_char_map['<pad>']
            label_str = processor.decode_batch_predictions(char_labels, group_tokens=False)
            printed_results.append(
                (label_str[0],
                 pred_ids[0])
            )
            word_error_rate = wer(label_str, pred_ids)
            char_error_rate = cer(label_str, pred_ids)
            running_cer += char_error_rate
            running_wer += word_error_rate
            running_forward_time += float(end_forward_time - start_forward_time)
            running_data_gen_time += float(end_loading_time - start_loading_time)

        if len(printed_results) > 5:
            printed_results = random.sample(printed_results, k=5)
            info_msg = ''
            for ground_truth, pred_str in printed_results:
                info_msg += f'\nPREDICTED: {pred_str}\n' \
                            f'TARGET: {ground_truth}' \
                            f'{"-" * 100}'
            logger.info(info_msg)
        tb.add_scalar('WER/test', running_wer / len(test_dataloader), epoch)
        tb.add_scalar('CER/test', running_cer / len(test_dataloader), epoch)
        tb.add_scalar('Average Loss/overall/test', running_loss / n, epoch)
        tb.add_scalar('Average Loss/character/test', running_char_loss / n, epoch)
        tb.add_scalar('Average Loss/pos_tags/test', running_pos_tags_loss / n, epoch)
        tb.add_scalar('Average Loss/phonems/test', running_phonem_loss / n, epoch)
        logger.info(
            f'Average test loss on {epoch} epoch: {running_loss / n}')
        logger.info(
            f'Average forward time during test on {epoch} epoch: {running_forward_time / n}')
        logger.info(
            f'Average data gen time during test on {epoch} epoch: {running_data_gen_time / n}')


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

    num_training_steps = NUM_EPOCHS * len(train_dataloader)
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
            model=multitask_wav2vec2,
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
    file_handler = logging.FileHandler('Multitask_Wav2Vec2_v0.log')

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main()
