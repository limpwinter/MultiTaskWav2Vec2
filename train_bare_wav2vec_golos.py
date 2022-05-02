import argparse
import logging
import os.path
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2ForCTC

from russian_g2p.DataHandler import (DataProcessor,
                                     DataCollator,
                                     GolosDataset
                                     )

# 01.05.2022 (15:20)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
tb = SummaryWriter('runs/baseline-10h-multistep_lr-adamw-pkl')


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


def train(model, device, train_dataloader, lr_scheduler, epoch, optimizer):
    model.train()
    running_loss = 0
    running_forward_time = 0
    running_data_gen_time = 0
    iterable_dataloder = iter(train_dataloader)
    for idx in range(len(train_dataloader)):
        start_loading_data = time.time()
        batch = next(iterable_dataloder)
        data = batch['input_values']
        mask = batch['attention_mask']
        labels = batch['char_labels']
        end_loading_data = time.time()

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
        tb.add_scalar('Time/forward/train', float(end_forward - start_forward), epoch)
        tb.add_scalar('Time/data gen/train', float(end_loading_data - start_loading_data), epoch)
        running_loss += loss.item()
        running_data_gen_time += (end_loading_data - start_loading_data)
        running_forward_time += (end_forward - start_forward)
        info_msg = f'Epoch: {epoch} (Iteration: {idx} with loss: {loss.item()}'
        logger.info(info_msg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    lr_scheduler.step()
    tb.add_scalar('Learning Rate', lr_scheduler.get_lr()[-1], epoch)
    tb.add_scalar('Average Loss/train', running_loss / len(train_dataloader), epoch)

    logger.info(f'Average forward time on {epoch} epoch: {running_forward_time / len(train_dataloader)}')
    logger.info(f'Average data gen time on {epoch} epoch: {running_data_gen_time / len(train_dataloader)}')
    logger.info(f'Average loss on {epoch} epoch: {running_loss / len(train_dataloader)}')


def test(model, device, test_dataloader, processor, epoch):
    model.eval()
    running_wer = 0
    running_cer = 0
    running_loss = 0
    running_forward_time = 0
    running_data_gen_time = 0
    iterable_dataloder = iter(test_dataloader)
    with torch.no_grad():
        printed_results = []
        for idx in range(len(test_dataloader)):
            start_loading_data = time.time()
            batch = next(iterable_dataloder)
            data = batch['input_values']
            mask = batch['attention_mask']
            labels = batch['char_labels']
            end_loading_data = time.time()

            data, mask = data.to(device), mask.to(device)

            start_forward_time = time.time()
            w2v2_output = model(
                data,
                mask,
                labels=labels.to(device)
            )
            end_forward_time = time.time()

            logits = w2v2_output.logits.detach().cpu()
            loss = w2v2_output.loss
            del w2v2_output, data, mask
            tb.add_scalar('Loss/test', loss.item(), epoch)
            tb.add_scalar('Time/data gen/test', float(end_loading_data - start_loading_data))
            tb.add_scalar('Time/forward/test', float(end_forward_time - start_forward_time))
            running_loss += loss.item()
            logits = torch.argmax(logits, dim=-1)
            pred_ids = processor.decode_batch_predictions(logits)
            labels[labels == -100] = processor.characters_char_map['<pad>']
            label_str = processor.decode_batch_predictions(labels, group_tokens=False)

            printed_results.append(
                (label_str[0],
                 pred_ids[0])
            )
            word_error_rate = wer(pred_ids, label_str)
            running_forward_time += float(end_forward_time - start_forward_time)
            running_data_gen_time += float(end_loading_data - start_loading_data)
            char_error_rate = cer(pred_ids, label_str)
            running_wer += word_error_rate
            running_cer += char_error_rate
        if len(printed_results) > 5:
            printed_results = random.sample(printed_results, k=5)
            info_msg = ''
            for ground_truth, pred_str in printed_results:
                info_msg += f'\nPREDICTED: {pred_str}\n' \
                            f'TARGET: {ground_truth}\n' \
                            f'{"-" * 100}'
            logger.info(info_msg)
        tb.add_scalar(
            'Average Loss/test', running_loss / len(test_dataloader), epoch)
        tb.add_scalar('WER/test', running_wer / len(test_dataloader), epoch)
        tb.add_scalar('CER/test', running_cer / len(test_dataloader), epoch)
        logger.info(
            f'Test loss on {epoch} epoch: {running_loss / len(test_dataloader)}')
        logger.info(
            f'Average forward time during test on {epoch} epoch: {running_forward_time / len(test_dataloader)}')
        logger.info(
            f'Average data gen time during test on {epoch} epoch: {running_data_gen_time / len(test_dataloader)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    parser.add_argument('pretrained_wav2vec2_path', type=str)
    parser.add_argument('batch_size', type=int)
    # parser.add_argument('checkpoint_path', type=str)
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
    # logger.info('Saving pretrained model')
    # model.save_pretrained('models/wav2vec2-large-xlsr-53')
    model.gradient_checkpointing_enable()
    model.freeze_feature_extractor()

    logger.info(f'len of tokenizer: {len(data_processor.processor.tokenizer)}\n'
                f'len of vocab_size: {len(data_processor.characters_char_map)}')
    num_training_steps = NUM_EPOCHS * len(test_dataloader)
    logger.info(f'Num of training steps: {num_training_steps}')

    optimizer = torch.optim.AdamW(model.parameters())  # lr=1e-3
    # 0.3162 * 0.3162 = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.3162)
    # checkpoint = torch.load(args.checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # next_epoch = checkpoint['epoch'] + 1
    next_epoch = 1
    for epoch in range(next_epoch, NUM_EPOCHS + next_epoch):
        logger.info(f'Start training on {epoch} epoch.')
        train(model=model, device=device, train_dataloader=train_dataloader, lr_scheduler=lr_scheduler, epoch=epoch,
              optimizer=optimizer)
        logger.info(f'Start testing on {epoch} epoch.')
        test(model=model, device=device, test_dataloader=test_dataloader, processor=data_processor, epoch=epoch)
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'lr_scheduler_dict': lr_scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'models/checkpoints/baseline/baseline_{epoch}_epoch')
    tb.close()
    logger.info('Saving model')
    torch.save({
        'model_state_dict': model.state_dict(),
        'lr_scheduler_dict': lr_scheduler.state_dict(),
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
    file_handler = logging.FileHandler('Multitask_Wav2Vec2_v5.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main()
