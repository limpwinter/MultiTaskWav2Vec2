import argparse
import logging
import os.path
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List

from jiwer import cer, wer
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

# 14.05.2022 (3:40)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
tb = SummaryWriter('stat_run/Baseline_dataloder_pkl_t20')


def train(model, device, train_dataloader, lr_scheduler, epoch, optimizer):
    model.train()
    running_loss = 0
    running_forward_time = 0
    running_data_gen_time = 0
    iterable_dataloder = iter(train_dataloader)
    n = len(train_dataloader)
    for idx in range(len(train_dataloader)):
        start_loading_data = time.time()
        batch = next(iterable_dataloder)
        # batch = {k: v.to(device) for k, v in batch.items()}
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
        running_loss += loss.item()
        running_data_gen_time += (end_loading_data - start_loading_data)
        running_forward_time += (end_forward - start_forward)
        tb.add_scalar('Loss/character/train', loss.item(), epoch * n + idx)
        tb.add_scalar('Time/forward/train', float(end_forward - start_forward), epoch * n + idx)
        tb.add_scalar('Time/data gen/train', float(end_loading_data - start_loading_data), epoch * n + idx)

        info_msg = f'Epoch: {epoch} (Iteration: {idx} with loss: {loss.item()}'
        logger.info(info_msg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    tb.add_scalar('Learning Rate', lr_scheduler.get_lr()[-1], epoch)
    lr_scheduler.step()
    tb.add_scalar('Average Loss/character/train', running_loss / len(train_dataloader), epoch)
    logger.info(f'Average forward time during test on {epoch} epoch: {running_forward_time / len(train_dataloader)}')
    logger.info(f'Average data gen time during test on {epoch} epoch: {running_data_gen_time / len(train_dataloader)}')
    logger.info(f'Average train loss on {epoch} epoch: {running_loss / len(train_dataloader)}')


def test(model, device, test_dataloader, processor, epoch):
    model.eval()
    running_wer = 0
    running_cer = 0
    running_loss = 0
    running_forward_time = 0
    running_data_gen_time = 0
    iterable_dataloder = iter(test_dataloader)
    n = len(test_dataloader)
    with torch.no_grad():
        printed_results = []
        for idx in range(len(test_dataloader)):
            start_loading_data = time.time()
            batch = next(iterable_dataloder)
            end_loading_data = time.time()
            data = batch['input_values']
            mask = batch['attention_mask']
            labels = batch['char_labels']
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
            # tb.add_scalar('Loss/test', loss.item(), epoch)
            # tb.add_scalar('Time/data gen/test', float(end_loading_data - start_loading_data), epoch)
            # tb.add_scalar('Time/forward/test', float(end_forward_time - start_forward_time), epoch)

            # tb.add_scalar('Time/data gen/test', float(end_loading_time - start_loading_time), epoch * n + idx)
            tb.add_scalar('Time/forward/test', float(end_forward_time - start_forward_time), epoch * n + idx)
            tb.add_scalar('Time/getting batch/test', float(end_loading_data - start_loading_data), epoch * n + idx)

            running_loss += loss.item()
            logits = torch.argmax(logits, dim=-1)
            pred_ids = processor.decode_batch_predictions(logits)
            labels[labels == -100] = processor.characters_char_map['<pad>']
            label_str = processor.decode_batch_predictions(labels, group_tokens=False)

            printed_results.append(
                (label_str[0],
                 pred_ids[0])
            )
            word_error_rate = wer(label_str, pred_ids)
            running_wer += word_error_rate
            char_error_rate = cer(label_str, pred_ids)
            running_cer += char_error_rate
            running_forward_time += float(end_forward_time - start_forward_time)
            running_data_gen_time += float(end_loading_data - start_loading_data)
        if len(printed_results) > 5:
            printed_results = random.sample(printed_results, k=5)
            info_msg = ''
            for ground_truth, pred_str in printed_results:
                info_msg += f'\nPREDICTED: {pred_str}\n' \
                            f'TARGET: {ground_truth}\n' \
                            f'{"-" * 100}'
            logger.info(info_msg)
        tb.add_scalar(
            'Average Loss/character/test', running_loss / n, epoch)
        tb.add_scalar('WER/test', running_wer / n, epoch)
        tb.add_scalar('CER/test', running_cer / n, epoch)
        logger.info(
            f'Test loss on {epoch} epoch: {running_loss / n}')
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
    RANDOM_STATE = 64
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    logger.info('Reading Data...')

    train_data = pd.read_json(os.path.join(args.data_folder, 'train', '10hours.jsonl'), lines=True)
    test_data = pd.read_json(os.path.join(args.data_folder, 'test', 'crowd', 'manifest.jsonl'), lines=True)

    test_data = test_data[test_data['text'] != ' ']
    train_data = train_data[train_data['text'] != ' ']
    train_data = train_data[train_data['duration'] <= 6]  # removing long audio during training

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

    model = Wav2Vec2ForCTC.from_pretrained(
        args.pretrained_wav2vec2_path,
        ctc_loss_reduction="mean",
        pad_token_id=data_processor.characters_char_map['<pad>'],
        vocab_size=len(data_processor.processor.tokenizer),
        ctc_zero_infinity=True
    ).to(device)
    model.gradient_checkpointing_enable()
    logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.freeze_feature_extractor()
    logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))

    logger.info(f'len of tokenizer: {len(data_processor.processor.tokenizer)}\n'
                f'len of vocab_size: {len(data_processor.characters_char_map)}')
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    logger.info(f'Num of training steps: {num_training_steps}')

    optimizer = torch.optim.AdamW(model.parameters())  # lr=1e-3
    # 0.3162 * 0.3162 = 0.1
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100, 150, 200], gamma=0.3162)
    # checkpoint = torch.load(args.checkpoint_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # next_epoch = checkpoint['epoch'] + 1
    next_epoch = 1

    for epoch in range(next_epoch, NUM_EPOCHS + next_epoch):
        logger.info(f'Start training on {epoch} epoch.')
        train(
            model=model,
            device=device,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            epoch=epoch
        )

        logger.info(f'Start testing on {epoch} epoch.')
        test(
            model=model,
            device=device,
            test_dataloader=test_dataloader,
            processor=data_processor,
            epoch=epoch)
        if epoch % 20 == 0:
            Path('models/checkpoints/baseline').mkdir(
                parents=True,
                exist_ok=True
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'models/checkpoints/baseline/baseline_{epoch}_epoch_dataloader_pkl_t20')
    tb.close()
    logger.info('Saving model')
    torch.save({
        'epoch': NUM_EPOCHS + next_epoch - 1,
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
    file_handler = logging.FileHandler('Logs/Baseline_dataloder_pkl_t20.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main()
