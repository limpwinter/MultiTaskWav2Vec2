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
from typing import List
import numpy as np

from russian_g2p.DataHandler import (DataGenerator,
                                     DataProcessor
                                     )

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
tb = SummaryWriter()


def mean_pooling(token_embeddings):
    input_mask_expanded = torch.ones_like(token_embeddings, dtype=torch.long)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


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


class MultitaskWav2vecModel(nn.Module):
    def __init__(self, wav2vec2model_path, characters, phonems, pos_tags):
        super(MultitaskWav2vecModel, self).__init__()
        self.characters = characters
        self.phonems = phonems
        self.pos_tags = pos_tags
        self.vocab_size = len(self.characters)  # char_to_num.vocabulary_size()
        self.phonems_size = len(self.phonems)  # char_to_num.vocabulary_size()
        self.pos_tags_size = len(self.pos_tags)  # char_to_num.vocabulary_size()
        self.sentence_embedding_size = 1024  # sentence embedding size TODO Убрать это магическое число

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            wav2vec2model_path,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean",
            pad_token_id=data_processor.characters_char_map['<pad>'],
            output_hidden_states=True)
        self.wav2vec2.gradient_checkpointing_enable()
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.wav2vec2_config = self.wav2vec2.config
        self.hidden_size = self.wav2vec2_config.hidden_size

        self.dp0 = nn.Dropout(0.1)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        self.fc0 = nn.Linear(self.hidden_size, self.sentence_embedding_size)  # sentence embedding
        self.fc1 = nn.Linear(self.hidden_size, self.pos_tags_size)  # pos tags head
        self.fc2 = nn.Linear(self.hidden_size, self.phonems_size)  # phonems recognition head
        self.fc3 = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head

    def forward(self,
                input_values,
                attention_mask,
                labels=None
                ):
        w2v2_output = self.wav2vec2(input_values, attention_mask)
        hidden_states = w2v2_output.hidden_states
        hs_len = len(hidden_states)

        hs_0 = hidden_states[0]  # lowest level features
        hs_1 = hidden_states[hs_len // 3 - 1]  # low level features
        hs_2 = hidden_states[(hs_len * 2) // 3 - 1]  # middle level features
        hs_3 = hidden_states[(hs_len * 3) // 3 - 1]  # high level features

        # logits = []
        hs_0 = self.dp0(hs_0)
        head_0_logits = self.fc0(hs_0)  # sentence embedding recognition head
        # logits.append(head_0_logits)

        hs_1 = self.dp1(hs_1)
        head_1_logits = self.fc1(hs_1)  # part of speech tags recognition head
        # logits.append(head_1_logits)

        hs_2 = self.dp2(hs_2)
        head_2_logits = self.fc2(hs_2)  # phonems recognition head
        # logits.append(head_2_logits)

        hs_3 = self.dp3(hs_3)
        head_3_logits = self.fc3(hs_3)  # characters recognition head
        # logits.append(head_3_logits)
        losses = None

        def compute_ctc_loss(y_true, y_pred, blank_label_idx, masks):

            masks = (
                masks if masks is not None else torch.ones_like(input_values, dtype=torch.long)
            )

            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(masks.sum(-1)).to(torch.long)
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = y_true >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = y_true.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(y_pred, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,  # ok
                    flattened_targets,  # ok
                    input_lengths,  # ok
                    target_lengths,  # ok
                    blank=blank_label_idx,
                    reduction='mean',
                    zero_infinity=True
                )
            return loss

        if labels is not None:
            char_labels, pos_tags_labels, sentences_embeddings = labels
            char_loss = compute_ctc_loss(char_labels, head_3_logits, self.characters['<pad>'], attention_mask)
            # phonem_loss = compute_ctc_loss(phonem_labels, head_2_logits, self.phonems['<pad>'], attention_mask)
            pos_tag_loss = compute_ctc_loss(pos_tags_labels, head_1_logits, self.pos_tags['<pad>'], attention_mask)

            sent_embedding = mean_pooling(head_0_logits)
            batch_size = sent_embedding.shape[0]
            # assert batch_size == BATCH_SIZE
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            target_simmilarity = torch.ones(batch_size).to(device)
            embedding_difference_loss = nn.functional.cosine_embedding_loss(sent_embedding,
                                                                            sentences_embeddings,
                                                                            target_simmilarity)
            # embedding_difference_loss *= 100
            # TODO Добавить коофициенты ко всем loss функциям и убрать магическое число)))
            losses = char_loss, pos_tag_loss, embedding_difference_loss

        if losses is not None:
            return head_3_logits, losses
        else:
            return head_3_logits


def train(model, device, data_generator, epoch, lr_scheduler, optimizer):
    model.train()
    running_loss = 0
    running_char_loss = 0
    running_pos_tags_loss = 0
    running_sent_emb_loss = 0
    for i in trange(len(data_generator)):
        (data, mask), (char_labels, pos_tags_labels, sent_emb_labels) = data_generator[i]

        data, mask = data.to(device), mask.to(device)
        char_labels, pos_tags_labels, sent_emb_labels = char_labels.to(device), pos_tags_labels.to(
            device), sent_emb_labels.to(device)
        labels = char_labels, pos_tags_labels, sent_emb_labels
        logits, losses = model(data,
                               mask,
                               labels
                               )

        del data, mask, char_labels, pos_tags_labels, sent_emb_labels
        char_loss, pos_tags_loss, sent_emb_loss = losses
        loss = char_loss + pos_tags_loss + sent_emb_loss
        running_loss += loss.item()
        tb.add_scalar('Loss/train/overall', loss.item(), epoch)
        tb.add_scalar('Loss/train/character', char_loss.item(), epoch)
        tb.add_scalar('loss/train/pos_tags', pos_tags_loss.item(), epoch)
        tb.add_scalar('Loss/train/sent_embedding', sent_emb_loss, epoch)
        running_char_loss += char_loss.item()
        running_pos_tags_loss += pos_tags_loss.item()
        running_sent_emb_loss += sent_emb_loss.item()
        info_msg = f'Epoch: {epoch} (Iteration: {i} with loss: {loss.item()})'
        logger.info(info_msg)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    tb.add_scalar('Loss/characters', running_char_loss / len(data_generator), epoch)
    tb.add_scalar('Loss/Parts of speech', running_pos_tags_loss / len(data_generator), epoch)
    tb.add_scalar('Loss/sentence_embeddings', running_sent_emb_loss / len(data_generator), epoch)
    tb.add_scalar('Loss/train', running_loss / len(data_generator), epoch)
    logger.info(f'Loss on {epoch} epoch: {running_loss / len(data_generator)}')


def test(model, device, data_generator, epoch):
    model.eval()
    running_wer = 0
    running_cer = 0
    running_loss = 0
    running_char_loss = 0
    running_pos_tags_loss = 0
    running_sent_emb_loss = 0
    with torch.no_grad():
        printed_results = []
        for i in trange(len(data_generator)):
            (data, mask), (char_labels, pos_tags_labels, sent_emb_labels) = data_generator[i]
            data, mask = data.to(device), mask.to(device)
            char_labels, pos_tags_labels, sent_emb_labels = char_labels.to(device), pos_tags_labels.to(
                device), sent_emb_labels.to(device)
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
            pred_ids = data_generator.data_processor.decode_batch_predictions(logits)
            char_labels[char_labels == -100] = data_generator.data_processor.characters_char_map['<pad>']
            label_str = data_generator.data_processor.decode_batch_predictions(char_labels, group_tokens=False)
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
        tb.add_scalar('WER/test', running_wer / len(data_generator), epoch)
        tb.add_scalar('CER/test', running_cer / len(data_generator), epoch)


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
    NUM_EPOCHS = 2000
    RANDOM_STATE = 256
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    logger.info('Reading Data...')
    train_data = pd.read_json(os.path.join(args.data_folder, 'train', '10hours.jsonl'), lines=True)
    test_data = pd.read_json(os.path.join(args.data_folder, 'test', 'crowd', 'manifest.jsonl'), lines=True)
    test_data = test_data[test_data['text'] != ' ']
    train_data = train_data[train_data['text'] != ' ']
    train_data = train_data[train_data['duration'] <= 10]

    test_data_len = min(len(test_data), len(train_data))
    test_data = test_data.sample(
        n=test_data_len,
        random_state=RANDOM_STATE).sort_values('duration')

    logger.info(f'Training examples: {len(train_data)}')
    logger.info(f'Testing examples: {len(test_data)}')

    data_processor = DataProcessor()

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
        logger=logger,
        train=False,
        random_state=RANDOM_STATE
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
    optimizer = AdamW(multitask_wav2vec2.parameters(), lr=1e-3)
    # lr_scheduler = get_scheduler(name="linear",
    #                              optimizer=optimizer,
    #                              num_warmup_steps=1000,
    #                              num_training_steps=num_training_steps
    #                              )
    lr_scheduler = MultiStepLR(optimizer, milestones=[5, 10, 20, 40], gamma=0.3162)

    # optimizer = torch.optim.Adam(multitask_wav2vec2.parameters(), lr=1e-5)

    for epoch in range(1, NUM_EPOCHS + 1):
        train(multitask_wav2vec2, device, train_data_gen, epoch, lr_scheduler, optimizer)
        test(multitask_wav2vec2, device, test_data_gen, epoch)
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': multitask_wav2vec2.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ls_scheduler_state_dict': lr_scheduler.state_dict()},
                f'models/checkpoints/configuration_v0/MultitaskWav2vec_{epoch}_epoch')
    tb.close()
    logger.info('Saving model')
    torch.save({
        'model_state_dict': multitask_wav2vec2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ls_scheduler_state_dict': lr_scheduler.state_dict()},
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
