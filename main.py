import random
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchinfo import summary
from tqdm.auto import tqdm
from tqdm import trange
from transformers import Wav2Vec2Model
from transformers import get_scheduler
from datasets import load_metric
from torch.utils.tensorboard import SummaryWriter

from russian_g2p.DataHandler import (DataGenerator,
                                     phonems,
                                     characters,
                                     pos_tags,
                                     )

warnings.filterwarnings("ignore")

TRAIN_DATA_FOLDER = Path('data/train')
TEST_DATA_FOLDER = Path('data/test')

MODEL_ID = 'facebook/wav2vec2-large-xlsr-53'

BATCH_SIZE = 16
NUM_EPOCHS = 3

print('Reading Data...', end='')
train_data = pd.read_json(TRAIN_DATA_FOLDER / Path("10min.jsonl"), lines=True)  # .sort_values('duration')
test_data = pd.read_json(TEST_DATA_FOLDER / Path('crowd/manifest.jsonl'), lines=True)  # .sort_values('duration')
test_data_len = max(len(test_data), len(train_data))
test_data = test_data.sample(n=test_data_len)
test_data = test_data[test_data['text'] != ' ']
train_data = train_data[train_data['text'] != ' ']
print('Done')

print('Creating Data Generators...', end='')
train_data_gen = DataGenerator(train_data,
                               batch_size=BATCH_SIZE,
                               train=True
                               )

test_data_gen = DataGenerator(test_data,
                              batch_size=BATCH_SIZE,
                              train=False)

CHAR_PAD_TOKEN_ID = train_data_gen.data_processor.char_pad_token_id
PHONEM_PAD_TOKEN_ID = train_data_gen.data_processor.phn_pad_token_id
POSTAGS_PAD_TOKEN_ID = train_data_gen.data_processor.postag_pad_token_id
print('Done')


def mean_pooling(token_embeddings):
    input_mask_expanded = torch.ones_like(token_embeddings, dtype=torch.long)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_ID, output_hidden_states=True)
        self.wav2vec2_config = self.wav2vec2.config
        self.hidden_size = self.wav2vec2_config.hidden_size
        self.vocab_size = len(train_data_gen.data_processor.characters_index_map)  # char_to_num.vocabulary_size()
        self.phonems_size = len(train_data_gen.data_processor.phonems_index_map)  # char_to_num.vocabulary_size()
        self.pos_tags_size = len(train_data_gen.data_processor.postags_index_map)  # char_to_num.vocabulary_size()
        self.sentence_embedding_size = 1024  # sentence embedding size TODO Убрать это магическое число

        self.dropout = nn.Dropout(0.1)
        self.fc0 = nn.Linear(self.hidden_size, self.sentence_embedding_size)  # sentence embedding
        self.fc1 = nn.Linear(self.hidden_size, self.pos_tags_size)  # pos tags head
        self.fc2 = nn.Linear(self.hidden_size, self.phonems_size)  # phonems recognition head
        self.fc3 = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head
        self.wav2vec2.feature_extractor._freeze_parameters()

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
        hs_0 = self.dropout(hs_0)
        head_0_logits = self.fc0(hs_0)  # sentence embedding recognition head
        # logits.append(head_0_logits)

        hs_1 = self.dropout(hs_1)
        head_1_logits = self.fc1(hs_1)  # part of speech tags recognition head
        # logits.append(head_1_logits)

        hs_2 = self.dropout(hs_2)
        head_2_logits = self.fc2(hs_2)  # phonems recognition head
        # logits.append(head_2_logits)

        hs_3 = self.dropout(hs_3)
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
                    zero_infinity=False
                )
            return loss

        if labels is not None:
            char_labels, phonem_labels, pos_tags_labels, sentences_embeddings = labels
            char_loss = compute_ctc_loss(char_labels, head_3_logits, CHAR_PAD_TOKEN_ID, attention_mask)
            phonem_loss = compute_ctc_loss(phonem_labels, head_2_logits, PHONEM_PAD_TOKEN_ID, attention_mask)
            pos_tag_loss = compute_ctc_loss(pos_tags_labels, head_1_logits, POSTAGS_PAD_TOKEN_ID, attention_mask)

            sent_embedding = mean_pooling(head_0_logits)
            batch_size = sent_embedding.shape[0]
            assert batch_size == BATCH_SIZE
            target_simmilarity = torch.ones(batch_size)
            embedding_difference_loss = nn.functional.cosine_embedding_loss(sent_embedding,
                                                                            sentences_embeddings,
                                                                            target_simmilarity)
            embedding_difference_loss *= 100
            # TODO Добавить коофициенты ко всем loss функциям и убрать магическое число)))
            losses = char_loss + pos_tag_loss + phonem_loss + embedding_difference_loss

        if losses is not None:
            return head_3_logits, losses
        else:
            return head_3_logits


def train(model, device, train_data_generator, epoch, lr_scheduler, optimizer):
    num_training_steps = epochs * len(train_data_generator)
    progress_bar = trange(num_training_steps)
    model.train()
    running_loss = 0
    for i in trange(len(train_data_generator)):
        (data, mask), labels = train_data_generator[i]
        data, mask, labels = data.to(device), mask.to(device), labels.to(device)
        logits, loss = model(data,
                             mask,
                             labels
                             )
        # logits = torch.argmax(logits, dim=-1)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    print(f'Epoch: {epoch}\tLoss:{running_loss / len(train_data_gen)}')


wer_metric = load_metric("wer")


def test(model, device, data_generator):
    model.eval()
    with torch.no_grad():
        for i in trange(len(data_generator)):
            (data, mask), labels = data_generator[i]
            data, mask, labels = data.to(device), mask.to(device), labels
            logits = model(data,
                           mask)
            logits = torch.argmax(logits, dim=-1)
            pred_ids = data_generator.data_processor.decode_batch_predictions(logits)
            labels = labels[0]
            label_str = data_generator.data_processor.decode_batch_predictions(labels, group_tokens=False)
            wer = wer_metric.compute(predictions=pred_ids, references=label_str)
            print(wer)


print('Creating Model...', end='')
model = MyModel()
print('Done')

# writer = SummaryWriter()
# (data, mask), labels = train_data_gen[0]
# writer.add_graph(model, [data, mask])
# writer.close()
num_training_steps = NUM_EPOCHS * len(train_data_gen)
progress_bar = tqdm(range(num_training_steps))
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(name="linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps
                             )

use_cuda = not torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

for epoch in range(1, NUM_EPOCHS + 1):
    # train(model, device, train_data_gen, epoch, lr_scheduler, optimizer)
    test(model, device, train_data_gen)
