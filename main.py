from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import warnings
from torch.optim import AdamW
from transformers import get_scheduler

from russian_g2p.DataHandler import (DataGenerator,
                                     phonems,
                                     characters,
                                     pos_tags,
                                     )

from transformers import Wav2Vec2Model

# from transformers.models.wav2vec2.modeling_wav2vec2 import CausalLMOutput

warnings.filterwarnings("ignore")

TRAIN_DATA_FOLDER = Path('data/train')
TEST_DATA_FOLDER = Path('data/test')

# MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
MODEL_PATH = 'models/wav2vec2-large-xlsr-53'
MODEL_ID = 'facebook/wav2vec2-large-xlsr-53'

BATCH_SIZE = 16

print('Reading Data...', end='')
train_data = pd.read_json(TRAIN_DATA_FOLDER / Path("10min.jsonl"), lines=True)  # .sort_values('duration')
test_data = pd.read_json(TEST_DATA_FOLDER / Path('crowd/manifest.jsonl'), lines=True)  # .sort_values('duration')

test_data = test_data[test_data['text'] != ' ']
train_data = train_data[train_data['text'] != ' ']
print('Done')
# train_data
print('Creating Data Generator...', end='')

train_data_gen = DataGenerator(train_data,
                               batch_size=BATCH_SIZE,
                               train=True,
                               padding=True
                               )
CHAR_PAD_TOKEN_ID = train_data_gen.data_processor.char_pad_token_id
PHONEM_PAD_TOKEN_ID = train_data_gen.data_processor.phn_pad_token_id
# assert train_data_gen.data_processor.phonems_phn_map[PHONEM_PAD_TOKEN_ID] == '[PAD]'
POSTAGS_PAD_TOKEN_ID = train_data_gen.data_processor.postag_pad_token_id

print('Done')
print('Getting First Batch...', end='')
X, y = train_data_gen[0]
assert len(X) == 2 and len(y) == 4
print('Done')


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_ID, output_hidden_states=True)
        self.wav2vec2_config = self.wav2vec2.config
        self.embedding_size = self.wav2vec2_config.hidden_size
        self.vocab_size = len(train_data_gen.data_processor.characters_index_map)  # char_to_num.vocabulary_size()
        self.phonems_size = len(train_data_gen.data_processor.phonems_index_map)  # char_to_num.vocabulary_size()
        self.pos_tags_size = len(train_data_gen.data_processor.postags_index_map)  # char_to_num.vocabulary_size()

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.embedding_size, self.vocab_size)
        self.fc2 = nn.Linear(self.embedding_size, self.phonems_size)
        self.fc3 = nn.Linear(self.embedding_size, self.pos_tags_size)
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self,
                input_values,
                attention_mask,
                labels=None
                ):
        w2v2_output = self.wav2vec2(input_values, attention_mask)
        hidden_states = w2v2_output.hidden_states
        hs_len = len(hidden_states)

        hs_1 = hidden_states[hs_len // 3 - 1]  # Числа взяты с потолка)))
        hs_2 = hidden_states[(hs_len * 2) // 3 - 1]
        hs_3 = hidden_states[(hs_len * 3) // 3 - 1]

        logits = []

        hs_1 = self.dropout(hs_1)
        head_1_logits = self.fc1(hs_1)  # character recognition head
        # logits.append(head_1_logits)

        hs_2 = self.dropout(hs_2)
        head_2_logits = self.fc2(hs_2)  # phonems recognition head
        # logits.append(head_2_logits)

        hs_3 = self.dropout(hs_3)
        head_3_logits = self.fc3(hs_3)  # part of speech tags recognition head
        # logits.append(head_3_logits)
        losses = None

        def compute_loss(labels, logits, blank_label_idx, attention_mask):
            # if targets.max() >= vocab_size:
            #     raise ValueError(f"Label values must be <= vocab_size: {vocab_size}")
            #
            # # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )

            input_lengths = self.wav2vec2._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

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
            char_labels, phonem_labels, pos_tags_labels = labels
            char_loss = compute_loss(char_labels, head_1_logits, CHAR_PAD_TOKEN_ID, attention_mask)
            phonem_loss = compute_loss(phonem_labels, head_2_logits, PHONEM_PAD_TOKEN_ID, attention_mask)
            pos_tag_loss = compute_loss(pos_tags_labels, head_3_logits, POSTAGS_PAD_TOKEN_ID, attention_mask)
            losses = char_loss + pos_tag_loss + phonem_loss

        if losses is not None:
            return losses
        else:
            return logits


# mod_id = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"


print('Creating Model...', end='')
model = MyModel()
print('Done')

# summary(wav2vec, input_size=[(BATCH_SIZE, 240_000), (BATCH_SIZE, 240_000)])
# print('-' * 18, 'Call Forward Method with loss', '-' * 18)

# pred, loss = model(X[0],
#                    X[1],
#                    labels=y)
# print('-' * 18, 'Done', '-' * 18)

# print(pred)
# print(loss)
# print(pred.shape)
num_epochs = 1
num_training_steps = num_epochs * len(train_data_gen)
progress_bar = tqdm(range(num_training_steps))
optimizer = AdamW(model.parameters(), lr=5e-5)

lr_scheduler = get_scheduler(name="linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps
                             )
model.train()
for epoch in range(num_epochs):
    for i in range(num_training_steps):
        (data, mask), labels = train_data_gen[i]
        # batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(data,
                     mask,
                     labels
                     )
        # loss = outputs.loss
        print(loss)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
