import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from russian_g2p.DataHandler import DataProcessor

# 11.05.2022 (23:24)


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
            ctc_loss_reduction="mean",
            pad_token_id=characters['<pad>'],
            output_hidden_states=True)
        self.wav2vec2.gradient_checkpointing_enable()
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.wav2vec2_config = self.wav2vec2.config
        self.hidden_size = self.wav2vec2_config.hidden_size

        self.dp0 = nn.Dropout(0.1)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        self.sent_emb_head = nn.Linear(self.hidden_size, self.sentence_embedding_size)  # sentence embedding
        self.pos_tag_head = nn.Linear(self.hidden_size, self.pos_tags_size)  # pos tags head
        self.phonem_head = nn.Linear(self.hidden_size, self.phonems_size)  # phonems recognition head
        self.char_head = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head

    def forward(self,
                input_values,
                attention_mask,
                labels=None
                ):
        w2v2_output = self.wav2vec2(
            input_values,
            attention_mask
        )
        hidden_states = w2v2_output.hidden_states
        hs_len = len(hidden_states)

        hs_0 = hidden_states[2]  # lowest level features 2 (phonems)
        hs_1 = hidden_states[10]  # low level features 10 pos_tags
        # hs_2 = hidden_states[(hs_len * 2) // 3 - 1]  # middle level features 15
        hs_3 = hidden_states[(hs_len * 3) // 3 - 1]  # high level features 23

        # logits = []
        hs_0 = self.dp0(hs_0)
        phonem_logits = self.phonem_head(hs_0)  # phonemes recognition head
        # logits.append(head_0_logits)

        hs_1 = self.dp1(hs_1)
        pos_tag_logits = self.pos_tag_head(hs_1)  # part of speech tags recognition head
        # logits.append(head_1_logits)

        # hs_2 = self.dp2(hs_2)
        # head_2_logits = self.fc2(hs_2)  # phonems recognition head
        # logits.append(head_2_logits)

        hs_3 = self.dp3(hs_3)
        char_logits = self.char_head(hs_3)  # characters recognition head
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
            chars_labels = labels['char_labels']
            pos_tags_labels = labels['pos_tag_labels']
            phonems_labels = labels['phonem_labels']
            # sentences_embeddings = labels['sentence_embedding']

            char_loss = compute_ctc_loss(chars_labels, char_logits, self.characters['<pad>'], attention_mask)
            phonem_loss = compute_ctc_loss(phonems_labels, phonem_logits, self.phonems['<pad>'], attention_mask)
            pos_tag_loss = compute_ctc_loss(pos_tags_labels, pos_tag_logits, self.pos_tags['<pad>'], attention_mask)

            # sent_embedding = mean_pooling(head_0_logits)
            # batch_size = sent_embedding.shape[0]
            # assert batch_size == BATCH_SIZE
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # target_simmilarity = torch.ones(batch_size).to(device)
            # embedding_difference_loss = nn.functional.cosine_embedding_loss(sent_embedding,
            #                                                                 sentences_embeddings,
            #                                                                 target_simmilarity)
            # embedding_difference_loss *= 100
            # TODO Добавить коофициенты ко всем loss функциям и убрать магическое число)))
            losses = {"char_loss": char_loss,
                      "pos_tag_loss": pos_tag_loss,
                      # "embedding_difference_loss": embedding_difference_loss,
                      "phonem_loss": phonem_loss
                      }

            if losses is not None:
                return char_logits, losses
        else:
            return char_logits


class MultitaskWav2vecModel_all_char(nn.Module):
    def __init__(self, wav2vec2model_path, characters, phonems, pos_tags):
        super(MultitaskWav2vecModel_all_char, self).__init__()
        self.characters = characters
        self.phonems = phonems
        self.pos_tags = pos_tags
        self.vocab_size = len(self.characters)  # char_to_num.vocabulary_size()
        self.phonems_size = len(self.phonems)  # char_to_num.vocabulary_size()
        self.pos_tags_size = len(self.pos_tags)  # char_to_num.vocabulary_size()
        self.sentence_embedding_size = 1024  # sentence embedding size

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            wav2vec2model_path,
            ctc_loss_reduction="mean",
            pad_token_id=characters['<pad>'],
            output_hidden_states=True)
        self.wav2vec2.gradient_checkpointing_enable()
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.wav2vec2_config = self.wav2vec2.config
        self.hidden_size = self.wav2vec2_config.hidden_size

        self.dp0 = nn.Dropout(0.1)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        # self.sent_emb_head = nn.Linear(self.hidden_size, self.sentence_embedding_size)  # sentence embedding
        # self.pos_tag_head = nn.Linear(self.hidden_size, self.pos_tags_size)  # pos tags head
        # self.phonem_head = nn.Linear(self.hidden_size, self.phonems_size)  # phonems recognition head
        self.char_head_0 = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head
        self.char_head_1 = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head
        self.char_head_2 = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head

    def forward(self,
                input_values,
                attention_mask,
                labels=None
                ):
        w2v2_output = self.wav2vec2(
            input_values,
            attention_mask
        )
        hidden_states = w2v2_output.hidden_states
        hs_len = len(hidden_states)

        hs_0 = hidden_states[2]  # lowest level features 2 (phonems)
        hs_1 = hidden_states[10]  # low level features 10 pos_tags
        # hs_2 = hidden_states[(hs_len * 2) // 3 - 1]  # middle level features 15
        hs_3 = hidden_states[(hs_len * 3) // 3 - 1]  # high level features 23

        # logits = []
        hs_0 = self.dp0(hs_0)
        # phonem_logits = self.phonem_head(hs_0)  # phonemes recognition head
        char_logits_0 = self.char_head_0(hs_0)
        # logits.append(head_0_logits)

        hs_1 = self.dp1(hs_1)
        char_logits_1 = self.char_head_1(hs_1)
        # logits.append(head_1_logits)

        # hs_2 = self.dp2(hs_2)
        # head_2_logits = self.fc2(hs_2)  # phonems recognition head
        # logits.append(head_2_logits)

        hs_2 = self.dp3(hs_3)
        char_logits_2 = self.char_head_2(hs_2)
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
            chars_labels = labels['char_labels']
            # pos_tags_labels = labels['pos_tag_labels']
            # phonems_labels = labels['phonem_labels']
            # sentences_embeddings = labels['sentence_embedding']

            char_loss_1 = compute_ctc_loss(chars_labels, char_logits_0, self.characters['<pad>'], attention_mask)
            char_loss_2 = compute_ctc_loss(chars_labels, char_logits_1, self.characters['<pad>'], attention_mask)
            char_loss_3 = compute_ctc_loss(chars_labels, char_logits_2, self.characters['<pad>'], attention_mask)
            # phonem_loss = compute_ctc_loss(phonems_labels, phonem_logits, self.phonems['<pad>'], attention_mask)
            # pos_tag_loss = compute_ctc_loss(pos_tags_labels, pos_tag_logits, self.pos_tags['<pad>'], attention_mask)

            # sent_embedding = mean_pooling(head_0_logits)
            # batch_size = sent_embedding.shape[0]
            # assert batch_size == BATCH_SIZE
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # target_simmilarity = torch.ones(batch_size).to(device)
            # embedding_difference_loss = nn.functional.cosine_embedding_loss(sent_embedding,
            #                                                                 sentences_embeddings,
            #                                                                 target_simmilarity)
            # embedding_difference_loss *= 100
            losses = {"char_loss_0": char_loss_1,
                      "char_loss_1": char_loss_2,
                      # "embedding_difference_loss": embedding_difference_loss,
                      "char_loss_2": char_loss_3
                      }

            if losses is not None:
                return char_logits_2, losses
        else:
            return char_logits_2


class MultitaskWav2vecModel_3(nn.Module):
    def __init__(self, wav2vec2model_path, characters, phonems, pos_tags):
        super(MultitaskWav2vecModel_3, self).__init__()
        self.characters = characters
        self.phonems = phonems
        self.pos_tags = pos_tags
        self.vocab_size = len(self.characters)  # char_to_num.vocabulary_size()
        self.phonems_size = len(self.phonems)  # char_to_num.vocabulary_size()
        self.pos_tags_size = len(self.pos_tags)  # char_to_num.vocabulary_size()
        # self.sentence_embedding_size = 1024  # sentence embedding size

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            wav2vec2model_path,
            ctc_loss_reduction="mean",
            pad_token_id=characters['<pad>'],
            output_hidden_states=True)
        self.wav2vec2.gradient_checkpointing_enable()
        self.wav2vec2.feature_extractor._freeze_parameters()
        self.wav2vec2_config = self.wav2vec2.config
        self.hidden_size = self.wav2vec2_config.hidden_size

        self.dp0 = nn.Dropout(0.1)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.dp3 = nn.Dropout(0.1)
        # self.sent_emb_head = nn.Linear(self.hidden_size, self.sentence_embedding_size)  # sentence embedding
        self.pos_tag_head = nn.Linear(self.hidden_size, self.pos_tags_size)  # pos tags head
        self.phonem_head = nn.Linear(self.hidden_size, self.phonems_size)  # phonems recognition head
        self.char_head = nn.Linear(self.hidden_size, self.vocab_size)  # characters recognition head

    def forward(self,
                input_values,
                attention_mask,
                labels=None
                ):
        w2v2_output = self.wav2vec2(
            input_values,
            attention_mask
        )
        hidden_states = w2v2_output.hidden_states
        hs_len = len(hidden_states)

        hs_0 = hidden_states[2]  # lowest level features 2 (phonems)
        hs_1 = hidden_states[10]  # low level features 10 pos_tags
        hs_2 = hidden_states[23]  # high level features 23

        hs_0 = self.dp0(hs_0)
        phonem_logits = self.phonem_head(hs_0)  # phonemes recognition head

        hs_1 = self.dp1(hs_1)
        pos_tag_logits = self.pos_tag_head(hs_1)  # part of speech tags recognition head

        hs_2 = self.dp3(hs_2)
        char_logits = self.char_head(hs_2)  # characters recognition head
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
            chars_labels = labels['char_labels']
            pos_tags_labels = labels['pos_tag_labels']
            phonems_labels = labels['phonem_labels']
            # sentences_embeddings = labels['sentence_embedding']

            char_loss = compute_ctc_loss(chars_labels, char_logits, self.characters['<pad>'], attention_mask)
            phonem_loss = 0.6 * compute_ctc_loss(phonems_labels, phonem_logits, self.phonems['<pad>'], attention_mask)
            pos_tag_loss = 0.8 * compute_ctc_loss(pos_tags_labels, pos_tag_logits, self.pos_tags['<pad>'], attention_mask)

            # sent_embedding = mean_pooling(head_0_logits)
            # batch_size = sent_embedding.shape[0]
            # assert batch_size == BATCH_SIZE
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # target_simmilarity = torch.ones(batch_size).to(device)
            # embedding_difference_loss = nn.functional.cosine_embedding_loss(sent_embedding,
            #                                                                 sentences_embeddings,
            #                                                                 target_simmilarity)
            # embedding_difference_loss *= 100
            losses = {"char_loss": char_loss,
                      "pos_tag_loss": pos_tag_loss,
                      # "embedding_difference_loss": embedding_difference_loss,
                      "phonem_loss": phonem_loss
                      }

            if losses is not None:
                return char_logits, losses
        else:
            return char_logits
