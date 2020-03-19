# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import copy
import pickle

from .modeling_utils import PreTrainedModel, prune_linear_layer, AverageMeter
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings
# from pudb import set_trace
from typing import Tuple

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)



class AdapterLayers(nn.Module):
    def __init__(self, config, num_layers):
        super(AdapterLayers, self).__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(num_layers)])
        self.num_layers = num_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **encoder_hidden_states**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``:
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model
            is configured as a decoder.
        **encoder_attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training:
                       a `masked language modeling` head and a `next sentence prediction (classification)` head. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **masked_lm_loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **ltr_lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next token prediction loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForNextSentencePrediction(BertPreTrainedModel):
    r"""
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next sequence prediction (classification) loss.
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        seq_relationship_scores = outputs[0]

    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
                      the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForMultipleChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
                      the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)

class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ner = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()
    
    def set_class(self, num_pos_labels, num_chunk_labels):
        self.num_pos_labels = num_pos_labels
        self.num_chunk_labels = num_chunk_labels
        self.classifier_pos = nn.Linear(self.hidden_size, self.num_pos_labels)
        self.classifier_chunk = nn.Linear(self.hidden_size, self.num_chunk_labels)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, input_embeds=None, labels=None, 
                pos_labels=None, chunk_labels=None):
        
        outputs = self.bert(input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     position_ids=position_ids,
                     head_mask=head_mask,
                     inputs_embeds=input_embeds)
        
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        ner_logits = self.classifier_ner(sequence_output)
        outputs = (ner_logits,) + outputs[2:]

      
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

                loss_set = (loss, )

                if pos_labels is not None:
                    pos_logits = self.classifier_pos(sequence_output)
                    active_pos_logits = pos_logits.view(-1, self.num_pos_labels)[active_loss]
                    active_pos_labels = pos_labels.view(-1)[active_loss]
                    pos_loss = loss_fct(active_pos_logits, active_pos_labels)
                    
                    loss_set = (loss, pos_loss)
                
                if chunk_labels is not None:
                    chunk_logits = self.classifier_chunk(sequence_output)
                    active_chunk_logits = chunk_logits.view(-1, self.num_chunk_labels)[active_loss]
                    active_chunk_labels = chunk_labels.view(-1)[active_loss]
                    chunk_loss = loss_fct(active_chunk_logits, active_chunk_labels)
                    
                    loss_set = (loss, pos_loss, chunk_loss)
            else:
                loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))

                loss_set = (loss, )

                if pos_labels is not None:
                    pos_logits = self.classifier_pos(sequence_output)
                    pos_loss = loss_fct(pos_logits.view(-1, self.num_pos_labels), pos_labels.view(-1))
                    loss_set = (loss, pos_loss)

                if chunk_labels is not None:
                    chunk_logits = self.classifier_chunk(sequence_output)
                    chunk_loss = loss_fct(chunk_logits.view(-1, self.num_chunk_labels), chunk_labels.view(-1))

                    loss_set = (loss, pos_loss, chunk_loss)

            outputs = loss_set + outputs
        return outputs


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
                      the hidden-states output to compute `span start logits` and `span end logits`). """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))] 
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)  
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet


    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

def copy_model(src_model, target_model):
    src_model_dict = src_model.state_dict()
    target_model_dict = target_model.state_dict()
    
    src_model_dict = {k:v for k,v in src_model_dict.items() if k in target_model_dict}

    target_model_dict.update(src_model_dict)
    target_model.load_state_dict(target_model_dict)
    # torch.save(src_model, "tmp.bin")
    # target_model = torch.load("tmp.bin").to(src_model.device)
    
    return target_model

    

class BiAffine(nn.Module):
    """Biaffine attention layer. (from Chris Manning.) """
    def __init__(self, input_dim, output_dim):
        super(BiAffine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim, input_dim))
        nn.init.xavier_uniform_(self.U)
    
    def forward(self, Rh, Rd):
        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)
        S = Rh @ self.U @ Rd.transpose(-1, -2) # Attention! do transpose !
        return S.squeeze(1)

class DeepBiAffineDecoder(nn.Module):
    """Parsing decodder"""
    def __init__(self, hidden_size, mlp_dim=300):
        super(DeepBiAffineDecoder, self).__init__()
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.mlp_head = nn.Linear(hidden_size, mlp_dim)
        self.mlp_dep = nn.Linear(hidden_size, mlp_dim)

        self.biaffine = BiAffine(mlp_dim, 1)

    
    def forward(self, sequence_output):
        s_head = self.mlp_head(sequence_output)
        s_dep = self.mlp_dep(sequence_output)
        logits = self.biaffine(s_head, s_dep)
        logits = logits.transpose(-1, -2) #[batch_size, max_seq_len, max_seq_len]
        return logits

class DeepBiAffineDecoderV2(nn.Module):
    """Parsing decodder"""
    def __init__(self, hidden_size, mlp_dim=300, num_labels=2):
        super(DeepBiAffineDecoderV2, self).__init__()
        self.mlp_dim = mlp_dim
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        
        self.mlp_head_arc = nn.Linear(hidden_size, mlp_dim)
        self.mlp_dep_arc = nn.Linear(hidden_size, mlp_dim)
        self.biaffine_arc = BiAffine(mlp_dim, 1)

        self.mlp_head_label = nn.Linear(hidden_size, mlp_dim)
        self.mlp_dep_label = nn.Linear(hidden_size, mlp_dim)
        self.biaffine_label = BiAffine(mlp_dim, num_labels)

    
    def forward(self, sequence_output):

        s_head_arc = self.mlp_head_arc(sequence_output)
        s_dep_arc = self.mlp_dep_arc(sequence_output)

        s_head_label = self.mlp_head_label(sequence_output)
        s_dep_label = self.mlp_head_label(sequence_output)

        logits_arc = self.biaffine_arc(s_head_arc, s_dep_arc) # [batch_size, seq_len, seq_len]
        logits_arc = logits_arc.transpose(-1, -2)

        logits_label = self.biaffine_label(s_head_label, s_dep_label) #[batch_size, num_labels, seq_len, seq_len]
        logits_label = logits_label.transpose(-1, -3) #[batch_size, seq_len, seq_len, num_labels]

        preds = torch.argmax(logits_arc, dim=1).unsqueeze(-1) #[batch_size, seq_len, 1]
        indices = preds.unsqueeze(-1).expand(preds.shape + (self.num_labels,)) #[batch_size, seq_len, 1 , num_labels]

        logits_label = torch.gather(logits_label, -2, indices).squeeze(-2) #[batch_size, seq_len,num_labels]

        return (logits_arc, logits_label)


class MTDNNModelV2(BertPreTrainedModel):
    def __init__(self, config, labels_list, task_list,
                 do_task_embedding=False, do_alpha=False,
                 do_adapter=False, num_adapter_layers=2):
        super(MTDNNModelV2, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_fixed_layers = config.num_hidden_layers - num_adapter_layers

        decoder_modules = []
        for labels ,task in zip(labels_list, task_list):
            if task.startswith("PARSING"):
                decoder_modules.append(DeepBiAffineDecoderV2(config.hidden_size, mlp_dim=300, num_labels=len(labels)))
            else:
                decoder_modules.append(nn.Linear(config.hidden_size, len(labels)))
        self.classifier_list =  nn.ModuleList(decoder_modules)

        if do_alpha:
            init_value = torch.zeros(config.num_hidden_layers, 1)
            self.alpha_list = nn.ModuleList([nn.Parameter(init_value, requires_grad=True)])

            self.softmax = nn.Softmax(dim=0)
            self.labels_list = labels_list

        if do_task_embedding:
            self.task_embedding = nn.Embedding(len(labels_list), config.hidden_size)
            self.w1 = nn.Linear(config.hidden_size, 1)
            self.w2 = nn.Linear(config.hidden_size, 1)

        self.labels_list = [len(x) for x in labels_list]
        self.do_task_embedding = do_task_embedding
        self.do_alpha = do_alpha
        self.do_adapter = do_adapter
        self.softmax = nn.Softmax(dim=0)

        self.crit_label_dst = nn.KLDivLoss(reduction="none")
 

        self.init_weights()


        

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, heads=None, labels=None,
                task_id=0, adapter_ft=False, soft_labels=None, soft_heads=None, gamma=0.5, attack=False):
        if self.do_adapter:
            
            if adapter_ft and labels is not None:
                for param in self.bert.encoder.parameters():
                    param.requires_grad = False

                for param in self.bert.encoder.layer[-1].parameters():
                    param.requires_grad = True
                
                for param in self.bert.encoder.layer[-2].parameters():
                    param.requires_grad = True
                
          

        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        hidden_states = outputs[-1]
        # sequence_output = hidden_states[-1]

        classifier = self.classifier_list[task_id]
        num_labels = self.labels_list[task_id]
        

        if self.do_task_embedding:
            task_embbedding = self.task_embedding(torch.tensor([task_id]).cuda()).view(-1)
            hidden_states = hidden_states[1:]
            hidden_states = torch.stack(hidden_states)
            # alpha = w1*hidden_states + w2*task_embedding + bias 
            out1 = self.w1(hidden_states) #[num_hidden_layers, batch_size, seq_len, 1]
            task_embedding = task_embedding.expand(hidden_states.size(0), hidden_states.size(1), task_embedding.size(0)) # [num_hidden_layers, batch_size, hidden_size]
            out2 = self.w2(task_embedding) # [num_hidden_layers, batch_size, 1]
            alpha = out1.squeeze(-1) + out2 # [num_hidden_layers, batch_size, seq_len]
            alpha = self.softmax(alpha).permute(1, 0, 2) #[batch_size, num_hidden_layers, seq_len]
            alpha_vis = torch.mean(alpha, dim=0)
            hidden_states = hidden_states.permute(1,0,2,3)  # [batch_size, num_hidden_layers, seq_len, hidden_size]
            alpha = alpha.view(alpha.size() + (1,)).expand(alpha.size() + (hidden_states.size(3), )) #[batch_size, num_hidden_layers, seq_len, hidden_size]
            sequence_output = torch.sum(alpha*hidden_states, dim=1) #[batch_size, seq_len, hidden_size]

        elif self.do_alpha:
            alpha = self.alpha_list[task_id]
            alpha = self.softmax(alpha)
            hidden_states = hidden_states[1:]
            hidden_states = torch.stack(hidden_states)   # [num_hidden_layers, batch_size, seq_len, hidden_size]
            hidden_states = hidden_states.permute(1,2,3,0)  # [batch_size, seq_len, hidden_size, num_hidden_layers]
            sequence_output = torch.matmul(hidden_states, alpha).squeeze(-1) # [batch_size, seq_len, hidden_size]

        sequence_output = self.dropout(sequence_output)
        if type(classifier) == DeepBiAffineDecoderV2: # parsing task
            logits_arc, logits_label = classifier(sequence_output)
            outputs = (logits_arc, logits_label) + outputs[2:]

            if labels is not None and heads is not None:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits_arc = logits_arc.contiguous().view(-1, logits_arc.size(-1))[active_loss]
                    active_heads = heads.view(-1)[active_loss]
                    active_logits_label = logits_label.contiguous().view(-1, num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]

                    loss_arc = loss_fct(active_logits_arc, active_heads)
                    loss_labels = loss_fct(active_logits_label, active_labels)
                    loss = loss_arc + loss_labels

                    if soft_labels is not None and soft_heads is not None:
                        active_soft_labels = soft_labels.contiguous().view(-1, num_labels)[active_loss]
                        active_soft_heads = soft_heads.contiguous().view(-1, soft_heads.size(-1))[active_loss]

                        kv_loss_arc = self.crit_label_dst(F.log_softmax(active_logits_arc.float(), dim=-1),
                                                          F.softmax(active_soft_heads.float(), dim=-1)).sum(dim=-1).mean()
                        kv_loss_label = self.crit_label_dst(F.log_softmax(active_logits_label.float(), dim=-1),
                                                            F.softmax(active_soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        kv_loss = kv_loss_arc + kv_loss_label
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                else:
                    logits_arc = logits_arc.contiguous().view(-1, logits_arc.size(-1))
                    heads = heads.view(-1)
                    loss_arc = loss_fct(logits_arc, heads)

                    logits_label = logits_label.contiguous().view(-1, num_labels)
                    labels = labels.view(-1)

                    loss_labels = loss_fct(logits_label, labels)
                    loss = loss_arc + loss_labels
                    if soft_labels is not None and soft_heads is not None:
                        soft_labels = soft_labels.contiguous().view(-1, num_labels)
                        soft_heads = soft_heads.contiguous().view(-1, soft_heads.size(-1))

                        kv_loss_arc = self.crit_label_dst(F.log_softmax(logits_arc.float(), dim=-1),
                                                          F.softmax(soft_heads.float(), dim=-1)).sum(dim=-1).mean()
                    

                        kv_loss_label = self.crit_label_dst(F.log_softmax(logits_label.float(), dim=-1),
                                                            F.softmax(soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        
                        kv_loss = kv_loss_arc + kv_loss_label
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                outputs = (loss, ) + outputs
        else:
            logits = classifier(sequence_output)

            outputs = (logits,) + outputs[2:]

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if soft_labels is not None:
                        soft_labels = soft_labels.view(-1, num_labels)[active_loss]
                        kv_loss = self.crit_label_dst(F.log_softmax(active_logits.float(), dim=-1),
                                                     F.softmax(soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                else:
                    logits = logits.view(-1, self.num_labels)
                    if soft_labels is not None:
                        soft_labels = soft_labels.view(-1, num_labels)
                        kv_loss = self.crit_label_dst(F.log_softmax(active_logits.float(), dim=-1),
                                                      F.softmax(soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                    loss = loss_fct(logits, labels.view(-1))
                outputs = (loss,) + outputs

        if self.do_task_embedding:
            outputs = (alpha_vis, ) + outputs
        elif self.do_alpha:
            outputs = (alpha, ) + outputs
        return outputs





class MTDNNModel(BertPreTrainedModel):
    def __init__(self, config, labels_list, task_list,
                 do_task_embedding=False, do_alpha=False,
                 do_adapter=False, num_adapter_layers=2):
        super(MTDNNModel, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_fixed_layers = config.num_hidden_layers - num_adapter_layers

        decoder_modules = []
        for labels ,task in zip(labels_list, task_list):
            if task.startswith("PARSING"):
                decoder_modules.append(DeepBiAffineDecoder(config.hidden_size, mlp_dim=300))
            else:
                decoder_modules.append(nn.Linear(config.hidden_size, len(labels)))
        self.classifier_list =  nn.ModuleList(decoder_modules)

        if do_alpha:
            init_value = torch.zeros(config.num_hidden_layers, 1)
            self.alpha_list = nn.ModuleList([nn.Parameter(init_value, requires_grad=True)])

            self.softmax = nn.Softmax(dim=0)
            self.labels_list = labels_list

        if do_task_embedding:
            self.task_embedding = nn.Embedding(len(labels_list), config.hidden_size)
            self.w1 = nn.Linear(config.hidden_size, 1)
            self.w2 = nn.Linear(config.hidden_size, 1)

        self.labels_list = [len(x) for x in labels_list]
        self.do_task_embedding = do_task_embedding
        self.do_alpha = do_alpha
        self.do_adapter = do_adapter
        self.softmax = nn.Softmax(dim=0)

        self.init_weights()

        

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                task_id=0, adapter_ft=False):
        if self.do_adapter:
            
            if adapter_ft and labels is not None:
                for param in self.bert.encoder.parameters():
                    param.requires_grad = False

                for param in self.bert.encoder.layer[-1].parameters():
                    param.requires_grad = True
                
                for param in self.bert.encoder.layer[-2].parameters():
                    param.requires_grad = True
                
                # update_params = [param for param in self.bert.parameters() if param.requires_grad]
                # no_update_params = [param for param in self.bert.parameters() if not param.requires_grad]
                # # print(update_params)
                # print(no_update_params)
            # self.bert.encoder.layer[-1] = self.adapter_layers[-1]
            # self.bert.encoder.layer[-2] = self.adapter_layers[-2]

        #     adapter_layer = self.adapter_layers[task_id]
        #     for i in range(len(adapter_layer.layers)):
        #         copy_model(adapter_layer.layers[i], self.bert.encoder.layer[-(i+1)])
                # self.bert.encoder.layer[-(i+1)] = adapter_layer.layers[i]

        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        hidden_states = outputs[-1]
        # sequence_output = hidden_states[-1]

        classifier = self.classifier_list[task_id]
        num_labels = self.labels_list[task_id]
        

        if self.do_task_embedding:
            task_embbedding = self.task_embedding(torch.tensor([task_id]).cuda()).view(-1)
            hidden_states = hidden_states[1:]
            hidden_states = torch.stack(hidden_states)
            # alpha = w1*hidden_states + w2*task_embedding + bias 
            out1 = self.w1(hidden_states) #[num_hidden_layers, batch_size, seq_len, 1]
            task_embedding = task_embedding.expand(hidden_states.size(0), hidden_states.size(1), task_embedding.size(0)) # [num_hidden_layers, batch_size, hidden_size]
            out2 = self.w2(task_embedding) # [num_hidden_layers, batch_size, 1]
            alpha = out1.squeeze(-1) + out2 # [num_hidden_layers, batch_size, seq_len]
            alpha = self.softmax(alpha).permute(1, 0, 2) #[batch_size, num_hidden_layers, seq_len]
            alpha_vis = torch.mean(alpha, dim=0)
            hidden_states = hidden_states.permute(1,0,2,3)  # [batch_size, num_hidden_layers, seq_len, hidden_size]
            alpha = alpha.view(alpha.size() + (1,)).expand(alpha.size() + (hidden_states.size(3), )) #[batch_size, num_hidden_layers, seq_len, hidden_size]
            sequence_output = torch.sum(alpha*hidden_states, dim=1) #[batch_size, seq_len, hidden_size]

        elif self.do_alpha:
            alpha = self.alpha_list[task_id]
            alpha = self.softmax(alpha)
            hidden_states = hidden_states[1:]
            hidden_states = torch.stack(hidden_states)   # [num_hidden_layers, batch_size, seq_len, hidden_size]
            hidden_states = hidden_states.permute(1,2,3,0)  # [batch_size, seq_len, hidden_size, num_hidden_layers]
            sequence_output = torch.matmul(hidden_states, alpha).squeeze(-1) # [batch_size, seq_len, hidden_size]

        sequence_output = self.dropout(sequence_output)
        logits = classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                
                if num_labels == 0: # do parsing, no labels, just heads
                    active_logits = logits.contiguous().view(-1, logits.size(-1))[active_loss]
                else:
                    active_logits = logits.view(-1, num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                if num_labels == 0: # do parsing, no labels, just heads
                    logits = logits.contiguous().view(-1, logits.size(-1)) # do Parsing
                else:
                    logits = logits.view(-1, self.num_labels)
                loss = loss_fct(logits, labels.view(-1))
            outputs = (loss,) + outputs

        if self.do_task_embedding:
            outputs = (alpha_vis, ) + outputs
        elif self.do_alpha:
            outputs = (alpha, ) + outputs
        return outputs


def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

class MTDNNModelAttack(BertPreTrainedModel):
    def __init__(self, config, labels_list, task_list,
                 do_task_embedding=False, do_alpha=False,
                 do_adapter=False, num_adapter_layers=2):
        super(MTDNNModelAttack, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.num_fixed_layers = config.num_hidden_layers - num_adapter_layers

        decoder_modules = []
        for labels ,task in zip(labels_list, task_list):
            if task.startswith("PARSING"):
                decoder_modules.append(DeepBiAffineDecoderV2(config.hidden_size, mlp_dim=300, num_labels=len(labels)))
            else:
                decoder_modules.append(nn.Linear(config.hidden_size, len(labels)))
        self.classifier_list =  nn.ModuleList(decoder_modules)

        if do_alpha:
            init_value = torch.zeros(config.num_hidden_layers, 1)
            self.alpha_list = nn.ModuleList([nn.Parameter(init_value, requires_grad=True)])

            self.softmax = nn.Softmax(dim=0)
            self.labels_list = labels_list

        if do_task_embedding:
            self.task_embedding = nn.Embedding(len(labels_list), config.hidden_size)
            self.w1 = nn.Linear(config.hidden_size, 1)
            self.w2 = nn.Linear(config.hidden_size, 1)

        self.labels_list = [len(x) for x in labels_list]
        self.do_task_embedding = do_task_embedding
        self.do_alpha = do_alpha
        self.do_adapter = do_adapter
        self.softmax = nn.Softmax(dim=0)

        self.crit_label_dst = nn.KLDivLoss(reduction="none")
 

        self.init_weights()


        

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, heads=None, labels=None,
                task_id=0, adapter_ft=False, soft_labels=None, soft_heads=None, gamma=0.5, 
                sequence_output=None, bias=None):
        
        if self.do_adapter:
            
            if adapter_ft and labels is not None:
                for param in self.bert.encoder.parameters():
                    param.requires_grad = False

                for param in self.bert.encoder.layer[-1].parameters():
                    param.requires_grad = True
                
                for param in self.bert.encoder.layer[-2].parameters():
                    param.requires_grad = True
                
          
        if sequence_output is None:
            outputs = self.bert(input_ids, 
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)
            hidden_states = outputs[-1]
            sequence_output = outputs[0]
        
        if bias is not None:
            sequence_output = sequence_output + bias
        
        # sequence_output = hidden_states[-1]

        classifier = self.classifier_list[task_id]
        num_labels = self.labels_list[task_id]
        

        if self.do_task_embedding:
            task_embbedding = self.task_embedding(torch.tensor([task_id]).cuda()).view(-1)
            hidden_states = hidden_states[1:]
            hidden_states = torch.stack(hidden_states)
            # alpha = w1*hidden_states + w2*task_embedding + bias 
            out1 = self.w1(hidden_states) #[num_hidden_layers, batch_size, seq_len, 1]
            task_embedding = task_embedding.expand(hidden_states.size(0), hidden_states.size(1), task_embedding.size(0)) # [num_hidden_layers, batch_size, hidden_size]
            out2 = self.w2(task_embedding) # [num_hidden_layers, batch_size, 1]
            alpha = out1.squeeze(-1) + out2 # [num_hidden_layers, batch_size, seq_len]
            alpha = self.softmax(alpha).permute(1, 0, 2) #[batch_size, num_hidden_layers, seq_len]
            alpha_vis = torch.mean(alpha, dim=0)
            hidden_states = hidden_states.permute(1,0,2,3)  # [batch_size, num_hidden_layers, seq_len, hidden_size]
            alpha = alpha.view(alpha.size() + (1,)).expand(alpha.size() + (hidden_states.size(3), )) #[batch_size, num_hidden_layers, seq_len, hidden_size]
            sequence_output = torch.sum(alpha*hidden_states, dim=1) #[batch_size, seq_len, hidden_size]

        elif self.do_alpha:
            alpha = self.alpha_list[task_id]
            alpha = self.softmax(alpha)
            hidden_states = hidden_states[1:]
            hidden_states = torch.stack(hidden_states)   # [num_hidden_layers, batch_size, seq_len, hidden_size]
            hidden_states = hidden_states.permute(1,2,3,0)  # [batch_size, seq_len, hidden_size, num_hidden_layers]
            sequence_output = torch.matmul(hidden_states, alpha).squeeze(-1) # [batch_size, seq_len, hidden_size]
        raw_sequence_output = sequence_output
        sequence_output = self.dropout(sequence_output)
        if type(classifier) == DeepBiAffineDecoderV2: # parsing task
            logits_arc, logits_label = classifier(sequence_output)
            outputs = (logits_arc, logits_label, raw_sequence_output)

            if labels is not None and heads is not None:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits_arc = logits_arc.contiguous().view(-1, logits_arc.size(-1))[active_loss]
                    active_heads = heads.view(-1)[active_loss]
                    active_logits_label = logits_label.contiguous().view(-1, num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]

                    loss_arc = loss_fct(active_logits_arc, active_heads)
                    loss_labels = loss_fct(active_logits_label, active_labels)
                    loss = loss_arc + loss_labels

                    if soft_labels is not None and soft_heads is not None:
                        active_soft_labels = soft_labels.contiguous().view(-1, num_labels)[active_loss]
                        active_soft_heads = soft_heads.contiguous().view(-1, soft_heads.size(-1))[active_loss]

                        kv_loss_arc = self.crit_label_dst(F.log_softmax(active_logits_arc.float(), dim=-1),
                                                          F.softmax(active_soft_heads.float(), dim=-1)).sum(dim=-1).mean()
                        kv_loss_label = self.crit_label_dst(F.log_softmax(active_logits_label.float(), dim=-1),
                                                            F.softmax(active_soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        kv_loss = kv_loss_arc + kv_loss_label
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                else:
                    logits_arc = logits_arc.contiguous().view(-1, logits_arc.size(-1))
                    heads = heads.view(-1)
                    loss_arc = loss_fct(logits_arc, heads)

                    logits_label = logits_label.contiguous().view(-1, num_labels)
                    labels = labels.view(-1)

                    loss_labels = loss_fct(logits_label, labels)
                    loss = loss_arc + loss_labels
                    if soft_labels is not None and soft_heads is not None:
                        soft_labels = soft_labels.contiguous().view(-1, num_labels)
                        soft_heads = soft_heads.contiguous().view(-1, soft_heads.size(-1))

                        kv_loss_arc = self.crit_label_dst(F.log_softmax(logits_arc.float(), dim=-1),
                                                          F.softmax(soft_heads.float(), dim=-1)).sum(dim=-1).mean()
                    

                        kv_loss_label = self.crit_label_dst(F.log_softmax(logits_label.float(), dim=-1),
                                                            F.softmax(soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        
                        kv_loss = kv_loss_arc + kv_loss_label
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                outputs = (loss, ) + outputs
        else:
            logits = classifier(sequence_output)

            outputs = (logits,raw_sequence_output)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, num_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                    if soft_labels is not None:
                        soft_labels = soft_labels.view(-1, num_labels)[active_loss]
                        kv_loss = self.crit_label_dst(F.log_softmax(active_logits.float(), dim=-1),
                                                     F.softmax(soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                else:
                    logits = logits.view(-1, self.num_labels)
                    if soft_labels is not None:
                        soft_labels = soft_labels.view(-1, num_labels)
                        kv_loss = self.crit_label_dst(F.log_softmax(active_logits.float(), dim=-1),
                                                      F.softmax(soft_labels.float(), dim=-1)).sum(dim=-1).mean()
                        # print("ce loss", loss)
                        # print("kv loss", kv_loss)
                        loss = gamma*loss + (1-gamma)*kv_loss
                    loss = loss_fct(logits, labels.view(-1))
                outputs = (loss,) + outputs

        if self.do_task_embedding:
            outputs = (alpha_vis, ) + outputs
        elif self.do_alpha:
            outputs = (alpha, ) + outputs
        return outputs



class MTDNNModelMobile(BertPreTrainedModel):
    def __init__(self, config, labels_list, task_list):
        super(MTDNNModelMobile, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        decoder_modules = []
        for labels, task in zip(labels_list, task_list):
            if task.startswith("PARSING"):
                decoder_modules.append(DeepBiAffineDecoderV2(config.hidden_size, mlp_dim=300, num_labels=len(labels)))
            else:
                decoder_modules.append(nn.Linear(config.hidden_size, len(labels)))
        
        self.label_list = labels_list
        self.classifier_list = nn.ModuleList(decoder_modules)
        self.num_labels_list = [len(x) for x in labels_list]
        self.task_list = task_list
        self.init_weights()

    @torch.jit.script
    def forward(self, tup:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
               
        input_ids, attention_mask, token_type_ids, task_id = tup
        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=None,
                            head_mask=None,
                            inputs_embeds=None)
        
        sequence_outputs = outputs[0]

        classifier = self.classifier_list[task_id]

        sequence_outputs = self.dropout(sequence_outputs)
        if type(classifier) == DeepBiAffineDecoderV2:
            logits_arc, logits_label = classifier(sequence_outputs)
            outputs = (logits_arc, logits_label) + outputs[2:]
            preds_arc = torch.argmax(preds_arc, dim=2)
            preds_label = torch.argmax(preds_label, dim=2)
            out = torck.stack(preds_arc, preds_label, dim=0)
            return out
        else:
            logits = classifier(sequence_outputs)
            preds = torch.argmax(logits, dim=2)
            return preds

        return outputs
    
    @torch.jit.export
    def get_tasks(self):
        return self.task_list

    @torch.jit.export
    def get_labels(self, task):
        task = task.upper()
        if task not in self.task_list:
            return ["UnSupported task"]
        else:
            task_id = self.task_list[task]
            return self.label_list[task_id]

class BertForNERPOS(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNERPOS, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_ner = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()
    
    def set_class(self, num_pos_labels, num_chunk_labels):
        self.num_pos_labels = num_pos_labels
        self.num_chunk_labels = num_chunk_labels
        self.classifier_pos = nn.Linear(self.hidden_size, self.num_pos_labels)
        self.classifier_chunk = nn.Linear(self.hidden_size, self.num_chunk_labels)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, input_embeds=None, labels=None, 
                pos_labels=None, chunk_labels=None):
        
        outputs = self.bert(input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids,
                     position_ids=position_ids,
                     head_mask=head_mask,
                     input_embeds=input_embeds)
        
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        
        ner_logits = self.classifier_ner(sequence_output)
        outputs = (ner_logits,) + outputs[2:]

      
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

                loss_set = (loss)

                if pos_labels:
                    pos_logits = self.classifier_pos(sequence_output)
                    active_pos_logits = pos_logits.view(-1, self.num_pos_labels)[active_loss]
                    active_pos_labels = pos_labels.view(-1)[active_loss]
                    pos_loss = loss_fct(active_pos_logits, active_pos_labels)
                    
                    loss_set = (loss, pos_loss)
                
                if chunk_labels:
                    chunk_logits = self.classifier_chunk(sequence_output)
                    active_chunk_logits = chunk_logits.view(-1, self.num_chunk_labels)[active_loss]
                    active_chunk_labels = chunk_labels.view(-1)[active_loss]
                    chunk_loss = loss_fct(active_chunk_logits, active_chunk_labels)
                    
                    loss_set = (loss, pos_loss, chunk_loss)
            else:
                loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))

                loss_set = (loss)

                if pos_labels:
                    pos_logits = self.classifier_pos(sequence_output)
                    pos_loss = loss_fct(pos_logits.view(-1, self.num_pos_labels), pos_labels.view(-1))
                    loss_set = (loss, pos_loss)

                if chunk_labels:
                    chunk_logits = self.classifier_chunk(sequence_output)
                    chunk_loss = loss_fct(chunk_logits.view(-1, self.num_chunk_labels), chunk_labels.view(-1))

                    loss_set = (loss, pos_loss, chunk_loss)

            outputs = loss_set + outputs
        return outputs
    
                

                
            




    


