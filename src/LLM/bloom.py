
"""PyTorch b model."""


import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import BloomConfig, BloomPreTrainedModel, BloomModel, AutoConfig, PreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, Seq2SeqLMOutput



logger = logging.get_logger(__name__)


import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        self.config = config
        
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        
        init_token_ids = tokenizer(config.prompt, return_tensors='pt')['input_ids']
        #print("config.pre_seq_len, config.hidden_size", config.pre_seq_len, config.hidden_size)
        tokenizer=None

        self.embedding = torch.nn.Embedding(init_token_ids.shape[1], config.hidden_size)

     
        init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)

        word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
        word_embedding_weights = word_embedding_weights.to(torch.float32)
        #print('word_embedding_weights', word_embedding_weights.shape)
        #print('word_embedding_weights', word_embedding_weights.squeeze(0).shape)
        self.embedding.weight = torch.nn.Parameter(word_embedding_weights.squeeze(0))  
        global virtual_tokens 
        virtual_tokens = torch.arange(0, init_token_ids.shape[1])
        

    def forward(
        self,
        device=None,
        batch_size=None,

    ):
        #print('hi', virtual_tokens)

        projection = self.embedding(virtual_tokens.to(device))
        #projection=projection.repeat(batch_size, 1, 1)
        
        #print('working', projection.shape)
        #print('working', projection)

        return projection.repeat(batch_size, 1, 1)



class PromptForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.transformer =  AutoModel.from_pretrained(config._name_or_path)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        #prefix_ids = config.tokenizer(config.prefix, return_tensors='pt')['input_ids']
        #print('prefix_ids', prefix_ids)
        self.score = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.n_layer = config.num_hidden_layers
        self.n_head = config.n_head
        self.n_embd = config.hidden_size // config.n_head

        #print('self.prefix_ids', self.prefix_ids)
        self.prompt_encoder = PromptEncoder(config, self.transformer.word_embeddings)
        self.config = config


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_tokens_embedding = self.transformer.word_embeddings(input_ids)
        #print('prefix_ids', prefix_ids)
        prompts =  self.prompt_encoder(self.transformer.device, batch_size)
        #print('prompts', prompts.shape)
        #print('raw_tokens_embedding', raw_tokens_embedding)
        #print('batch_size', batch_size, self.pre_seq_len)
        inputs_embeds = torch.cat((prompts, raw_tokens_embedding), dim=1)
        prompt_attention_mask = torch.ones(batch_size, prompts.shape[1]).to(self.transformer.device)
        attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)

        outputs = self.transformer(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )


        
        hidden_states = self.dropout(outputs[0])

        logits = self.score(hidden_states)
        logits = torch.mean(logits, dim=1)


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
