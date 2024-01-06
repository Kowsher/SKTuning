
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


class PrefixEncoder(torch.nn.Module):
    def __init__(self, config, transformer):
        super().__init__()

        self.config = config
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        self.transformer=transformer

        word_embeddings = transformer.word_embeddings
        
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        
        init_token_ids = tokenizer(config.text, return_tensors='pt')['input_ids']
        print("Prefix sequence length: ", init_token_ids.shape[1])
        tokenizer=None

        self.embedding = torch.nn.Embedding(init_token_ids.shape[1], config.hidden_size)

        if config.transform==True:
            self.transform = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.transform=None
     
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


        inputs_embeds = self.embedding(virtual_tokens.to(device))
        inputs_embeds=self.dropout(inputs_embeds)
        outputs = self.transformer(
            inputs_embeds=inputs_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
        )        
        #print('working', outputs.past_key_values)
        #print('working', projection)
        past_key_values=outputs.past_key_values
        if config.transform==True:
        # Apply transformations
            transformed_key_values = []
            for layer in past_key_values:
                key, value = layer
                #print(key.shape, value.shape)
                # Transpose, transform, and transpose back for key
                transformed_key = self.transform(key.transpose(1, 2)).transpose(1, 2)
                transformed_key=self.dropout(transformed_key)
                # Transpose, transform, and transpose back for value
                transformed_value = self.transform(value)
                transformed_value = self.dropout(transformed_value)
                transformed_key_values.append((transformed_key, transformed_value))

            transformed_past_key_values = tuple(transformed_key_values)
        
            return  (transformed_past_key_values, inputs_embeds.shape[0])
        else:
            return  (past_key_values, inputs_embeds.shape[0])



class PrefixForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.transformer =  AutoModel.from_pretrained(config._name_or_path)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        self.score = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.n_layer = config.num_hidden_layers
        self.n_head = config.n_head
        self.n_embd = config.hidden_size // config.n_head
        config.n_embd=self.n_embd

        #print('self.prefix_ids', self.prefix_ids)
        self.prompt_encoder = PrefixEncoder(config, self.transformer)
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
        
        #print('prefix_ids', prefix_ids)
        past_key_values, pre_length =  self.prompt_encoder(self.transformer.device, batch_size)
        #print('prompts', prompts.shape)
        #print('raw_tokens_embedding', raw_tokens_embedding)
        #print('batch_size', batch_size, self.pre_seq_len)
        #inputs_embeds = torch.cat((prompts, raw_tokens_embedding), dim=1)
        prompt_attention_mask = torch.ones(batch_size, pre_length).to(self.transformer.device)
        attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            past_key_values=past_key_values,
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


class PromptEncoder(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()

        self.config = config
        
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        
        init_token_ids = tokenizer(config.text, return_tensors='pt')['input_ids']
        print("Prompt sequence length: ", init_token_ids.shape[1])
        #print("config.pre_seq_len, config.hidden_size", config.pre_seq_len, config.hidden_size)
        tokenizer=None

        self.embedding = torch.nn.Embedding(init_token_ids.shape[1], config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout)

        if config.transform==True:
            self.transform = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        else:
            self.transform=None
            
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

        projection = self.embedding(virtual_tokens.to(device))
        projection=self.dropout(projection)
        
        if config.transform==True:
            projection = self.transform(projection)
            projection=self.dropout(projection)

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
        self.prompt_encoder = PromptEncoder(config, self.transformer.word_embeddings )
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
        raw_tokens_embedding = self.transformer.word_embeddings (input_ids)
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
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
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


class PromptForTokenClassification(PreTrainedModel):
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


        hidden_states = outputs[0][:, prompts.shape[1]:, :]
        #print('hidden_states', hidden_states.shape)
        #print('labels', labels.shape)
        
        hidden_states = self.dropout(hidden_states)
        logits = self.score(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PrefixForTokenClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.transformer =  AutoModel.from_pretrained(config._name_or_path)
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        self.score = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.n_layer = config.num_hidden_layers
        self.n_head = config.n_head
        self.n_embd = config.hidden_size // config.n_head
        config.n_embd=self.n_embd

        #print('self.prefix_ids', self.prefix_ids)
        self.prompt_encoder = PrefixEncoder(config, self.transformer)
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
        
        #print('prefix_ids', prefix_ids)
        past_key_values, pre_length =  self.prompt_encoder(self.transformer.device, batch_size)
        #print('prompts', prompts.shape)
        #print('raw_tokens_embedding', raw_tokens_embedding)
        #print('batch_size', batch_size, self.pre_seq_len)
        #inputs_embeds = torch.cat((prompts, raw_tokens_embedding), dim=1)
        prompt_attention_mask = torch.ones(batch_size, pre_length).to(self.transformer.device)
        attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        
        hidden_states = self.dropout(outputs[0])

        logits = self.score(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

