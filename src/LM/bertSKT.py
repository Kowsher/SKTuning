
import torch
import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import BertModel, BertPreTrainedModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, Seq2SeqLMOutput
import copy


class PrefixEncoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()


        self.config = config

        if config.pooling==True:
            self.W = torch.nn.Linear(config.hidden_size, 1)
            self.projection = torch.nn.Linear(config.pre_seq_len, config.pre_seq_len * config.num_hidden_layers * 2 * config.hidden_size)
        else:
            self.W =None
            self.projection = torch.nn.Linear(config.hidden_size, config.num_hidden_layers * 2 * config.hidden_size)            



    def forward(
        self,
        input_ids=None,
        batch_size=None,

    ):


        if self.config.pooling==True:
            att_w = self.W(input_ids).squeeze(-1)
            projection = self.projection(att_w)
            projection=projection.repeat(batch_size, 1)
        else:
            projection = self.projection(input_ids)
            projection=projection.repeat(batch_size, 1, 1)
        #print('working', projection.shape)
        #print('projection.expand(batch_size, -1)', projection.expand(batch_size, -1).shape)

        return projection

class PromptEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.pooling==True:
            self.W = torch.nn.Linear(config.hidden_size, 1)
            self.projection = torch.nn.Linear(config.pre_seq_len, config.pre_seq_len * config.hidden_size)
        else:
            self.W =None
            self.projection = torch.nn.Linear(config.hidden_size, config.hidden_size)   


    def forward(
        self,
        input_ids=None,
        batch_size=None,

    ):
        #print(batch_size)


        if self.config.pooling==True:
            att_w = self.W(input_ids).squeeze(-1)
            projection = self.projection(att_w).reshape(-1, self.config.pre_seq_len, self.config.hidden_size)
            projection=projection.repeat(batch_size, 1, 1)
        else:
            projection = self.projection(input_ids)
            projection=projection.repeat(batch_size, 1, 1)
        
        #print('working', projection.shape)

        return projection



class PrefixForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.transformer =  AutoModel.from_pretrained(config._name_or_path)
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        #prefix_ids = config.tokenizer(config.prefix, return_tensors='pt')['input_ids']
        #print('prefix_ids', prefix_ids)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.transformer.parameters():
            param.requires_grad = False

        global prefix_ids
        prefix_id = config.tokenizer(config.prefix, return_tensors='pt')['input_ids']
        prefix_ids = self.transformer(prefix_id)[0]
        config.prefix_ids=None
        config.tokenizer=None

        self.pre_seq_len = prefix_ids.shape[1]
        print('prefix sequence length: ', self.pre_seq_len )
        self.n_layer = config.num_hidden_layers
        self.n_head = config.n_head
        self.n_embd = config.hidden_size // config.n_head
        config.pre_seq_len = self.pre_seq_len

        #print('self.prefix_ids', self.prefix_ids)
        self.prefix_encoder = PrefixEncoder(config)
        self.config = config

    def get_prompt(self, prefix_idss, batch_size):
        #print(prefix_ids)
        #prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
   


        past_key_values = self.prefix_encoder(prefix_idss.to(self.transformer.device), batch_size)
        #print('past_key_values', past_key_values.shape)

        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        #print(past_key_values.shape)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

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
        #print('self.prefix_ids', self.prefix_ids)
        if prefix_ids is not None:
            past_key_values = self.get_prompt(prefix_idss=prefix_ids, batch_size=batch_size)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        else:
            past_key_values=None



        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        #print('here', outputs[0].shape)


        #pooled_output = self.dropout(pooled_output)
        

        logits = self.classifier(outputs.last_hidden_state)
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


class PromptForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.transformer =  AutoModel.from_pretrained(config._name_or_path)
        self.embeddings = self.transformer.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        #prefix_ids = config.tokenizer(config.prefix, return_tensors='pt')['input_ids']
        #print('prefix_ids', prefix_ids)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.transformer.parameters():
            param.requires_grad = False

        global prefix_ids
        prefix_id = config.tokenizer(config.prompt, return_tensors='pt')['input_ids']
        prefix_ids = self.transformer(prefix_id)[0]
        config.prefix_ids=None
        config.tokenizer=None

        self.pre_seq_len = prefix_ids.shape[1]
        print('prompt sequence length: ', self.pre_seq_len )
        self.n_layer = config.num_hidden_layers
        self.n_head = config.n_head
        self.n_embd = config.hidden_size // config.n_head
        config.pre_seq_len = self.pre_seq_len

        #print('self.prefix_ids', self.prefix_ids)
        self.prompt_encoder = PromptEncoder(config)
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
        raw_tokens_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        #print('prefix_ids', prefix_ids)
        prompts =  self.prompt_encoder(prefix_ids.to(self.transformer.device), batch_size)
        #print('prompts', prompts)
        #print('raw_tokens_embedding', raw_tokens_embedding)
        #print('batch_size', batch_size, self.pre_seq_len)
        inputs_embeds = torch.cat((prompts, raw_tokens_embedding), dim=1)
        prompt_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.transformer.device)
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


        
        logits = self.classifier(outputs.last_hidden_state)
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