import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaPreTrainedModel

logger = logging.getLogger(__name__)


class EventDetectionModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.temperature = config.temperature
        self.use_label_semantics = config.use_label_semantics
        self.label_feature_enhanced = config.label_feature_enhanced
        self.label_score_enhanced = config.label_score_enhanced
        self.use_normalize = config.use_normalize
        self.crf_strategy = config.crf_strategy
        self.dist_func = config.dist_func

        self.roberta = RobertaModel(config)
        if not self.use_label_semantics:
            self.fc_layer = nn.Linear(config.hidden_size, config.num_labels)
            self.roberta._init_weights(self.fc_layer)        
            if self.use_normalize:
                self.fc_layer.weight.data = F.normalize(self.fc_layer.weight.data, p=2, dim=-1)
        else:
            self.fc_layer = None
        self.loss_fn = nn.CrossEntropyLoss()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        label_ids=None,
        start_list_list=None,
        end_list_list=None,
        prompt_input_ids=None,
        prompt_attention_mask=None, 
        prompt_start_list=None,
        prompt_end_list=None,
        compute_features_only=False,
        output_prob=False,
        prototypes=None,
    ):

        sequence_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )[0]
        tok_embeds = list()
        for (seq, start_list, end_list) in zip(sequence_output, start_list_list, end_list_list):
            for (start, end) in zip(start_list, end_list):
                embed = torch.mean(seq[start:end], dim=0, keepdim=True)
                tok_embeds.append(embed)
        tok_embeds = torch.cat(tok_embeds, dim=0)
        if self.use_normalize:
            tok_embeds = F.normalize(tok_embeds, p=2, dim=-1)
        assert(tok_embeds.dim()==2 and tok_embeds.size(1)==self.hidden_size)

        if compute_features_only:
            return tok_embeds
        else:
            total_loss, pred_labels, pred_probs = None, None, None
            if prototypes is not None:
                logits = F.normalize(tok_embeds, p=2, dim=-1)@F.normalize(prototypes, p=2, dim=-1).T
            elif not self.use_label_semantics:
                logits = self.fc_layer(tok_embeds)
            else:
                prompt_embeds = self.compute_prompt_embeddings(prompt_input_ids, prompt_attention_mask, prompt_start_list, prompt_end_list)
                if self.use_normalize:
                    prompt_embeds = F.normalize(prompt_embeds, p=2, dim=-1)
                assert(prompt_embeds.dim()==2 and prompt_embeds.size(1)==self.hidden_size and prompt_embeds.size(0)==self.num_labels)
                # logits = F.normalize(tok_embeds, p=2, dim=-1)@F.normalize(prompt_embeds, p=2, dim=-1).T
                logits = tok_embeds@prompt_embeds.T
            
            logits /= self.temperature
            if label_ids is not None:
                total_loss = self.loss_fn(logits, label_ids)
            else:
                if output_prob:
                    pred_probs = F.softmax(logits, dim=-1)
                pred_labels = logits.max(dim=-1).indices
            return total_loss, pred_labels, tok_embeds, pred_probs, logits

    
    def compute_prompt_embeddings(
        self,
        prompt_input_ids=None,
        prompt_attention_mask=None, 
        prompt_start_list=None,
        prompt_end_list=None,
    ):
        prompt_output = self.roberta(
            prompt_input_ids,
            attention_mask=prompt_attention_mask,
        )[0]
        prompt_embeds = list()
        for (seq, start, end) in zip(prompt_output, prompt_start_list, prompt_end_list):
            embed = torch.mean(seq[start:end], dim=0, keepdim=True)
            prompt_embeds.append(embed)
        prompt_embeds = torch.cat(prompt_embeds, dim=0)
        return prompt_embeds
