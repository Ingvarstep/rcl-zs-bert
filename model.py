import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from transformers import AutoTokenizer, AutoModel, AutoConfig



class RCL(nn.Module):  
    def __init__(self, bert_model, temp, device, num_label=4, dropout=0.5, alpha=0.15, special_tokenizer=None, rel_desc = False):
        super(RCL, self).__init__()
        self.config = AutoConfig.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        if special_tokenizer is not None:
            self.model.resize_token_embeddings(len(special_tokenizer)) 
        # Load model from HuggingFace Hub
        
        self.rel_desc = rel_desc
        if self.rel_desc:
            self.sent_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
            self.sent_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
            self.dist_func = 'cosine'
            self.gamma = 7.5
            self.margin = torch.tensor(self.gamma)
            self.fc_layer = nn.Linear(self.model.hidden_size*3, self.sent_model.hidden_size)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
#         self.dense_mark = nn.Linear(self.config.hidden_size*2, self.config.hidden_size*2)
        self.activation = nn.Tanh()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device
        
        # classification
        self.alpha = alpha
        self.classifier = nn.Linear(self.config.hidden_size*3, num_label)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, entity_idx=None, classify_labels=None, use_cls=False, rel_descs = None):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len) (32*2, 32)
        outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        )
        # class
        
        bert_emb = outputs[0]
        se_length = bert_emb.size(1)
        bert_emb = bert_emb.view((batch_size, num_sent, se_length, bert_emb.size(-1)))
        bert_hidden = bert_emb[:, 0] #[bs, length, hidden]
        
        relation_hidden = []
        for i in range(len(entity_idx)):
            head_idx = entity_idx[i][0][0]
            tail_idx = entity_idx[i][0][1]
            cls_token = bert_hidden[i][0].view(1, -1)
            head_entity = bert_hidden[i][head_idx].view(1, -1)
            tail_entity = bert_hidden[i][tail_idx].view(1, -1)
            relation = torch.cat([cls_token, head_entity, tail_entity], dim=-1)
            relation_hidden.append(relation)
        relation_hidden = torch.cat(relation_hidden, dim=0)      
        
        relation_hidden = self.dropout(self.activation(relation_hidden))
        logit = self.classifier(relation_hidden)
        loss_ce = nn.CrossEntropyLoss()
        ce_loss = loss_ce(logit, classify_labels.view(-1))

        #:TODO:add prediciton of relational embedding

        if self.rel_desc and rel_descs:
            # Tokenize sentences
            rel_desc_encoded_input = self.sent_tokenizer(rel_descs, padding=True, truncation=True, return_tensors='pt')

            # Compute token embeddings
            sent_model_output = self.sent_model(**rel_desc_encoded_input)
            relation_embeddings = self.fc_layer(relation_hidden)
            # Perform pooling. In this case, max pooling.
            rel_desc_embeddings = self.dropout(utils.mean_pooling(sent_model_output, rel_desc_encoded_input['attention_mask']))
            
            gamma = self.margin.to(self.device)
            zeros = torch.tensor(0.).to(self.device)
            for a, b in enumerate(relation_embeddings):
                max_val = torch.tensor(0.).to(self.device)
                for i, j in enumerate(rel_desc_embeddings):
                    if a==i:
                        if self.dist_func == 'inner':
                            pos = torch.dot(b, j).to(self.device)
                        elif self.dist_func == 'euclidian':
                            pos = torch.dist(b, j, 2).to(self.device)
                        elif self.dist_func == 'cosine':
                            pos = torch.cosine_similarity(b, j, dim=0).to(self.device)
                    else:
                        if self.dist_func == 'inner':
                            tmp = torch.dot(b, j).to(self.device)
                        elif self.dist_func == 'euclidian':
                            tmp = torch.dist(b, j, 2).to(self.device)
                        elif self.dist_func == 'cosine':
                            tmp = torch.cosine_similarity(b, j, dim=0).to(self.device)
                        if tmp > max_val:
                            if classify_labels[a] != classify_labels[i]:
                                max_val = tmp
                            else:
                                continue
                neg = max_val.to(self.device)
#                 print(f'neg={neg}')
#                 print(f'neg-pos+gamma={neg - pos + gamma}')
#                 print('===============')
                if self.dist_func == 'inner' or self.dist_func == 'cosine':
                    ce_loss += (torch.max(zeros, neg - pos + gamma) * (1-self.alpha))
                elif self.dist_func == 'euclidian':
                    ce_loss += (torch.max(zeros, pos - neg + gamma) * (1-self.alpha))
        # cls
        if use_cls:
            last_hidden = outputs[0] # last_hidden
            cls_hidden = last_hidden[:, 0]
            pooler_output = cls_hidden.view((batch_size, num_sent, cls_hidden.size(-1))) # (bs, num_sent, hidden)
            pooler_output = self.dense(pooler_output)
            pooler_output = self.activation(pooler_output)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
                   
        # marker
        elif not use_cls and entity_idx is not None:
            last_hidden = outputs[0] # last_hidden [bs, sent_length, hidden]
            last_hidden = self.dense(last_hidden)
            last_hidden = self.activation(last_hidden)
            sent_length = last_hidden.size(1)
            last_hidden = last_hidden.view((batch_size, num_sent, sent_length, last_hidden.size(-1)))
            sent1_hidden, sent2_hidden = last_hidden[:, 0], last_hidden[:, 1]  #[bs, sent_length, hidden]
            z1 = []
            z2 = []
            for i in range(len(entity_idx)):
                sent1_head_idx, sent1_tail_idx = entity_idx[i][0][0], entity_idx[i][0][1]
                sent2_head_idx, sent2_tail_idx = entity_idx[i][1][0], entity_idx[i][1][1]
                #sent1
                sent1_head_entity = sent1_hidden[i][sent1_head_idx]
                sent1_tail_entity = sent1_hidden[i][sent1_tail_idx]
                #sent2
                sent2_head_entity = sent2_hidden[i][sent2_head_idx]
                sent2_tail_entity = sent2_hidden[i][sent2_tail_idx]
                
                sent1_relation_expresentation = torch.cat([sent1_head_entity, sent1_tail_entity], dim=-1)
                sent2_relation_expresentation = torch.cat([sent2_head_entity, sent2_tail_entity], dim=-1)
                z1.append(sent1_relation_expresentation.unsqueeze(0))
                z2.append(sent2_relation_expresentation.unsqueeze(0))
            z1 = torch.cat(z1, dim=0)
            z2 = torch.cat(z2, dim=0)
            
        cos_sim = self.cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp
        con_labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        cl_loss = loss_fct(cos_sim, con_labels) 
        
        return ce_loss + self.alpha*cl_loss

    
    def encode(self, input_ids, attention_mask, token_type_ids=None, entity_idx=None, use_cls=False, use_desc = False):
        outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        )
        if use_cls:
            cls_token = outputs[0][:, 0]
            pooler_output = self.dense(cls_token)
            sent_emb = self.activation(pooler_output)
            return sent_emb
        elif not use_cls and entity_idx is not None and not use_desc:
            last_hidden = outputs[0] # last_hidden [1, sent_length, hidden]
            #test
            last_hidden = self.dense(last_hidden)
            last_hidden = self.activation(last_hidden)
            
            head_idx = entity_idx[0]
            tail_idx = entity_idx[1]
            head_entity = last_hidden[0][head_idx]
            tail_entity = last_hidden[0][tail_idx]
            sent_emb = torch.cat([head_entity, tail_entity], dim=-1)
            return sent_emb.unsqueeze(0)
        elif not use_cls and entity_idx is not None:
            last_hidden = outputs[0] # last_hidden [1, sent_length, hidden]
            
            head_idx = entity_idx[0]
            tail_idx = entity_idx[1]
            head_entity = last_hidden[0][head_idx]
            tail_entity = last_hidden[0][tail_idx]
            cls_token = last_hidden[0][0]
            relation_hidden = torch.cat([cls_token, head_entity, tail_entity], dim=-1)
            relation_hidden = self.dropout(self.activation(relation_hidden))
            relation_embeddings = self.fc_layer(relation_hidden)
            return relation_embeddings.unsqueeze(0)