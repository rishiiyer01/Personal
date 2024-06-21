import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class DictionaryModel(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_hidden_layers):
        super(DictionaryModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        for param in self.bert.parameters():
            param.requires_grad = False
            #for training speedups, could alternatively finetune bert instead of just the output adapter architecture if compute/time is available
        self.char_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.desc_proj=nn.Linear(self.bert.config.hidden_size,hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads),
            num_layers=num_hidden_layers
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.fc = nn.Linear(hidden_size, 27)
        self.Softmax=nn.Softmax(dim=2)
        self.fcmiddle1=nn.Linear(hidden_size,hidden_size)
        self.fcmiddle2=nn.Linear(hidden_size,hidden_size)
        
    def forward(self, masked_word_chars_ids, description_ids, attention_mask):
        # Pass the masked character IDs through BERT
        char_outputs = self.bert(
            masked_word_chars_ids,
            attention_mask=masked_word_chars_ids.ne(0), 
            token_type_ids=None  # Disable token type embeddings
        )
        char_embeddings = char_outputs.last_hidden_state

        # Project the character embeddings to the desired hidden size
        char_embeddings = self.char_proj(char_embeddings)
        char_embeddings=F.gelu(char_embeddings)

        # Pass the description through BERT
        description_outputs = self.bert(
            description_ids,
            attention_mask=attention_mask,
            token_type_ids=None  # Disable token type embeddings
        )
        description_embeddings = description_outputs.last_hidden_state
        description_embeddings=self.desc_proj(description_embeddings)
        description_embeddings=F.gelu(description_embeddings)
        # Pass the character embeddings through the transformer encoder
        #encoder_outputs = self.transformer_encoder(char_embeddings)
        #feed foward after transformer encoder self attention
        #encoder_outputs=self.fcmiddle1(encoder_outputs)
        #encoder_outputs=F.gelu(encoder_outputs)
        # Perform cross attention between the character embeddings and the description embeddings
        cross_attention_outputs ,_  = self.cross_attention(
            query=char_embeddings.transpose(0, 1),  # Transpose to (seq_len, batch_size, hidden_size)
            key=description_embeddings.transpose(0, 1),
            value=description_embeddings.transpose(0, 1)
        )
        cross_attention_outputs = cross_attention_outputs.transpose(0, 1)  # Transpose back to (batch_size, seq_len, hidden_size)
        #feed forward step before mapping to vocab size
        cross_attention_outputs=cross_attention_outputs
        #cross_attention_outputs=self.fcmiddle1(cross_attention_outputs)
        #cross_attention_outputs=F.gelu(cross_attention_outputs)
        # Project the cross attention outputs to the letter vocabulary size (essentially just taking the information from bert and using an encoder decoder architecture to
        # figure out the masked tokens)
        
        output=self.fcmiddle2(cross_attention_outputs)+char_embeddings
        
        output=F.gelu(output)
        output = self.fc(output)
        
        output=self.Softmax(output)
        return output