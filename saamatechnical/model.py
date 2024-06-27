import torch
import torch.nn as nn
from transformers import BertModel
\
import torch.nn.functional as F

class DictionaryModel(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_hidden_layers):
        super(DictionaryModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.bert=RobertaModel.from_pretrained("FacebookAI/roberta-base")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        for param in self.bert.parameters():
            param.requires_grad = False
            #for training speedups, could alternatively finetune bert instead of just the output adapter architecture if compute/time is available
        self.char_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.desc_proj=nn.Linear(self.bert.config.hidden_size,hidden_size)
        self.query_proj=nn.Linear(hidden_size,hidden_size)
        self.key_proj=nn.Linear(hidden_size,hidden_size)
        self.value_proj=nn.Linear(hidden_size,hidden_size)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.fc = nn.Linear(hidden_size, 27)
        self.Softmax=nn.Softmax(dim=2)
        self.fcmiddle1=nn.Linear(hidden_size,hidden_size)
        self.fcmiddle2=nn.Linear(hidden_size,hidden_size)

    def forward(self, masked_word_chars_ids, description_ids, attention_mask):
        # Pass the masked character IDs through BERT
        #char_outputs = self.bert(
        #    masked_word_chars_ids,
        #    attention_mask=masked_word_chars_ids.ne(0), 
        #    token_type_ids=None  # Disable token type embeddings
        #)
        #char_embeddings = char_outputs.last_hidden_state

        # Project the character embeddings to the desired hidden size
        #char_embeddings = self.char_proj(char_embeddings)
        #char_embeddings=F.gelu(char_embeddings)
        #concatenating the masked word to its description prior to BERT for word-description awareness/context
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size=masked_word_chars_ids.shape[0]
        sep_token_ids=torch.ones((batch_size,1)).int().to(device)
        
        
        description_ids=torch.cat((masked_word_chars_ids,sep_token_ids,description_ids),dim=1)
        #concatenating attention masks as well to match
        attention_mask=torch.cat((masked_word_chars_ids.ne(0),sep_token_ids.ne(0),attention_mask),dim=1)
        
        # Pass the description+word characters through BERT
        description_outputs = self.bert(
            description_ids,
            attention_mask=attention_mask,
            token_type_ids=None  # Disable token type embeddings
        )
        description_embeddings = description_outputs.last_hidden_state
        char_length=masked_word_chars_ids.shape[1]
        char_embeddings=description_embeddings[:,:char_length,:]
        description_embeddings=self.desc_proj(description_embeddings)
        char_embeddings=self.char_proj(char_embeddings)
        description_embeddings=F.gelu(description_embeddings)
        char_embeddings=F.gelu(char_embeddings)
        #query_tensor=self.query_proj(char_embeddings)
        #key_tensor=self.key_proj(description_embeddings)
        #value_tensor=self.value_proj(description_embeddings)
        
        # Perform cross attention between the character embeddings and the description embeddings
        # Doing cross attention with the original word sequence
        cross_attention_outputs , _  = self.cross_attention(
            query=char_embeddings.transpose(0, 1),  # Transpose to (seq_len, batch_size, hidden_size)
           key=description_embeddings.transpose(0, 1),
           value=description_embeddings.transpose(0, 1)
        )
        cross_attention_outputs = cross_attention_outputs.transpose(0, 1)  # Transpose back to (batch_size, seq_len, hidden_size)
        #feed forward step before mapping to vocab size
        #cross_attention_outputs=cross_attention_outputs
        cross_attention_outputs=self.fcmiddle1(cross_attention_outputs+char_embeddings)
        #cross_attention_outputs=F.gelu(cross_attention_outputs)
        # Project the cross attention outputs to the letter vocabulary size (essentially just taking the information from bert and using a small encoder decoder architecture to
        # figure out the masked tokens)
        
        output=self.fcmiddle2(cross_attention_outputs)

        output=F.gelu(output)
        output = self.fc(output)

        #output=self.Softmax(output)
        return output