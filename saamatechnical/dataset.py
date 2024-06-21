import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

# Initialize tokenizer and model outside the class to avoid reloading it multiple times
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class DictionaryDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            self.entries = [
                ([tokenizer.encode(char, add_special_tokens=False) for char in word], data[word], word)
                for word in data.keys()
            ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        word_chars, description, full_word = self.entries[idx]
        # Flatten the list of lists for character IDs
        word_chars_ids = [item for sublist in word_chars for item in sublist]
        word_chars_tensor = torch.tensor(word_chars_ids)
        
        # Combine word and description for tokenization
        combined_input = full_word + " [SEP] " + description
        combined_tokens = tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        return {
            'word_chars_ids': word_chars_tensor,
            'combined_ids': combined_tokens.input_ids.squeeze(0),
            'combined_attention_mask': combined_tokens.attention_mask.squeeze(0),
            'full_word': full_word
        }

def collate_fn(batch):
    word_chars_ids = pad_sequence([item['word_chars_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    combined_ids = pad_sequence([item['combined_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    combined_attention_mask = pad_sequence([item['combined_attention_mask'] for item in batch], batch_first=True, padding_value=0)
    full_words = [item['full_word'] for item in batch]
    return {
        'word_chars_ids': word_chars_ids,
        'combined_ids': combined_ids,
        'combined_attention_mask': combined_attention_mask,
        'full_words': full_words
    }

def get_data_loaders(json_file, batch_size=16, test_size=0.2):
    dataset = DictionaryDataset(json_file)
    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    
    return train_loader, test_loader

# for testing
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders('dictionary.json')
    print("Train loader:")
    for batch in train_loader:
        print("Word chars shape:", batch['word_chars_ids'].shape)
        print("Combined tokens shape:", batch['combined_ids'].shape)
        print("Combined attention mask shape:", batch['combined_attention_mask'].shape)
        break
    print("Test loader:")
    for batch in test_loader:
        print("Word chars shape:", batch['word_chars_ids'].shape)
        print("Combined tokens shape:", batch['combined_ids'].shape)
        print("Combined attention mask shape:", batch['combined_attention_mask'].shape)
        break