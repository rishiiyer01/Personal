import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_data_loaders
from model import DictionaryModel
from tqdm import tqdm
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = 27
hidden_size = 256
num_attention_heads = 4
num_hidden_layers = 4
learning_rate = 1e-3
epochs = 5
mask_token_id = tokenizer.mask_token_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_masking(word_chars_ids, mask_token_id, mask_ratio=0.33):
    mask = torch.rand(word_chars_ids.shape) < mask_ratio
    masked_word_chars_ids = word_chars_ids.clone()
    masked_word_chars_ids[mask] = mask_token_id
    return masked_word_chars_ids

def train(train_loader, test_loader):
    model = DictionaryModel(hidden_size, num_attention_heads, num_hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  
    optimizer = Adam(model.parameters(), lr=learning_rate)
    initial_mask_ratio = 0.33
    final_mask_ratio = 0.33

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        current_mask_ratio = initial_mask_ratio + (final_mask_ratio - initial_mask_ratio) * (epoch / (epochs - 1))
        
        for batch in tqdm(train_loader):
            word_chars_ids = batch['word_chars_ids'].to(device)
            combined_ids = batch['combined_ids'].to(device)
            combined_attention_mask = batch['combined_attention_mask'].to(device)

            masked_word_chars_ids = random_masking(word_chars_ids, mask_token_id, current_mask_ratio)

            output = model(masked_word_chars_ids, combined_ids, combined_attention_mask)
            target = word_chars_ids - 1036
            target[target < 0] = 0
            
            loss = criterion(output.transpose(1, 2), target)
            mask = target.ne(0)
            loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                word_chars_ids = batch['word_chars_ids'].to(device)
                combined_ids = batch['combined_ids'].to(device)
                combined_attention_mask = batch['combined_attention_mask'].to(device)

                masked_word_chars_ids = random_masking(word_chars_ids, mask_token_id)

                output = model(masked_word_chars_ids, combined_ids, combined_attention_mask)
                target = word_chars_ids - 1036
                target[target < 0] = 0

                loss = criterion(output.transpose(1, 2), target)
                mask = target.ne(0)
                loss = (loss * mask).sum() / mask.sum()

                test_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
       
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders('dictionary.json', batch_size=16)
    train(train_loader, test_loader)