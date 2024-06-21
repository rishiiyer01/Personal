import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_data_loaders
from model import DictionaryModel
from tqdm import tqdm
# Hyperparameters
vocab_size = 27 # 26 letters of the alphabet + pad token
hidden_size = 256
num_attention_heads = 4
num_hidden_layers = 4
learning_rate = 1e-3
epochs = 5
mask_token_id = 103  # mask token id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_masking(word_chars_ids, mask_token_id, mask_ratio=0.33):
    mask = torch.rand(word_chars_ids.shape) < mask_ratio
    masked_word_chars_ids = word_chars_ids.clone()
    masked_word_chars_ids[mask] = mask_token_id
    return masked_word_chars_ids

def train(train_loader,test_loader):
    model = DictionaryModel( hidden_size, num_attention_heads, num_hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            word_chars_ids = batch['word_chars_ids'].to(device)
            description_ids = batch['description_ids'].to(device)
            attention_mask = batch['description_ids'].ne(0).to(device)  # Generate attention mask

            masked_word_chars_ids = random_masking(word_chars_ids, mask_token_id)

            optimizer.zero_grad()

            output = model(masked_word_chars_ids, description_ids, attention_mask)
            #target=word_chars_ids
            target = word_chars_ids - 1036  # Convert token IDs to 0-26 range
            target[target < 0] = 0  # Set padding tokens to 0

            loss = criterion(output.transpose(1, 2), target)  # Compute cross-entropy loss
            mask = target.ne(0)  # Create a mask for non-padding tokens
            loss = (loss * mask).sum() / mask.sum()  # Apply the mask and compute the average loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_loss=0
        for batch in tqdm(test_loader):
            with torch.no_grad():
                word_chars_ids = batch['word_chars_ids'].to(device)
                description_ids = batch['description_ids'].to(device)
                attention_mask = batch['description_ids'].ne(0).to(device)  # Generate attention mask

                masked_word_chars_ids = random_masking(word_chars_ids, mask_token_id)

                output = model(masked_word_chars_ids, description_ids, attention_mask)
                target = word_chars_ids - 1036  # Convert token IDs to 0-26 range
                target[target < 0] = 0  # Set padding tokens to 0

                loss = criterion(output.transpose(1, 2), target)  # Compute cross-entropy loss
                mask = target.ne(0)  # Create a mask for non-padding tokens
                loss = (loss * mask).sum() / mask.sum()  # Apply the mask and compute the average loss

                test_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_loss=test_loss/len(test_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), 'model.pth')

if name == 'main':
    train_loader,test_loader = get_data_loaders('dictionary.json', batch_size=16)
    train(train_loader,test_loader)