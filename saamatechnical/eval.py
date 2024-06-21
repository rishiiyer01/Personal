import torch
from transformers import BertTokenizer
from model import DictionaryModel

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
hidden_size = 256
num_attention_heads = 4
num_hidden_layers = 4
model = DictionaryModel(hidden_size, num_attention_heads, num_hidden_layers).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_masked_word(masked_word):
    processed_word = []
    for char in masked_word:
        if char == '_':
            processed_word.append(tokenizer.mask_token_id)
        else:
            processed_word.extend(tokenizer.encode(char, add_special_tokens=False))
    return processed_word

def inference(masked_word, description):
    # Preprocess the masked word
    masked_word_chars_ids = preprocess_masked_word(masked_word)
    masked_word_chars_tensor = torch.tensor(masked_word_chars_ids).unsqueeze(0).to(device)
    
    # Tokenize the description
    description_ids = tokenizer(description, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)
    attention_mask = description_ids.ne(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(masked_word_chars_tensor, description_ids, attention_mask)
        predictions = output.argmax(dim=-1).squeeze(0)

    # Convert predictions to characters
    predicted_chars = []
    for i, (orig_char, pred) in enumerate(zip(masked_word, predictions)):
        if orig_char == '_':
            pred_char = tokenizer.decode(pred+1036)
            predicted_chars.append(pred_char)
        else:
            orig_char=tokenizer.decode(pred+1036)
            predicted_chars.append(orig_char)
    
    predicted_word = ''.join(predicted_chars)

    return predicted_word

# Example usage
masked_word = "spine"
description = "the series of vertebrae forming the axis of the skeleton and protecting the spinal cord"

predicted_word = inference(masked_word, description)
print("Masked Word:", masked_word)
print("Description:", description)
print("Predicted Word:", predicted_word)
