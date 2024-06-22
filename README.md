# Extractive-Summariation
Code for summarization using Pegasus Model:<br/>
import torch<br/>
from transformers import PegasusForConditionalGeneration, PegasusTokenizer<br/>
from datasets import load_dataset<br/>
from rouge_score import rouge_scorer<br/> 
import numpy as np<br/> 
# Load the SAMSum dataset__ 
dataset = load_dataset('samsum', split='train[:30%]')__ 
validation_dataset = load_dataset('samsum', split='validation[:30%]')__ 
# Initialize the Pegasus model and tokenizer__
model_name = 'google/pegasus-cnn_dailymail'__ 
model = PegasusForConditionalGeneration.from_pretrained(model_name)__ 
tokenizer = PegasusTokenizer.from_pretrained(model_name)__ 
# Device configuration__ 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')__ 
model.to(device)__ 
# Preprocess the data__ 
def preprocess_data(examples):__ 
inputs = tokenizer(examples['dialogue'], max_length=512, truncation=True, padding='max_length')__ 
targets = tokenizer(examples['summary'], max_length=128, truncation=True,padding='max_length')__ 
return {__ 
'input_ids': inputs['input_ids'],__ 
'attention_mask': inputs['attention_mask'],__ 
'labels': targets['input_ids']__ 
}__ 
train_dataset = dataset.map(preprocess_data, batched=True, remove_columns=['id', 'dialogue', 'summary'])__ 
val_dataset = validation_dataset.map(preprocess_data, batched=True, remove_columns=['id', 'dialogue', 'summary'])__ 
# DataLoader__ 
from torch.utils.data import DataLoader__ 
def collate_fn(batch):__ 
input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])__ 
attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])__ 
labels = torch.stack([torch.tensor(x['labels']) for x in batch])__ 
return {__ 
'input_ids': input_ids,__ 
'attention_mask': attention_mask,__ 
'labels': labels__ 
}__ 
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)__ 
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)__ 
# Define the optimizer and loss function__ 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)__ 
# Training loop__ 
for epoch in range(1):__ 
model.train()__ 
total_loss = 0__ 
for batch in train_loader:__ 
input_ids = batch['input_ids'].to(device)__ 
attention_mask = batch['attention_mask'].to(device)__ 
labels = batch['labels'].to(device)__ 
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)__ 
loss = outputs.loss__ 
total_loss += loss.item()__ 
optimizer.zero_grad()__ 
loss.backward()__ 
optimizer.step()__ 
avg_train_loss = total_loss / len(train_loader)__ 
print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')__ 
model.eval()__ 
val_loss = 0__ 
with torch.no_grad():__ 
for batch in val_loader:__ 
input_ids = batch['input_ids'].to(device)__ 
attention_mask = batch['attention_mask'].to(device)__ 
labels = batch['labels'].to(device)__ 
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)__ 
loss = outputs.loss__ 
val_loss += loss.item()__ 
avg_val_loss = val_loss / len(val_loader)__ 
print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')__ 
# Generate summaries for custom inputs__ 
def generate_summary(input_text):__ 
inputs = tokenizer(input_text, max_length=512, truncation=True, padding='max_length',return_tensors='pt')__ 
input_ids = inputs['input_ids'].to(device)__ 
attention_mask = inputs['attention_mask'].to(device)__ 
summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=60, num_beams=4, length_penalty=2.0, early_stopping=True)__ 
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)__ 
return summary_text__ 
# Example usage__ 
custom_input = "Aluko nets winner with ten minutes remaining at KC Stadium. Tomas Marek put visitors into shock lead after two minutes. Ahmed Elmohamady equalised for the hosts. Steve Bruce's side now await Europa League play-off."__ 
custom_summary = generate_summary(custom_input)__ 
print(custom_summary)__ 
# Evaluate ROUGE score on the validation set__ 
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)__ 
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}__ 
model.eval()__ 
with torch.no_grad():__ 
for example in validation_dataset:__ 
input_text = example['dialogue']__ 
reference_summary = example['summary']__ 
generated_summary = generate_summary(input_text)__ 
scores = scorer.score(reference_summary, generated_summary)__ 
for key in rouge_scores.keys():__ 
rouge_scores[key].append(scores[key].fmeasure)__ 
# Calculate average ROUGE scores__ 
avg_rouge_scores = {key: np.mean(rouge_scores[key]) for key in rouge_scores.keys()}__ 
print("Average ROUGE scores on the validation set:")__ 
print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")__ 
print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")__ 
print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")__ 

# Rouge scores:__ 
Average ROUGE scores on the validation set:__ 
ROUGE-1: 0.4024__ 
ROUGE-2: 0.1740__ 
ROUGE-L: 0.3148__
