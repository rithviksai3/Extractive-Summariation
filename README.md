# Extractive-Summariation
Code for summarization using Pegasus Model:  
import torch 
from transformers import PegasusForConditionalGeneration, PegasusTokenizer 
from datasets import load_dataset 
from rouge_score import rouge_scorer 
import numpy as np 
# Load the SAMSum dataset 
dataset = load_dataset('samsum', split='train[:30%]') 
validation_dataset = load_dataset('samsum', split='validation[:30%]') 
# Initialize the Pegasus model and tokenizer 
model_name = 'google/pegasus-cnn_dailymail' 
model = PegasusForConditionalGeneration.from_pretrained(model_name) 
tokenizer = PegasusTokenizer.from_pretrained(model_name) 
# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device) 
# Preprocess the data 
def preprocess_data(examples): 
inputs = tokenizer(examples['dialogue'], max_length=512, truncation=True, padding='max_length') 
targets = tokenizer(examples['summary'], max_length=128, truncation=True, 
padding='max_length') 
return { 
'input_ids': inputs['input_ids'], 
'attention_mask': inputs['attention_mask'], 
'labels': targets['input_ids'] 
} 
train_dataset = dataset.map(preprocess_data, batched=True, remove_columns=['id', 'dialogue', 
'summary']) 
val_dataset = validation_dataset.map(preprocess_data, batched=True, remove_columns=['id', 
'dialogue', 'summary']) 
# DataLoader 
from torch.utils.data import DataLoader 
def collate_fn(batch): 
input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch]) 
attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch]) 
labels = torch.stack([torch.tensor(x['labels']) for x in batch]) 
return { 
'input_ids': input_ids, 
'attention_mask': attention_mask, 
'labels': labels 
} 
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn) 
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn) 
# Define the optimizer and loss function 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) 
# Training loop 
for epoch in range(1): 
model.train() 
total_loss = 0 
for batch in train_loader: 
input_ids = batch['input_ids'].to(device) 
attention_mask = batch['attention_mask'].to(device) 
labels = batch['labels'].to(device) 
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
loss = outputs.loss 
total_loss += loss.item() 
optimizer.zero_grad() 
loss.backward() 
optimizer.step() 
avg_train_loss = total_loss / len(train_loader) 
print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}') 
model.eval() 
val_loss = 0 
with torch.no_grad(): 
for batch in val_loader: 
input_ids = batch['input_ids'].to(device) 
attention_mask = batch['attention_mask'].to(device) 
labels = batch['labels'].to(device) 
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) 
loss = outputs.loss 
val_loss += loss.item() 
avg_val_loss = val_loss / len(val_loader) 
print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}') 
# Generate summaries for custom inputs 
def generate_summary(input_text): 
inputs = tokenizer(input_text, max_length=512, truncation=True, padding='max_length', 
return_tensors='pt') 
input_ids = inputs['input_ids'].to(device) 
attention_mask = inputs['attention_mask'].to(device) 
summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, 
max_length=60, num_beams=4, length_penalty=2.0, early_stopping=True) 
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True) 
return summary_text 
# Example usage 
custom_input = "Aluko nets winner with ten minutes remaining at KC Stadium. Tomas Marek put 
visitors into shock lead after two minutes. Ahmed Elmohamady equalised for the hosts. Steve Bruce's 
side now await Europa League play-off." 
custom_summary = generate_summary(custom_input) 
print(custom_summary) 
# Evaluate ROUGE score on the validation set 
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) 
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []} 
model.eval() 
with torch.no_grad(): 
for example in validation_dataset: 
input_text = example['dialogue'] 
reference_summary = example['summary'] 
generated_summary = generate_summary(input_text) 
scores = scorer.score(reference_summary, generated_summary) 
for key in rouge_scores.keys(): 
rouge_scores[key].append(scores[key].fmeasure) 
# Calculate average ROUGE scores 
avg_rouge_scores = {key: np.mean(rouge_scores[key]) for key in rouge_scores.keys()} 
print("Average ROUGE scores on the validation set:") 
print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}") 
print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}") 
print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}") 

# Rouge scores: 
Average ROUGE scores on the validation set: 
ROUGE-1: 0.4024 
ROUGE-2: 0.1740 
ROUGE-L: 0.3148
