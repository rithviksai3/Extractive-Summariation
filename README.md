# Extractive-Summariation
# Code for summarization using Pegasus Model:<br/>
## import libraries
import torch<br/>
from transformers import PegasusForConditionalGeneration, PegasusTokenizer<br/>
from datasets import load_dataset<br/>
from rouge_score import rouge_scorer<br/> 
import numpy as np<br/> 
## Load the SAMSum dataset<br/>
dataset = load_dataset('samsum', split='train[:30%]')<br/>
validation_dataset = load_dataset('samsum', split='validation[:30%]')<br/> 
## Initialize the Pegasus model and tokenizer<br/>
model_name = 'google/pegasus-cnn_dailymail'<br/>
model = PegasusForConditionalGeneration.from_pretrained(model_name)<br/>
tokenizer = PegasusTokenizer.from_pretrained(model_name)<br/> 
## Device configuration<br/> 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')<br/> 
model.to(device)<br/> 
## Preprocess the data<br/>
def preprocess_data(examples):<br/>
inputs = tokenizer(examples['dialogue'], max_length=512, truncation=True, padding='max_length')<br/>
targets = tokenizer(examples['summary'], max_length=128, truncation=True,padding='max_length')<br/>
return {<br/>
'input_ids': inputs['input_ids'],<br/>
'attention_mask': inputs['attention_mask'],<br/>
'labels': targets['input_ids']<br/>
}<br/>
train_dataset = dataset.map(preprocess_data, batched=True, remove_columns=['id', 'dialogue', 'summary'])<br/>
val_dataset = validation_dataset.map(preprocess_data, batched=True, remove_columns=['id', 'dialogue', 'summary'])<br/>
## DataLoader<br/>
from torch.utils.data import DataLoader<br/> 
def collate_fn(batch):<br/>
input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])<br/>
attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])<br/>
labels = torch.stack([torch.tensor(x['labels']) for x in batch])<br/>
return {<br/>
'input_ids': input_ids,<br/>
'attention_mask': attention_mask,<br/>
'labels': labels<br/>
}<br/> 
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)<br/>
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)<br/> 
## Define the optimizer and loss function<br/>
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)<br/>
## Training loop<br/>
for epoch in range(1):<br/>
model.train()<br/>
total_loss = 0<br/>
for batch in train_loader:<br/>
input_ids = batch['input_ids'].to(device)<br/>
attention_mask = batch['attention_mask'].to(device)<br/>
labels = batch['labels'].to(device)<br/>
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)<br/>
loss = outputs.loss<br/>
total_loss += loss.item()<br/> 
optimizer.zero_grad()<br/>
loss.backward()<br/>
optimizer.step()<br/>
avg_train_loss = total_loss / len(train_loader)<br/>
print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')<br/>
model.eval()<br/>
val_loss = 0<br/>
with torch.no_grad():<br/>
for batch in val_loader:<br/>
input_ids = batch['input_ids'].to(device)<br/>
attention_mask = batch['attention_mask'].to(device)<br/>
labels = batch['labels'].to(device)<br/>
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)<br/>
loss = outputs.loss<br/>
val_loss += loss.item()<br/>
avg_val_loss = val_loss / len(val_loader)<br/>
print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')<br/>
## Generate summaries for custom inputs<br/>
def generate_summary(input_text):<br/>
inputs = tokenizer(input_text, max_length=512, truncation=True, padding='max_length',return_tensors='pt')<br/>
input_ids = inputs['input_ids'].to(device)<br/>
attention_mask = inputs['attention_mask'].to(device)<br/>
summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=60, num_beams=4, length_penalty=2.0, early_stopping=True)<br/>
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)<br/>
return summary_text<br/>
## Example usage<br/>
custom_input = "Aluko nets winner with ten minutes remaining at KC Stadium. Tomas Marek put visitors into shock lead after two minutes. Ahmed Elmohamady equalised for the hosts. Steve Bruce's side now await Europa League play-off."<br/>
custom_summary = generate_summary(custom_input)<br/>
print(custom_summary)<br/>
## Evaluate ROUGE score on the validation set<br/>
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)<br/>
rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}<br/>
model.eval()<br/> 
with torch.no_grad():<br/>
for example in validation_dataset:<br/>
input_text = example['dialogue']<br/>
reference_summary = example['summary']<br/>
generated_summary = generate_summary(input_text)<br/>
scores = scorer.score(reference_summary, generated_summary)<br/>
for key in rouge_scores.keys():<br/>
rouge_scores[key].append(scores[key].fmeasure)<br/>
## Calculate average ROUGE scores<br/>
avg_rouge_scores = {key: np.mean(rouge_scores[key]) for key in rouge_scores.keys()}<br/>
print("Average ROUGE scores on the validation set:")<br/>
print(f"ROUGE-1: {avg_rouge_scores['rouge1']:.4f}")<br/>
print(f"ROUGE-2: {avg_rouge_scores['rouge2']:.4f}")<br/>
print(f"ROUGE-L: {avg_rouge_scores['rougeL']:.4f}")<br/>

# Rouge scores:<br/>
Average ROUGE scores on the validation set:<br/>
ROUGE-1: 0.4024<br/>
ROUGE-2: 0.1740<br/>
ROUGE-L: 0.3148<br/>
