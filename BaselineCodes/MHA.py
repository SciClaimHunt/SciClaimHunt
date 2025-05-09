import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm
import torch.nn.functional as F
#import numpy as nps
import torch
import nltk
import csv
import os
#nltk.download('punkt_tab')
def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)


def read_dataset(df_location:str):
    df = pd.read_csv(df_location)
    #CHECK FOR TESTING
    #df=df.head(200)
    # Create the 'label' column based on 'Type' (0 for 'positive', 1 for 'negative')
    df=df.fillna(value='')
    if df.Introduction.iloc[0]!=None:
        df['retrievedsentences']=df['Introduction'].astype(str)+' '+df['Related Work'].astype(str)+' '+df['Proposed Method'].astype(str)+' '+df['Results and Analysis']
    df['label'] = df['Type'].apply(lambda x: 0 if x == 'positive' else 1)
    # Split the dataset into train (80%), test (10%), and dev (10%)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, dev_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    ##MULTI GPU training

    # Custom Dataset class
    class CustomTextDataset(Dataset):
        def __init__(self, df,tokenizer, max_num_sent=140,max_len=60):
            self.df = df
            self.claims = df['Claim'].tolist()
            self.sentences = df['retrievedsentences'].tolist()
            self.labels = df['label'].tolist()
            self.tokenizer = tokenizer
            self.max_num_sent = max_num_sent
            self.max_len = max_len

        def __len__(self):
            return len(self.claims)

        def __getitem__(self, index):
            claim = self.claims[index]
            retrieved_sentences= sent_tokenize(self.sentences[index])  # Split sentences
            label = self.labels[index]
            
            if len(retrieved_sentences) > self.max_num_sent:
                retrieved_sentences = retrieved_sentences[:self.max_num_sent]
            else:
                retrieved_sentences += [""] * (self.max_num_sent - len(retrieved_sentences))  # Padding

            # Encode claim and retrieved sentences using Sentence-BERT
            claim_embedding = self.tokenizer.encode(claim, convert_to_tensor=True)
            sentence_embeddings = self.tokenizer.encode(retrieved_sentences, convert_to_tensor=True) 

            return {
                'claim_embedding': claim_embedding,
                'sentence_embeddings': sentence_embeddings,
                'label': torch.tensor(label, dtype=torch.long),
            }
    # train_claims = train_df['Claim'].tolist()
    # train_retrieved_sentences = train_df['retrievedsentences'].tolist()
    # train_labels = train_df['label'].tolist()
    sentence_bert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # dev_claims = dev_df['Claim'].tolist()
    # dev_retrieved_sentences = dev_df['retrievedsentences'].tolist()
    # dev_labels = dev_df['label'].tolist()

    test_claims = test_df['Claim'].tolist()
    test_retrieved_sentences = test_df['retrievedsentences'].tolist()
    test_labels = test_df['label'].tolist()
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    train_dataset = CustomTextDataset(train_df, sentence_bert_model)
    dev_dataset = CustomTextDataset(dev_df, sentence_bert_model)
    test_dataset = CustomTextDataset(test_df, sentence_bert_model)
    return train_dataset,dev_dataset,test_dataset,test_claims,test_retrieved_sentences,test_labels
# Define a model that uses Roberta for encoding and a custom head for classification
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        # Load the pre-trained Roberta model for encoding
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        
        # Define the classifier with 2 fully connected layers
        
    def forward(self, claim_embedding, sentence_embeddings):
        # Get the [CLS] token output from both the claim and the retrieved sentence
        # Ensure claim_embedding has a sequence length of 1 for multi-head attention
        claim_embedding = claim_embedding.unsqueeze(1)  # (batch_size, 1, embed_dim)
        attention_output, attention_weights = self.attention(claim_embedding, sentence_embeddings, sentence_embeddings)
        return attention_output.squeeze(1), attention_weights  # (batch_size, embed_dim)
class SentenceBERTWithAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, hidden_size=1024, num_classes=2):
        super(SentenceBERTWithAttention, self).__init__()
        
        # Multi-Head Attention layer
        self.multi_head_attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_size),  # Element-wise difference + multiplication + concatenated embeddings
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, claim_embedding, sentence_embeddings):

        # print(" the shape of claim_embedding", claim_embedding.shape)
        # print(" the shape of sentence_embeddings", sentence_embeddings.shape)
        attention_output, _ = self.multi_head_attention(claim_embedding, sentence_embeddings)

        # print(" the shape of attention_output", attention_output.shape)
        
        # Element-wise difference and multiplication
        diff = torch.abs(claim_embedding - attention_output)
        mult = claim_embedding * attention_output
        
        # Concatenate claim_embedding, attention_output, element-wise diff, and mult
        combined_representation = torch.cat([claim_embedding, attention_output, diff, mult], dim=-1)
        #print(" the shape of feature", combined_representation.shape)
        
        # Pass through the fully connected classifier
        logits = self.classifier(combined_representation)
        
        return logits   

def load_train_objs(train_dataset,dev_dataset,test_dataset):

    # Load the tokenizer
    train_dataset=train_dataset
    dev_dataset=dev_dataset
    test_dataset=test_dataset
    # Prepare training, validation, and test data
    embedding_dim = 384  # MiniLM model dimension
    model = SentenceBERTWithAttention(embedding_dim=embedding_dim, num_heads=8)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    return train_dataset,dev_dataset,test_dataset,model,optimizer
    
    
def collate_fn(batch):
    #print(batch.keys())
    claim_embeddings = [item['claim_embedding'] for item in batch]
    #print(claim_embeddings)
   # print('-----------------------------------------------------------------------------------------------')
    sentence_embeddings = [item['sentence_embeddings'] for item in batch]
    labels = [item['label'] for item in batch]
   # print('====================================')

    # Pad sentence embeddings to ensure all are the same size
    sentence_embeddings_padded = torch.nn.utils.rnn.pad_sequence(sentence_embeddings, batch_first=True)

    # Stack claim embeddings and labels
    claim_embeddings = torch.stack(claim_embeddings)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return {
        'claim_embedding': claim_embeddings,
        'sentence_embeddings': sentence_embeddings_padded,
        'label': labels
    }

def prepare_dataloader(train_dataset:Dataset,dev_dataset:Dataset,test_dataset:Dataset,batch_size:int):

    ## MULTI GPU SETUP keep shuffle as FALSE and add sampler
    #print('&&&&&&&&&&&******************************************************************************************************************')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False,sampler=DistributedSampler(train_dataset),collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader,dev_dataloader,test_dataloader
def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    predictions = []
    #prediction_probs = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        claim_embedding = batch['claim_embedding'].to(device)
        sentence_embeddings = batch['sentence_embeddings'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            logits = model(claim_embedding, sentence_embeddings)
        
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
     #   prediction_probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average=None)  # Class-wise F1
    report = classification_report(true_labels, predictions, output_dict=True)
    
    return accuracy, f1, report, predictions
class Trainer:
    def __init__(
            self,
            model:torch.nn.Module,
            train_data:DataLoader,
            dev_data:DataLoader,
            test_data:DataLoader,
            optimizer:torch.optim.Optimizer,
            gpu_id:int,
            lr_scheduler,
            #save_every:int,
            save_every:int,
            #func:function,
            test_claims,
            test_retrieved_sentences,
            test_labels
    )->None:
        self.gpu_id=gpu_id
        self.model=model.to(gpu_id)
        self.train_dataloader=train_data
        self.dev_dataloader=dev_data
        self.test_dataloder=test_data
        self.optimizer=optimizer
        self.save_every=save_every
        # self.model=DDP(model,device_ids=[gpu_id],find_unused_parameters=True)
        self.model=DDP(model,device_ids=[gpu_id])
        self.lr_scheduler=lr_scheduler
        self.test_claims=test_claims
        self.test_retrieved_sentences=test_retrieved_sentences
        self.test_labels=test_labels
           
    def eval_test(self,epoch):
        test_accuracy, test_f1,report, test_predictions = evaluate_model(self.model, self.test_dataloder, self.gpu_id)

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Scores: {test_f1}")

        #Save predictions and probabilities to a CSV file
        output_df = pd.DataFrame({
            'Claim': self.test_claims,
            'Retrieved_Sentences': self.test_retrieved_sentences,
            'True_Labels': self.test_labels,
            'Predicted_Labels': test_predictions,
           # 'Prediction_Probabilities': [list(probs) for probs in test_prediction_probs],
        })

        output_df.to_csv(f'Predictions/RoBERTa_evid_claim_t_predictions_for_{epoch}_with_accuracy_{test_accuracy}_and_{test_f1}.csv', index=False)

        print("Predictions on the test set saved ")        

    def train(self,max_epochs:int):
        best_val_accuracy=0.0
        best_model_path="best_model.pth"
        torch.save(self.model.module.state_dict(),best_model_path)
        log_file="training_log.txt"
        NUM_ACCUMULATION_STEPS=1
        with open(log_file,'w') as f:
            f.write("Epoch\tTrain Loss\tVal Accuracy\tClasswise F1\n")
        for epoch in range(max_epochs):
            self.train_dataloader.sampler.set_epoch(epoch)
            self.model.train()
            total_loss=0
            for idx,batch in enumerate(tqdm(self.train_dataloader,desc=f"Training Epoch {epoch+1}")):
                claim_embedding = batch['claim_embedding'].to(self.gpu_id)
                sentence_embeddings = batch['sentence_embeddings'].to(self.gpu_id)
                # retrieved_input_ids = batch['retrieved_input_ids'].to(self.gpu_id)
                # retrieved_attention_mask = batch['retrieved_attention_mask'].to(self.gpu_id)
                labels = batch['label'].to(self.gpu_id)
                
                logits = self.model(claim_embedding, sentence_embeddings)
                loss = nn.CrossEntropyLoss()(logits, labels)                
                loss=loss/NUM_ACCUMULATION_STEPS
                total_loss += loss.item()
                loss.backward()
                if((idx+1)%NUM_ACCUMULATION_STEPS==0) or(idx+1==len(self.train_dataloader)):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
            avg_train_loss = total_loss / len(self.train_dataloader)
            val_accuracy, classwise_f1, classwise_report, _ = evaluate_model(self.model, self.dev_dataloader, self.gpu_id)
            with open(log_file, 'a') as f:
                f.write(f"{epoch + 1}\t{avg_train_loss:.4f}\t{val_accuracy:.4f}\t{classwise_f1}\n")
            

            print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print(f"Classwise F1 Scores: {classwise_f1}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if self.gpu_id==0:
                    torch.save(self.model.module.state_dict(),best_model_path)
                    print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")
            #self.model.module.load_state_dict(torch.load(best_model_path))
            self.eval_test(epoch)
    
# test_accuracy, test_f1, test_report, test_predictions, test_prediction_probs = evaluate_model(model, test_dataloader, device)

# print(f"Test Accuracy: {test_accuracy:.4f}")
# print(f"Test F1 Scores: {test_f1}")

# #Save predictions and probabilities to a CSV file
# output_df = pd.DataFrame({
#     'Claim': test_claims,
#     'Retrieved_Sentences': test_retrieved_sentences,
#     'True_Labels': test_labels,
#     'Predicted_Labels': test_predictions,
#     'Prediction_Probabilities': [list(probs) for probs in test_prediction_probs]
# })

# output_df.to_csv('RoBERTa_evid_claim_t_predictions.csv', index=False)

# print("Predictions on the test set saved to 'test_predictions.csv'.")
#Track the best model
def get_lr_Scheduler(train_data:DataLoader,optimizer:torch.optim.Optimizer,num_epochs:int):
    num_training_steps = num_epochs * len(train_data)
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    return lr_scheduler

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int,df_location:str):
    ddp_setup(rank, world_size)
    num_epochs = total_epochs
    train_dataset,dev_dataset,test_dataset,test_claims,test_retrieved_sentences,test_labels=read_dataset(df_location)
    train_dataset,dev_dataset,test_dataset,model,optimizer=load_train_objs(train_dataset,dev_dataset,test_dataset)
    train_data,dev_data,test_data=prepare_dataloader(train_dataset,dev_dataset,test_dataset,batch_size)
    lr_scheduler=get_lr_Scheduler(train_data,optimizer,total_epochs)
    trainer=Trainer(model,train_data,dev_data,test_data,optimizer,rank,lr_scheduler,save_every,test_claims,test_retrieved_sentences,test_labels)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('locationofdf',type=str,default='/scratch/fourccmodels/csv_file_new',help='Place from where data is taken')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size,args.locationofdf), nprocs=world_size)
