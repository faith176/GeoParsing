from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.metrics import accuracy_score
import warnings
import os
import math

import torch
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertModel, BertTokenizer, BertForTokenClassification, RobertaTokenizer, RobertaForTokenClassification
from transformers import AdamW
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from TorchCRF import CRF

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BERT_Manager:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.predictions_dict = {
            0: 'B-LOC',
            1:'I_LOC',
            2: 'O',
            3: 'PAD'
        }
        self.model_class = BERT(gaz, preprocess)
    
    def predict_corpus(self, train_size=0.9):        
        predicted_labels, true_labels = self.model_class.train_test(train_size)
        self.score_predictions(true_labels, predicted_labels)
        self.create_classification_report(predicted_labels, true_labels)

    def new_prediction(self, text):
        if self.model_class.model is not None:
            iob, tokens = self.model_class.predict_new(text)
            iob = self.convert_to_iob(iob)
            # print(f"Unique values in prediction: {np.unique(iob)}")
            # for word, pred in zip(tokens, iob):
            #     print(f"{word} : {pred}")
            return self.get_location_mentions(tokens, iob)
        else:
            print("Model has not been trained yet")
    
    def convert_to_iob(self, predictions):
        return [self.predictions_dict[pred] for pred in predictions]
    
    def get_location_mentions(self, text, iob):
        current_location = ''
        location_mentions = []

        for word, bio_tag in zip(text, iob):
            if bio_tag == 'B-LOC':
                current_location = word.lower()
            elif bio_tag == 'I-LOC':
                current_location += ' ' + word.lower()
            elif bio_tag == 'O' and current_location:
                location_mentions.append(current_location)
                current_location = ''
        # Check for the last location mention
        if current_location:
            location_mentions.append(current_location)
        return location_mentions 
        
    # Scoring Functions ---------------------------------------
    def score_predictions(self, true_labels, predicted_labels):
        print('Base Scores')
        precision, recall, f1_score, accuracy = self.model_class.evaluate_predictions_non_o(predicted_labels, true_labels, [3])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
        print()

        print("Relevant Scores: Labels of Interest: B-LOC and I-LOC")
        precision, recall, f1_score, accuracy = self.model_class.evaluate_predictions_non_o(predicted_labels, true_labels, [2,3])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
        
    def create_classification_report(self,y_pred, y_test):
        report = classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=['B-LOC', 'I-LOC', 'O'])
        print("-"*50)
        print("Classification Report:")
        print(report)


class CustomRobertaForTokenClassification(nn.Module):
    def __init__(self, num_labels):
        super(CustomRobertaForTokenClassification, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        init.xavier_uniform_(self.classifier.weight)
        # Initialize bias to zeros
        if self.classifier.bias is not None:
            init.constant_(self.classifier.bias, 0.0)
        self.crf = CRF(num_labels, batch_first = True)
        
        # for getting loss and logits after forward pass
        self.loss = None
        self.logits = None

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            # loss = -self.crf(emissions, labels, mask=attention_mask.type(torch.uint8))
            # loss = -self.crf(F.log_softmax(emissions, 2), labels, mask=attention_mask.type(torch.uint8), reduction='mean')
            loss = F.cross_entropy(emissions.view(-1, emissions.size(-1)), 
                                       labels.view(-1), 
                                       weight=torch.tensor([1.7, 1.8, 0.5, 0.1]),
                                       reduction='mean')
            # predictions = self.crf.decode(emissions, mask=attention_mask.type(torch.uint8))
            predictions = F.softmax(emissions, dim=-1).argmax(dim=-1)
            self.loss = loss
            self.logits = predictions
            return self
        else:
            # predictions = self.crf.decode(emissions, mask=attention_mask.type(torch.uint8))
            predictions = F.softmax(emissions, dim=-1).argmax(dim=-1)
            self.logits = predictions
            return self
        

class BERT:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.model = CustomRobertaForTokenClassification(num_labels=4)
        self.batch_size = 32
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.num_epochs= 1
        learning_rate= 3e-5
        self.max_grad_norm= 1.0
        eps =  1e-8
        
        # optimizer = AdamW(model.parameters(), lr=learning_rate, eps=eps)
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate,eps=eps )
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, eps=eps)
        
    def load_weights(self):
        filepath = f"data/saved_data/model_checkpoints/Custom_Roberta.pt"
        if os.path.exists(filepath):
            self.model.load_state_dict(torch.load(filepath))
            print("Weights loaded successfully.")
            return True
        else:
            print(f"No weights found")
            return False
        
    def predict_new(self, text):
        dataloader, tokens, subword_indicator = self.tokenize_and_format_new_data(text)
        self.model.eval() # Evaluation Mode
        y_pred = []
        for batch in tqdm(dataloader, desc="Prediction"):
            inputs, attention_masks = batch
            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=attention_masks)
                loss, logits = outputs.loss, outputs.logits
                predicted = logits
            predicted = [[pred.item() for pred, m in zip(seq, mask) if m] for seq, mask in zip(predicted, attention_masks)]
            flat_pred = [item for sublist in predicted for item in sublist]
            y_pred = flat_pred
            
        formatted_pred = []
        formatted_token = []
        y_pred = y_pred[1:-1]
        for pred, token, subword in zip(y_pred, tokens, subword_indicator):
            if subword == True:
                formatted_token[-1] = formatted_token[-1] + token
            else:
                formatted_token.append(token)
                formatted_pred.append(pred)
        return formatted_pred, formatted_token
        
    def train_test(self, train_split=0.9):
        train_dataloader, test_dataloader = self.create_dataloaders(self.batch_size, train_split=train_split)
        y_test, y_pred = self.setup(train_dataloader, test_dataloader)
        return y_pred, y_test
    
    def setup(self, train_dataloader, test_dataloader):
        if self.load_weights():
            y_test, y_pred = self.test(test_dataloader)
        else:
            print(f"Training from scratch.")
            for epoch in range(0, self.num_epochs):
                print(f'Epoch {epoch + 1}/{self.num_epochs} Summary')
                self.train(train_dataloader)
                y_test, y_pred = self.test(test_dataloader)
                print("="*50)
                torch.save(self.model.state_dict(), f"data/saved_data/model_checkpoints/Custom_Roberta_epoch_{epoch}.pt")
            print(f"Weights saved")
        return y_test, y_pred
                    
    def train(self, train_dataloader):
        tr_loss = 0
        num_batches = 0
        self.model.train() # Set to training mode
        # ========================================
        #               Training
        # ========================================
        for batch in tqdm(train_dataloader, desc="Training"):
            inputs, attention_masks, labels = batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, labels=labels, attention_mask=attention_masks)
            loss, logits = outputs.loss, outputs.logits
            tr_loss += loss.item()
            num_batches += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # Prevent exploding gradient
            self.optimizer.step()
        avg_train_loss = tr_loss / num_batches
        print("Train loss: {}".format(avg_train_loss))
    
    def test(self,test_dataloader):
        # ========================================
        #               Validation
        # ========================================
        total_eval_accuracy_relevant, total_eval_accuracy = 0, 0
        total_eval_loss = 0
        y_test = []
        y_pred = []
        num_batches = 0
        self.model.eval() # Evaluation Mode
        for batch in tqdm(test_dataloader, desc="Validation"):
            inputs, attention_masks, labels = batch
            with torch.no_grad():
                outputs = self.model(inputs, labels=labels, attention_mask=attention_masks)
                loss, logits = outputs.loss, outputs.logits
                predicted = logits
            total_eval_loss += loss.item()
            labels = [[label.item() for label, m in zip(seq, mask) if m] for seq, mask in zip(labels, attention_masks)]
            predicted = [[pred.item() for pred, m in zip(seq, mask) if m] for seq, mask in zip(predicted, attention_masks)]
            # Flatten labels and predictions
            flat_labels = [item for sublist in labels for item in sublist]
            flat_pred = [item for sublist in predicted for item in sublist]
            # accuracy = accuracy_score(flat_labels, flat_pred)
            _, _, _, accuracy = self.evaluate_predictions_non_o(flat_pred, flat_labels, [2,3])
            total_eval_accuracy_relevant += accuracy
            _, _, _, accuracy = self.evaluate_predictions_non_o(flat_pred, flat_labels, [3])
            total_eval_accuracy += accuracy
            
            num_batches += 1
            y_test.extend(flat_labels)
            y_pred.extend(flat_pred)
    
        avg_val_loss = total_eval_loss / num_batches
        avg_val_accuracy_relevant = total_eval_accuracy_relevant / num_batches
        avg_val_accuracy = total_eval_accuracy / num_batches
        
        print("Validation Loss: {0:.2f}".format(avg_val_loss))
        print("Accuracy: {}".format(avg_val_accuracy))
        print("Relevant Accuracy: {}".format(avg_val_accuracy_relevant))
        print("-"*50)
        return y_test, y_pred
        
    # Scoring Functions---------------------------------------------------
    def evaluate_predictions_non_o(self, predictions, target, excluded_values=[3], debug=False):    
        warnings.filterwarnings('ignore', category=UserWarning)
        filtered_predictions = [pred for pred, label in zip(predictions, target) if label not in excluded_values]
        filtered_labels = [label for label in target if label not in excluded_values]
        if debug:
            print(filtered_labels)
            print(filtered_predictions)
        # Calculate precision, recall, and F1 score for non-"O" tags
        precision, recall, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_predictions, average='weighted')
        accuracy = accuracy_score(filtered_labels, filtered_predictions)
        warnings.filterwarnings('default', category=UserWarning)
        return precision, recall, f1, accuracy  
        
    # Preprocessing Functions---------------------------------------------
    def create_dataloaders(self, batch_size, train_split):
        dataset = self.tokenize_and_format_test_train_data()
        # Split dataset
        train_size = math.floor(train_split * len(dataset))
        test_size = len(dataset) - train_size
        # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataset = dataset[:train_size],
        test_dataset = dataset[train_size:]
        # Prepare Data Loaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        return train_dataloader, test_dataloader    
    
    def tokenize_and_format_new_data(self, text, batch_size=32):
        tokens = []
        tokenized_data = []
        max_len = 512
        preprocess_input = []
        subword_indicator = []
        for input in text.split(" "):
            tokenized_word = self.tokenizer.tokenize(input)
            tokens.extend(tokenized_word)
            tokenized_ids = self.tokenizer.convert_tokens_to_ids(tokenized_word)
            preprocess_input.extend(tokenized_ids)
            subword_indicator.extend([False]) #First word represents the start of a word
            subword_indicator.extend([True] * (len(tokenized_word)-1))
        chunked_text = [preprocess_input[i:i+max_len-2] for i in range(0, len(preprocess_input), max_len-2)]
        for input_ids in chunked_text:
            input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
            input_ids = input_ids + [self.tokenizer.pad_token_id]*(max_len-len(input_ids))
            attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
            attention_mask[0] = 1

            tokenized_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            })
        input_ids_tensor = torch.tensor([item['input_ids'] for item in tokenized_data])
        attention_mask_tensor = torch.tensor([item['attention_mask'] for item in tokenized_data])
            
        # Convert tokenized data to PyTorch dataset
        dataset = TensorDataset(input_ids_tensor, attention_mask_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        return dataloader, tokens, subword_indicator
        
    def tokenize_and_format_test_train_data(self):
        tokenized_data = []
        max_len = 512
        for book in self.preprocess.corpus:
            text = book["text"].split(" ")
            labels = book["labels_numerical"]
            
            preprocess_input = []
            preprocess_labels = []
            
            for input, label in zip(text, labels):
                tokenized_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input))
                preprocess_input.extend(tokenized_ids)
                preprocess_labels.extend([label] * len(tokenized_ids))
            
            chunked_text = [preprocess_input[i:i+max_len-2] for i in range(0, len(preprocess_input), max_len-2)]
            chunked_labels = [preprocess_labels[i:i+max_len-2] for i in range(0, len(preprocess_labels), max_len-2)]
            
            for input_ids, label_ids in zip(chunked_text, chunked_labels):
                # Deal with Inputs
                input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                input_ids = input_ids + [self.tokenizer.pad_token_id]*(max_len-len(input_ids))
                
                # Deal with Labels for current chunk
                label_ids = [3] + label_ids + [3]
                label_ids = label_ids + [3]*(max_len-len(label_ids))
                
                attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
                attention_mask[0] = 1

                tokenized_data.append({
                    'input_ids': input_ids,
                    'labels': label_ids,
                    'attention_mask': attention_mask,
                })
        
        input_ids_tensor = torch.tensor([item['input_ids'] for item in tokenized_data])
        label_ids_tensor = torch.tensor([item['labels'] for item in tokenized_data])
        attention_mask_tensor = torch.tensor([item['attention_mask'] for item in tokenized_data])
        
        dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, label_ids_tensor)
        return dataset
