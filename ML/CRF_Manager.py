import os
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from joblib import dump, load

class CRF_Manager:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.model_class = CRF(gaz, preprocess)
        
    def predict_corpus(self, test_size=0.1):        
        predicted_labels, true_labels = self.model_class.train_test(self.preprocess.corpus,test_size)
        self.score_predictions(true_labels, predicted_labels)
        self.create_classification_report(predicted_labels, true_labels)
        
    def new_prediction(self, text):
        if self.model_class.model is not None:
            y_pred = self.model_class.predict_new(text)
            # print(f"Unique values in prediction: {np.unique(y_pred)}")
            # print(y_pred[0][:len(text.split(" "))])
            # for word, pred in zip(text.split(" "), y_pred[0]):
            #     print(f"{word} : {pred}")
            return self.get_location_mentions(text, y_pred)
        else:
            print("Model has not been trained yet")
            
    def get_location_mentions(self, text, iob):
        #Flatten Prediction
        iob = [item for sublist in iob for item in sublist]
        current_location = ''
        location_mentions = []

        for word, bio_tag in zip(text.split(" "), iob):
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
        
    # Scoring Functions-------------------------------------------
    def score_predictions(self, true_labels, predicted_labels):
        precision, recall, f1_score, accuracy = self.evaluate_predictions_o(predicted_labels, true_labels)
        print("-"*50)
        print("Base Scores")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
        
        print("-"*50)
        
        print("Relevant Scores: Labels of Interest: B-LOC and I-LOC")
        precision, recall, f1_score, accuracy = self.evaluate_predictions_non_o(predicted_labels, true_labels)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
    
    def evaluate_predictions_o(self, predictions, target):      
        target_flat = [item for sublist in target for item in sublist]
        predictions_flat = [item for sublist in predictions for item in sublist]
    
        precision, recall, f1, _ = precision_recall_fscore_support(target_flat, predictions_flat, average='weighted')
        accuracy = accuracy_score(target_flat, predictions_flat)
        return precision, recall, f1, accuracy

    def evaluate_predictions_non_o(self, predictions, target):
        target_flat = [item for sublist in target for item in sublist]
        predictions_flat = [item for sublist in predictions for item in sublist]

        # Filter out "O" tags
        non_o_indices = [i for i, label in enumerate(target_flat) if label != 'O']
        non_o_target_labels = [target_flat[i] for i in non_o_indices]
        non_o_predicted_labels = [predictions_flat[i] for i in non_o_indices]
        
        precision, recall, f1, _ = precision_recall_fscore_support(non_o_target_labels, non_o_predicted_labels, average='weighted')
        accuracy = accuracy_score(non_o_target_labels, non_o_predicted_labels)
        return precision, recall, f1, accuracy
        
    def create_classification_report(self, predicted_labels, true_labels):
        y_true_flat = [item for sublist in true_labels for item in sublist]
        y_pred_flat = [item for sublist in predicted_labels for item in sublist]

        report = classification_report(y_true_flat, y_pred_flat)
        print("-"*50)
        print("Classification Report:")
        print(report)
       

class CRF:
    def __init__(self, gaz, preprocess) -> None:
        self.preprocess = preprocess
        self.gaz = gaz
        self.model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=200,
        all_possible_transitions=True
        )
        
    def load_weights(self):
        filepath = f"data/saved_data/model_checkpoints/CRF_model_weights.joblib"
        if os.path.exists(filepath):
            loaded_model = load(filepath)
            self.model = loaded_model
            print("Weights loaded successfully.")
            return True
        else:
            print(f"No weights found")
            return False
    
    def train_test(self, corpus, test_size=0.1):
        labels, features = self.preprocess_data(corpus)
        X_train, X_test, y_train, y_test = self.split_train_test(labels, features, test_size)
        self.train_model(X_train, y_train)
        y_pred = self.predict(X_test)
        return y_pred, y_test
    
    def preprocess_data(self, corpus):
        labels = [book['labels'] for book in corpus]
        features = [book['features'] for book in corpus]
        return labels, features
    
    def split_train_test(self, labels, features, test_size):    
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        filepath = f"data/saved_data/model_checkpoints/CRF_model_weights.joblib"
        if self.load_weights():
            pass
        else:
            print(f"Training from scratch and saving.")
            self.model.fit(X_train, y_train)
            dump(self.model, filepath)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_new(self, text):
        features = self.preprocess.extract_corpus_from_text([text])[0]["features"]
        return self.predict([features])
    

def main():
    pass
    
if __name__ == "__main__":
    main()