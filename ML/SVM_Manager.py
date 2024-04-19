import os
import numpy as np

from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from joblib import dump, load
from sklearn.model_selection import GridSearchCV


class SVM_Manager:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.model_class = SVM(preprocess)
        
    def predict_corpus(self, test_size=0.1):        
        predicted_labels, true_labels = self.model_class.train_test(test_size)
        self.score_predictions(self.convert_to_iob([true_labels]), self.convert_to_iob([predicted_labels]))
        self.create_classification_report(predicted_labels, true_labels)
        
    def new_prediction(self, text):
        if self.model_class.model is not None:
            y_pred = self.model_class.predict_new(text)
            y_pred = self.convert_to_iob([y_pred])
            # print(f"Unique values in prediction: {np.unique(y_pred)}")
            # print(y_pred[0][:len(text.split(" "))])
            # for word, pred in zip(text.split(" "), y_pred[0]):
                # print(f"{word} : {pred}")
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
    
    def convert_to_iob(self, predictions):
        iob_tags = []
        for sequence in predictions:
            sequence_iob = []
            for prediction in sequence:
                if prediction.argmax() == 0:  # B-LOC
                    tag = 'B-LOC'
                elif prediction.argmax() == 1:  # I-LOC
                    tag = 'I-LOC'
                else:  # O
                    tag = 'O'
                sequence_iob.append(tag)
            iob_tags.append(sequence_iob)
        return iob_tags
        
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
        report = classification_report(true_labels, predicted_labels, labels=[0, 1, 2], target_names=['B-LOC', 'I-LOC', 'O'])
        print("-"*50)
        print("Classification Report:")
        print(report)
       

class SVM:
    def __init__(self, preprocess, embedding_size=50) -> None:
        self.preprocess = preprocess
        self.embedding_size = embedding_size
        self.glove_embeddings_index = self.load_glove_embeddings(f"data/glove.6B/glove.6B.{self.embedding_size}d.txt")    
        self.model = MultiOutputClassifier(SVC(kernel='rbf', C=10.0, gamma=0.01, degree=2), n_jobs=-1)
        
    def load_weights(self):
        filepath = f"data/saved_data/model_checkpoints/SVM_model_weights.joblib"
        if os.path.exists(filepath):
            loaded_model = load(filepath)
            self.model = loaded_model
            print("Weights loaded successfully.")
            return True
        else:
            print(f"No weights found")
            return False
    
    def train_test(self, test_size=0.1):
        labels, features = self.preprocess_data()
        X_train, X_test, y_train, y_test = self.split_train_test(labels, features, test_size)
        self.train_model(X_train, y_train)
        y_pred = self.predict(X_test)
        return y_pred, y_test
    
    def preprocess_data(self):
        X = []
        y = []
        for book in self.preprocess.corpus:
            y = y + book["labels_numerical"]
            X = X + self.get_relevant_features(book["features"])
        y = np.array([self.one_hot_encode(item, 3) for item in y])
        X = np.array(X)
        return y, X
    
    def split_train_test(self, labels, features, test_size):    
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    
    def train_model(self, x_train, y_train):
        filepath = f"data/saved_data/model_checkpoints/SVM_model_weights.joblib"
        if self.load_weights():
            pass
        else:
            print(f"Training from scratch and saving.")
            self.model.fit(x_train, y_train)
            dump(self.model, filepath)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_new(self, text):
        book = self.preprocess.extract_corpus_from_text([text])[0]
        X = np.array([self.get_relevant_features(book["features"])])
        X = X.reshape(-1, 219)
        predictions = self.model.predict(X) 
        return predictions
    
    
     # Get Best Weights
    def get_best_params(self, x_train, y_train, cv=5):
        param_grid = {
        'estimator__kernel': ['linear', 'rbf', 'poly'],
        'estimator__C': [0.1, 1, 10],
        'estimator__gamma': [0.001, 0.01, 0.1],
        'estimator__degree': [2, 3, 4]
        }
        svm_model = MultiOutputClassifier(SVC())
        grid_search = GridSearchCV(svm_model, param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(x_train, y_train)
        
        best_params = grid_search.best_params_
        print("Best hyperparameters:", best_params)

        best_score = grid_search.best_score_
        print("Best cross-validation score:", best_score)
     
    
     # Helper Functions------------------------------------    
    def load_glove_embeddings(self, embedding_file):
        embeddings_index = {}
        with open(embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
    
    def one_hot_encode(self, value, num_classes):
        encoding = np.zeros(num_classes)
        encoding[value] = 1
        return encoding
    
    def get_relevant_features(self, features):
        input = []
        for feat in features:
            #Embeddings 
            word_embedding = self.glove_embeddings_index.get(feat["word"], np.zeros(self.embedding_size))
            prev_word_embedding = self.glove_embeddings_index.get(feat["-1:word"], np.zeros(self.embedding_size))
            next_word_embedding = self.glove_embeddings_index.get(feat["+1:word"], np.zeros(self.embedding_size))
            # Word Features
            inGaz = [feat["IsLOCGaz"]]
            POS_tag_OH = (self.one_hot_encode(self.preprocess.POS_tags[feat["pos_tag"]], len(self.preprocess.POS_tags))).flatten()
            Is_Preposition = [feat["Is_Preposition"]]
            has_internal_punctuations = [feat["has_internal_punctuations"]]
            capitalized = [str(feat["word_shape"]).istitle()]
            # Nearby Word's Features
            prev_POS_tag_OH = (self.one_hot_encode(self.preprocess.POS_tags[feat["-1:pos_tag"]], len(self.preprocess.POS_tags))).flatten()
            prev_capitalized = [str(feat["-1:word_shape"]).istitle()]
            next_POS_tag_OH = (self.one_hot_encode(self.preprocess.POS_tags[feat["+1:pos_tag"]], len(self.preprocess.POS_tags))).flatten()
            prev_capitalized = [str(feat["+1:word_shape"]).istitle()]
            # Concat features together
            input.append(np.concatenate([word_embedding, POS_tag_OH, inGaz, Is_Preposition, has_internal_punctuations, capitalized,prev_word_embedding, prev_POS_tag_OH, prev_capitalized, next_word_embedding, next_POS_tag_OH, prev_capitalized]))
        return input
    

def main():
    pass
    
if __name__ == "__main__":
    main()