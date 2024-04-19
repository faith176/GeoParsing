import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from keras.preprocessing.sequence import pad_sequences
from keras import Input
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, GRU
from keras.callbacks import ModelCheckpoint
import tensorflow as tf



class BI_LSTM_Manager:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.model_class = BI_LSTM(gaz, preprocess)
        
    def predict_corpus(self, test_size=0.2):        
        predicted_labels, true_labels = self.model_class.train_test(test_size)
        self.score_predictions(true_labels, predicted_labels)
        self.create_classification_report(predicted_labels, true_labels)
        
        
    def new_prediction(self, text):
        if self.model_class.model is not None:
            iob = self.convert_to_iob(self.model_class.predict_new(text))
            # print(f"Unique values in prediction: {np.unique(iob)}")
            # print(iob[0][:len(text.split(" "))])
            # for word, pred in zip(text.split(" "), iob[0]):
            #     print(f"{word} : {pred}")
            return self.get_location_mentions(text, iob)
        else:
            print("Model has not been trained yet")
      
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
   
   # Scoring Functions ---------------------------------------
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
        # Flatten predictions and target arrays
        predictions_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])

        non_padding_indices = [index for index, label in enumerate(target_flat) if not np.array_equal(label, [0, 0, 0, 1])]
        non_padding_predictions = predictions_flat[non_padding_indices]
        non_padding_labels = target_flat[non_padding_indices]
        
        # Convert predictions to class labels
        predicted_labels = non_padding_predictions.argmax(axis=-1)
        target_labels = non_padding_labels.argmax(axis=-1)
                
        # Calculate precision, recall, and F1 score for non padding tags
        precision, recall, f1, _ = precision_recall_fscore_support(target_labels, predicted_labels, average='weighted')
        accuracy = accuracy_score(target_labels, predicted_labels)
        return precision, recall, f1, accuracy

    def evaluate_predictions_non_o(self, predictions, target):
        # Flatten predictions and target arrays
        predictions_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        
        non_padding_indices = [index for index, label in enumerate(target_flat) if not np.array_equal(label,  [0, 0, 0, 1])]
        non_padding_predictions = predictions_flat[non_padding_indices]
        non_padding_labels = target_flat[non_padding_indices]
        
        # Convert predictions to class labels
        predicted_labels = non_padding_predictions.argmax(axis=-1)
        target_labels = non_padding_labels.argmax(axis=-1)
        
        # Filter out "O" tags
        non_o_indices = target_labels != 2
        non_o_predicted_labels = predicted_labels[non_o_indices]
        non_o_target_labels = target_labels[non_o_indices]
        
        # Calculate precision, recall, and F1 score for non-"O" and non padding tags
        precision, recall, f1, _ = precision_recall_fscore_support(non_o_target_labels, non_o_predicted_labels, average='weighted')
        accuracy = accuracy_score(non_o_target_labels, non_o_predicted_labels)
        return precision, recall, f1, accuracy
    
    def create_classification_report(self,y_pred, y_test):
        y_true_flat = np.argmax(y_test, axis=-1).flatten()
        y_pred_flat = np.argmax(y_pred, axis=-1).flatten()

        # Ignore padding values in true labels
        mask = y_true_flat != 3
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]

        report = classification_report(y_true_flat, y_pred_flat, labels=[0, 1, 2], target_names=['B-LOC', 'I-LOC', 'O'])
        print("-"*50)
        print("Classification Report:")
        print(report)



       
       
class BI_LSTM:
    def __init__(self, gaz, preprocess, embedding_size = 50) -> None:
        self.model = None
        self.gaz = gaz
        self.preprocess = preprocess
        self.embedding_size = embedding_size
        self.glove_embeddings_index = self.load_glove_embeddings(f"data/glove.6B/glove.6B.{self.embedding_size}d.txt")    
        input_shape = (self.preprocess.MAX_SENTENCE, 219)
        output_shape = 4
        
        self.model = Sequential([
        Input(shape=input_shape, name='word_input'),
        Bidirectional(LSTM(128, return_sequences=True)), 
        GRU(units=128, return_sequences=True),
        GRU(units=128, return_sequences=True),
        Dense(output_shape, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss=lambda y_true, y_pred: self.weighted_loss(y_true, y_pred, class_weights=[1.5, 1.5, 0.5, 0.01]), metrics=['accuracy'])
        
    def load_weights(self):
        filepath = f"data/saved_data/model_checkpoints/BI-LSTM-GRU_model_weights.weights.h5"
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
            print("Weights loaded successfully.")
            return True
        else:
            print(f"No weights found")
            return False
    
    def train_test(self, test_size=0.1):
        labels, features = self.preprocess_data(self.preprocess.corpus)
        x_train, x_test, y_train, y_test = self.split_train_test(labels, features, test_size)
        self.train_model(x_train, y_train)
        y_pred = self.predict(x_test)
        return y_pred, y_test
    
    def split_train_test(self, labels, features, test_size):    
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test
        
    def weighted_loss(self, y_true, y_pred, class_weights=None):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        if class_weights is not None:
            class_weights = tf.constant(class_weights, dtype=tf.float32)
            weighted_cross_entropy = cross_entropy * class_weights
        else:
            weighted_cross_entropy = cross_entropy
        return tf.reduce_mean(weighted_cross_entropy)
        
    def train_model(self, x_train, y_train):
        if self.load_weights():
            pass
        else:
            print(f"Training from scratch.")
            save_filepath = f"data/saved_data/model_checkpoints/BI-LSTM-GRU_{{epoch:02d}}_model_weights.weights.h5"
            checkpoint_callback = ModelCheckpoint(filepath=save_filepath, save_weights_only=True, verbose=1)
            callbacks = [checkpoint_callback]
            history = self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=callbacks)
            print(f"Weights saved")
            return history

    def predict(self, x_test):
        return self.model.predict(x_test)
    
    def predict_new(self, text):
        book = self.preprocess.extract_corpus_from_text([text])[0]
        X = np.array([self.get_relevant_features(book["features"])])
        predictions = self.model.predict(X) 
        return predictions
                
    # HELPER FUNCTIONS---------------------------------------
    def preprocess_data(self, corpus):
        X = []
        y = []
        for book in corpus:
            y.append(book["labels_numerical"])
            X.append(self.get_relevant_features(book["features"]))
            
        max_len = self.preprocess.MAX_SENTENCE
        y = pad_sequences(y, maxlen=max_len, padding='post', truncating='post', value=3)
        y = np.array(np.eye(4)[y])
        X = np.array(X)
        return y, X
        
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
            word_embedding = self.glove_embeddings_index.get(feat["word"], np.zeros(50))
            prev_word_embedding = self.glove_embeddings_index.get(feat["-1:word"], np.zeros(50))
            next_word_embedding = self.glove_embeddings_index.get(feat["+1:word"], np.zeros(50))
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
        for i in range(self.preprocess.MAX_SENTENCE - len(features)):
            input.append(np.zeros(219))
        return input