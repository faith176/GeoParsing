from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

class Baseline_Manager:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.Baseline = Baseline(gaz, preprocess)
        
    def predict_corpus(self):
        predicted_labels, true_labels = self.Baseline.train_test()
        self.score_predictions(true_labels, predicted_labels)
        self.create_classification_report(predicted_labels,true_labels)
        
    def new_prediction(self, text):
        y_pred =  self.Baseline.find_locations(text)
        return self.get_location_mentions(text, y_pred)
    
    def get_location_mentions(self, text, iob):
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
        
    # Scoring Functions-----------------------------------------
    def score_predictions(self, true_labels, predicted_labels):
        print("-"*50)
        print("Base Scores")
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        
        print("-"*50)
        non_o_indices = [i for i, label in enumerate(true_labels) if label != 'O']
        non_o_target_labels = [true_labels[i] for i in non_o_indices]
        non_o_predicted_labels = [predicted_labels[i] for i in non_o_indices]
        print("Relevant Scores: Labels of Interest: B-LOC and I-LOC")
        accuracy = accuracy_score(non_o_target_labels, non_o_predicted_labels)
        precision = precision_score(non_o_target_labels, non_o_predicted_labels, average='weighted')
        recall = recall_score(non_o_target_labels, non_o_predicted_labels, average='weighted')
        f1 = f1_score(non_o_target_labels, non_o_predicted_labels, average='weighted')
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
    
    def create_classification_report(self,y_pred, y_test):
        report = classification_report(y_test, y_pred)
        print("-"*50)
        print("Classification Report:")
        print(report)
            

class Baseline:
    def __init__(self, gaz, preprocess) -> None:
        self.gaz = gaz
        self.preprocess = preprocess
        self.max_check_length = 9
        
    def train_test(self):
        true_labels, predicted_labels = [], []
        for book in self.preprocess.corpus:
            true_labels += book["labels"]
            predicted_labels += self.find_locations(book["text"])
        return predicted_labels, true_labels
        
    def find_locations(self, text):
        text = text.split(" ")
        iob_tags = ['O']* len(text)
        
        word_index = 0
        while word_index < len(text):
            word = text[word_index]
            max_so_far = 0
            found = False
            check = ''
            for i in range(0, min(self.max_check_length, len(text) - word_index)):
                if check == '':
                    check = word
                else:
                    check += (' ' +(''.join(char for char in text[word_index + i] if char.isalnum() or char.isspace())).lower())
                if self.gaz.check_location_exist(check) == True:
                    max_so_far = i
                    found = True
            if found:
                for i in range(max_so_far + 1):
                    if i == 0:
                        iob_tags[word_index + i] = 'B-LOC'
                    else:
                        iob_tags[word_index + i] = 'I-LOC'
                word_index += max_so_far + 1
            else:
                word_index += 1
        return iob_tags
    