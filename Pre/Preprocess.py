import pandas as pd
import spacy
import pickle
import xml.etree.ElementTree as ET
import re
import os
from alive_progress import alive_bar
import re

class Preprocess:
    def __init__(self, gaz=None) -> None:
        self.nlp = spacy.load('en_core_web_sm')
        self.corpus = []
        self.gaz = gaz
        self.stopwords = self.get_utility(file_path = "data/utility/stopwords.txt"), 
        self.prepositions = self.get_utility(file_path = "data/utility/prepositions.txt")
        self.POS_tags = {
            "ADJ": 0,
            "ADP": 1,
            "ADV": 2,
            "AUX": 3,
            "CONJ": 4,
            "DET": 5,
            "INTJ": 6,
            "NOUN": 7,
            "NUM": 8,
            "PART": 9,
            "PRON": 10,
            "PROPN": 11,
            "PUNCT": 12,
            "SCONJ": 13,
            "SYM": 14,
            "VERB": 15,
            "CCONJ": 16,
            "EOL": 17,
            "SPACE": 18,
            "X": 19,
            "UNDEFINED": 19,
        }
        
        self.word2index = {
            "--PADDING--": 0,
            "--UNKNOWN_WORD--": 1,
        }
        
        self.label2index = {
            "B-LOC": 0,
            "I-LOC":1,
            "O":2,
            "PAD": 3
        }
        
        self.MAX_SENTENCE = 0

        
    def get_utility(self, file_path):
        with open(file_path, 'r') as file:
            word_set = set(line.strip() for line in file)
        return word_set

    def to_iob_format(self, text, toponyms):
        words = text.split(" ")
        iob_tags = ['O'] * len(words)
        iob_numerical = [self.label2index['O']] * len(words)
        geoIDs = [None]*len(words)

        char_count = 0
        for idx, _ in enumerate(words):
            if idx > 0:
                char_count += len(words[idx - 1]) + 1  # Add 1 for the space between words

            for toponym in toponyms:
                start_char = int(toponym["start"])
                end_char = int(toponym["end"])

                if char_count == start_char:
                    iob_tags[idx] = 'B-LOC'
                    iob_numerical[idx] = self.label2index['B-LOC']
                    if toponym["geonameid"] is not None:
                        geoIDs[idx] = int(toponym["geonameid"])
                elif char_count > start_char and char_count < end_char:
                    iob_tags[idx] = 'I-LOC'
                    iob_numerical[idx] = self.label2index['I-LOC']
                    if toponym["geonameid"] is not None:
                        geoIDs[idx] = int(toponym["geonameid"])
        return list(iob_tags), list(iob_numerical), list(geoIDs)
    
    def extract_corpus_from_text(self, documents):
        corpus = []
        for index, text in enumerate(documents):
            word_tuples = self.prep_sentence(text)
            features = self.sentence_to_features(index, word_tuples, False)
            features_vector = [list(feat.values()) for feat in features.copy()]
            corpus.append({'text': text, 'toponyms': [], 'labels': [], 'geoIDs':[],'features':features, 'features_vector': features_vector})
        return corpus
            
    def extract_train_data(self, data_path="./data/dataset/lgl.xml"):
        saved_file_path = 'data/saved_data/Preprocess/corpus.pkl'
        if os.path.exists(saved_file_path):
            print("Retrieving Corpus from Saved Data")
            self.retrieve_corpus()
            print(f"Corpus has {len(self.corpus)} documents")
        else:
            print("Creating New Corpus")
            tree = ET.parse(data_path)
            root = tree.getroot()
            self.extract_corpus_from_root(root)
            print(f"Corpus has {len(self.corpus)} documents")
    
    def extract_corpus_from_root(self, root):
        corpus = []
        para2index = []
        article_data = root.findall('.//article')
        with alive_bar(len(article_data), force_tty=True) as bar:
            count = 2
            for index, article in enumerate(article_data):
                words = (article.find('text').text).split(" ")
                self.MAX_SENTENCE = max(self.MAX_SENTENCE , len(words))
                converted_sent = []
                for w in words:
                    cleaned_token = (''.join(char for char in w if char.isalnum() or char.isspace())).lower()
                    if cleaned_token not in self.word2index.keys():
                        self.word2index[cleaned_token] = count
                        count += 1
                    converted_sent.append(self.word2index[cleaned_token])
                bar()
                para2index.append(converted_sent)
        article_data = root.findall('.//article')
        with alive_bar(len(article_data), force_tty=True) as bar:
            for index, article in enumerate(article_data):
                text = article.find('text').text
                data = []

                for toponym_tag in article.findall('.//toponym'):
                    toponym_dict = {
                        'phrase': toponym_tag.find('phrase').text,
                        'start': toponym_tag.find('start').text,
                        'end': toponym_tag.find('end').text,
                        "geonameid": None,
                        'name': None,
                        'fclass': None,
                        'fcode': None,
                        'lat': None,
                        'lon': None,
                        'country': None,
                        'admin1': None,
                    }
                    gaztag = toponym_tag.find("gaztag")
                    if gaztag:
                        toponym_dict.update({
                            'geonameid': int(gaztag.get('geonameid')) if gaztag.get('geonameid') is not None else None,
                            'name': gaztag.find('name').text if gaztag.find('name') is not None else None,
                            'fclass': gaztag.find('fclass').text if gaztag.find('fclass') is not None else None,
                            'fcode': gaztag.find('fcode').text if gaztag.find('fcode') is not None else None,
                            'lat': float(gaztag.find('lat').text) if gaztag.find('lat') is not None else None,
                            'lon': float(gaztag.find('lon').text) if gaztag.find('lon') is not None else None,
                            'country': gaztag.find('country').text if gaztag.find('country') is not None else None,
                            'admin1': gaztag.find('admin1').text if gaztag.find('admin1') is not None else None,
                        })
                    data.append(toponym_dict)
                #Extract all relevant formats
                iob, iob_numerical, geoIDs = self.to_iob_format(text, data)
                word_tuples = self.prep_sentence(text, iob)
                features = self.sentence_to_features(index, word_tuples)
                features_vector = [list(feat.values()) for feat in features.copy()]
                corpus.append({'text': text, 'toponyms': data, 'geoIDs':geoIDs, 'labels': iob, 'labels_numerical': iob_numerical, 'features':features, 'features_vector': features_vector})
                bar()
        self.corpus = corpus
        self.save_corpus()
        return corpus
        
    def prep_sentence(self, text, labels=None):
        sent = []
        text = text.replace("\n", " ")
        if labels:
            for word, label in zip(text.split(" "), labels):
                sent.append((word, label))
        else:
            for word in text.split(" "):
                sent.append((word))
        return sent
    
    def sentence_to_features(self, doc_index, sent, train_data=True):
        return [self.extract_features_from_word(doc_index, sent, i, train_data) for i in range(len(sent))]
    
    def extract_features_from_word(self, doc_index, sent, word_index, train_data=True):
        if train_data:
            word = sent[word_index][0]
            # start_index = sum(len(w) + 1 for w,_ in sent[:word_index])
        else:
            word = sent[word_index]
            # start_index = sum(len(w) + 1 for w in sent[:word_index])
        
        cleaned_token = (''.join(char for char in word if char.isalnum() or char.isspace())).lower()
        token = self.nlp(word)
        features = {
            # 'word_index': word_index, 
            # 'doc_index': doc_index,
            # 'start_index': start_index,
            'word.len': len(word),
            'has_internal_punctuations': int('.' in word or ',' in word),
            'BOS': int(word_index == 0),
            'EOS': int(word_index == len(sent)-1),
            'word':cleaned_token,
            'word_shape': re.sub(r'([A-Z]+)|([a-z]+)|(\d+)', lambda m: 'A' if m.group(1) else 'a' if m.group(2) else '0', word),
            # 'capitalized': int(word.istitle()),
            # 'allCaps': int(word.isupper()),
            # 'numerical': int(word.isdigit()),
            # 'contains_numerical': int(any(char.isdigit() for char in word)),
            'word.lemma': " ".join([w.lemma_ for w in token]),
            'word[-3:]': cleaned_token[-3:], #Suffix
            'word[:3]': cleaned_token[:3], #Prefix
            'pos_tag': token[0].pos_ if len(token) > 0 else 'UNDEFINED',
            'Is_Preposition': int(cleaned_token in self.prepositions),
            'Is_Stopword': int(cleaned_token in self.stopwords),                
            '-1:pos_tag': 'UNDEFINED',
            '+1:pos_tag': 'UNDEFINED',
            '-1:word_shape':'UNDEFINED',
            '+1:word_shape':'UNDEFINED',
            '-1:word': 'UNDEFINED',
            '+1:word':'UNDEFINED',
            '-1:has_internal_punctuations': 0,
            '+1:has_internal_punctuations': 0,
            '-1:Is_Preposition': 0,
            '+1:Is_Preposition': 0,
        }
        
        if word_index > 0:
            if train_data:
                prev_word = sent[word_index-1][0]
            else:
                prev_word = sent[word_index-1]
            prev_token = self.nlp(prev_word)
            prev_cleaned = (''.join(char for char in prev_word if char.isalnum() or char.isspace())).lower()
            features.update({
                '-1:pos_tag': prev_token[0].pos_ if len(prev_token) > 0 else 'UNDEFINED',
                '-1:word_shape': re.sub(r'([A-Z]+)|([a-z]+)|(\d+)', lambda m: 'A' if m.group(1) else 'a' if m.group(2) else '0', prev_word),
                '-1:word':prev_cleaned,
                '-1:has_internal_punctuations': int('.' in prev_word or ',' in prev_word),
                '-1:Is_Preposition': int(prev_cleaned in self.prepositions),
            })

        if word_index < len(sent)-1:
            if train_data:
                next_word = sent[word_index+1][0]
            else:
                next_word = sent[word_index+1]
            next_token = self.nlp(next_word)
            next_cleaned = (''.join(char for char in next_word if char.isalnum() or char.isspace())).lower()
            features.update({
                '+1:pos_tag': next_token[0].pos_ if len(next_token) > 0 else 'UNDEFINED',
                '+1:word_shape': re.sub(r'([A-Z]+)|([a-z]+)|(\d+)', lambda m: 'A' if m.group(1) else 'a' if m.group(2) else '0', next_word),
                '+1:word':next_cleaned,
                '+1:has_internal_punctuations': int('.' in next_word or ',' in next_word),
                '+1:Is_Preposition': int(next_cleaned in self.prepositions),
            })

        if self.gaz != None:
            features.update({'IsLOCGaz': int(self.gaz.check_location_exist(cleaned_token.lower()))})
        return features
    
    def save_corpus(self):
        with open('data/saved_data/Preprocess/corpus.pkl','wb') as f:
            pickle.dump(self.corpus, f)
        with open('data/saved_data/Preprocess/word2index.pkl','wb') as f:
            pickle.dump(self.word2index, f)
        with open('data/saved_data/Preprocess/max_sentence.pkl','wb') as f:
            pickle.dump(self.MAX_SENTENCE, f)
            
    def retrieve_corpus(self):
        with open('data/saved_data/Preprocess/corpus.pkl','rb') as f:
            loaded_corpus = pickle.load(f)
            self.corpus = loaded_corpus
        with open('data/saved_data/Preprocess/word2index.pkl','rb') as f:
            loaded_word2index= pickle.load(f)
            self.word2index = loaded_word2index
        with open('data/saved_data/Preprocess/max_sentence.pkl','rb') as f:
            loaded_MAX_SENTENCE= pickle.load(f)
            self.MAX_SENTENCE = loaded_MAX_SENTENCE
    
# Code not in use, previously for tweet dataset
# def extract_tweet_train_data(self, data_path="./data/dataset/small"):
#         saved_file_path = 'data/saved_data/corpus.pkl'
#         if os.path.exists(saved_file_path):
#             print("Retrieving Corpus from Saved Data")
#             self.retrieve_corpus()
#             print(f"Corpus has {len(self.corpus)} documents")
#         else:
#             print("Creating New Corpus")
#             self.extract_train_data_tweets()
#             print(f"Corpus has {len(self.corpus)} documents")
    
#     def tweet_to_iob_format(self, text, toponyms):
#         words = text.split(" ")
#         iob_tags = ['O'] * len(words)

#         char_count = 0
#         for idx, _ in enumerate(words):
#             if idx > 0:
#                 char_count += len(words[idx - 1]) + 1  # Add 1 for the space between words
#             for toponym in toponyms:
#                 start_char = int(toponym["start"])
#                 end_char = int(toponym["end"])
#                 if char_count == start_char:
#                     iob_tags[idx] = "B-" + toponym["label"].upper()
#                 elif char_count > start_char and char_count < end_char:
#                     iob_tags[idx] = "I-" + toponym["label"].upper()
#         return list(iob_tags)
    
#     def extract_train_data_tweets(self, directory="./data/dataset/small"):        
#         files = [file for file in os.listdir(directory) if file.endswith('.xml')]
#         labels_dict = self.extract_labels_from_tweets()
#         labels_dict = {key: labels_dict[key] for key in files}
#         corpus = []
        
#         with alive_bar(len(files), force_tty=True) as bar:
#             for index, file_name in enumerate(files):
#                 file_path = os.path.join(directory, file_name)
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     file_content = file.read()
#                     pattern = r"<[^>]+>"
#                     cleaned_content = re.sub(pattern, "", file_content)
#                     cleaned_content = cleaned_content.replace("\n", " ")
#                     text = cleaned_content
                    
#                     #Extract all relevant formats
#                     iob = self.tweet_to_iob_format(text, labels_dict[file_name])
#                     word_tuples = self.prep_sentence(text, iob)
#                     features = self.sentence_to_features(index, word_tuples)
#                     corpus.append({'text': text, 'toponyms': labels_dict[file_name], 'labels': iob, 'features':features})
#                     bar()
#         self.corpus = corpus
#         self.save_corpus()
#         return corpus
        
#     def extract_labels_from_tweets(self, file_path = "./data/dataset/small.labels"):
#         print("Extracting labels")
#         labels_dict = defaultdict(list)
#         with open(file_path, 'r') as file:
#             labels = set(line.strip() for line in file)
#         for label_str in labels:
#             label = label_str.split(" ")
#             if label[0] == "closeType" or label[4] == "Location":
#                 continue
#             labels_dict[label[1]].append({
#                 "start": int(label[2]),
#                 "end": int(int(label[2]) + int(label[3])), 
#                 "label": label[4],
#             })
#         return labels_dict