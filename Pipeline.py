
from Pre.Preprocess import Preprocess
from Gaz.Gazetteer import Gazetteer
from Dis.Disambiguate_Manager import Disambiguate_Manager
from ML.BERT_Manager import BERT_Manager
from ML.CRF_Manager import CRF_Manager
from ML.Baseline_Manager import Baseline_Manager
from ML.BI_LSTM_Manager import BI_LSTM_Manager
from ML.SVM_Manager import SVM_Manager

class Pipeline:
    def __init__(self) -> None:
        self.gaz = Gazetteer()
        self.preprocess = Preprocess(self.gaz)
        self.preprocess.extract_train_data()
        self.models = {
            "BERT" : BERT_Manager(self.gaz, self.preprocess),
            "CRF": CRF_Manager(self.gaz, self.preprocess),
            "SVM": SVM_Manager(self.gaz, self.preprocess),
            "BI-LSTM": BI_LSTM_Manager(self.gaz, self.preprocess),
        }
        for mod in self.models.values():
            mod.model_class.load_weights()
        self.disambiguation = Disambiguate_Manager(self.gaz, self.preprocess)
        
    def map(self, text, model_name="BERT"):
        loc_list = self.models[model_name].new_prediction(text)
        return self.disambiguation.map_locations(text, loc_list)
    