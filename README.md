# Geoparsing: Location Entity Resolution
### By Feyi Adesanya
---
## Python libraries requirements
- keras
- tensorflow
- spacy
- pytorch
- TorchCRF
- transformers
- scikit-learn

## To Replicate
- Due to limitations on GitHub repositories file sizes, please download the following sources and add them to the appropriate directories:
    - `data/saved_data/geo_data/GeoNames`
        - `allCountries.zip`, `alternateNamesV2.zip`, `hierarchy.zip`, `featureCodes_en.txt`: [GeoNames Data](https://download.geonames.org/export/dump/)
    - `data/saved_data/glove.6B/glove.6B.50d.txt`
        - [GloVe Embeddings](http://nlp.stanford.edu/projects/glove/)
- Please run the `Prediction_Evaluation.ipynb` file to set up all preprocessing modules and initialize/create model weights, then refer to `Pipeline_Evaluation.ipynb` to extract and map coordinates directly when given text input.

## File Descriptions
- `Pre/preprocess.py`: This is the preprocessing script used to convert the dataset into a corpus with relevant tokens, features, and IOB labels.
- `Gaz/Gazetteer.py`: This is a preprocessing script for the GeoNames data used to extract the relevant location names and metadata.
- `Gaz/BKTree.py`: This is a class used for quick string matching within the gazetteer, following a BKTree implementation.
- `Dis/Disambiguation_Manager.py`: The main script for running the disambiguation module.
- `ML/Baseline_Manager.py`: The main script for training and predicting using the baseline classifier.
- `ML/BERT_Manager.py`: The main script for training and predicting using the Custom RoBERTa classifier, weights are saved to `data/saved_data/model_checkpoints`.
- `ML/BI_LSTM_Manager.py`: The main script for training and predicting using the BI-LSTM classifier, weights are saved to `data/saved_data/model_checkpoints`.
- `ML/CRF_Manager.py`: The main script for training and predicting using the CRF classifier, weights are saved to `data/saved_data/model_checkpoints`.
- `ML/SVM_Manager.py`: The main script for training and predicting using the SVM classifier, weights are saved to `data/saved_data/model_checkpoints`.
- `Results.xlsx`: contains evaluation metrics results for each model in disambiguation and extraction
- `data/dataset`: contains the LGL XML file dataset.
- `data/geo_data`: contains all GeoNames files and scripts to create pickle files from them.
- `data/glove.6B`: contains 50 dimensional GloVe embedding data.
- `data/saved_data`: contains all saved weights for models along with the saved preprocessed GeoNames and dataset data.
- `data/utility`: contains txt files that list stopwords and prepositions.
---
