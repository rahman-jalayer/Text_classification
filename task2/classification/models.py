import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import keras
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.utils import resample, shuffle
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
from pathlib import Path
import os


# import nlpaug.augmenter.word as naw


class RealEstateTextClassification():
    def __init__(self):
        nltk.download('stopwords')
        self.num_classes = 2
        self.batch_size = 2
        self.log_dir = os.path.join(os.path.dirname(__file__), "tensorboard_data", "tb_bert")
        self.model_save_path = os.path.join(os.path.dirname(__file__), "trained_models", "bert_model.h5")
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.2,
                                                 num_labels=self.num_classes)
        self.bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                          config=self.config)
        self.bert_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.metric])

    def data_loading(self, data_file):
        data = pd.read_csv(data_file, engine='python', encoding='utf-8')

        return data

    def to_int(self, data):
        inputs = list()
        attention_masks = list()
        for doc in data['text']:
            bert_inp = self.data_transform(doc)
            inputs.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])
        inputs = np.asarray(inputs)
        attention_masks = np.array(attention_masks)
        labels = np.array(data['label'])
        return inputs, labels, attention_masks

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def clean_stopwords_shortwords(self, w):
        stopwords_list = stopwords.words('english')
        words = w.split()
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
        return " ".join(clean_words)

    def preprocess_doc(self, w):
        w = self.unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = self.clean_stopwords_shortwords(w)
        w = re.sub(r'@\w+', '', w)
        return w

    def data_transform(self, doc):
        bert_inp = self.bert_tokenizer.encode_plus(doc, add_special_tokens=True, max_length=512, padding='max_length',
                                                   truncation=True, return_attention_mask=True)
        return bert_inp


class Train(RealEstateTextClassification):
    def __init__(self, ):
        super().__init__()
        self.data_file = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.epochs = 5
        print('\nBert Model', self.bert_model.summary())
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=self.model_save_path, save_weights_only=True,
                                               monitor='val_loss',
                                               mode='min',
                                               save_best_only=True), keras.callbacks.TensorBoard(log_dir=self.log_dir)]

    def train_test_split(self):

        data_majority = self.data[self.data.Related == 0]
        data_minority = self.data[self.data.Related == 1]
        data_majority_downsampled = resample(data_majority,
                                             replace=False,
                                             n_samples=data_minority.shape[0],  # to match minority class
                                             random_state=123)
        _, self.test_data = train_test_split(data_majority_downsampled, test_size=0.10)
        self.train_data = self.data.loc[~self.data.index.isin(self.test_data.index)]

    def training_data_prepration(self, data):
        if 'Related' in data.columns:
            data = data.rename(columns={'Related': 'label'})
        data["text"] = data["Title"] + '. ' + data["Snippet"]
        related_samples1 = data[data.label == 1].reset_index(drop=True)
        related_samples2 = data[data.label == 1].reset_index(drop=True)
        related_samples1['text'] = related_samples1["Title"]
        related_samples2['text'] = related_samples2["Snippet"]
        data = shuffle(data.append(related_samples1).reset_index(drop=True))
        data = shuffle(data.append(related_samples2).reset_index(drop=True))
        return data

    def validation_data_prepration(self, data):
        if 'Related' in data.columns:
            data = data.rename(columns={'Related': 'label'})
        data["text"] = data["Title"] + '. ' + data["Snippet"]
        related_samples1 = data
        related_samples2 = data
        related_samples1['text'] = related_samples1["Title"]
        related_samples2['text'] = related_samples2["Snippet"]
        data = shuffle(data.append(related_samples1).reset_index(drop=True))
        data = shuffle(data.append(related_samples2).reset_index(drop=True))
        return data

    def data_augmentation(self, samples):
        return None
        # new_text = []
        # ##selecting the minority class samples
        # data_n = self.train_data[self.train_data.label == 1].reset_index(drop=True)
        # aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert')
        # ## data augmentation loop
        # for i in np.random.randint(0, len(data_n), samples):
        #     text = data_n.iloc[i]['text']
        #     augmented_text = aug.augment(text)
        #     new_text.append(augmented_text)
        # new = pd.DataFrame({'text': new_text, 'label': 1})
        # self.train_data = shuffle(self.train_data.append(new).reset_index(drop=True))
        #
        # return None

    def data_resampling(self):
        df_majority = self.train_data[self.train_data.label == 0]
        df_minority = self.train_data[self.train_data.label == 1]
        #     # Upsample minority class
        #     df_minority_upsampled = resample(df_minority,
        #                                      replace=True,
        #                                      n_samples=df_minority.shape[0],
        #                                      random_state=123)

        #     df_upsampled = pd.concat([df_minority, df_minority_upsampled])
        # =====================
        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,
                                           n_samples=df_minority.shape[0],  # to match minority class
                                           random_state=123)
        self.train_data = pd.concat([df_majority_downsampled, df_minority])
        print(self.train_data.label.value_counts())

    def data_preprocessing(self, data):
        data['text'] = data['text'].map(self.preprocess_doc)

    def train(self, data_file=os.path.join(os.path.dirname(__file__), "dataset", "train.csv"), retrain=0):
        self.data_file = data_file
        self.data = self.data_loading(self.data_file)
        self.train_test_split()
        self.train_data = self.training_data_prepration(self.train_data)
        self.test_data = self.validation_data_prepration(self.test_data)
        self.data_augmentation(400)
        self.data_resampling()
        train_inp, train_label, train_mask = self.to_int(self.train_data)
        test_inp, test_label, test_mask = self.to_int(self.test_data)
        model_file = Path(self.model_save_path)
        if retrain == 0 and model_file.exists():
            self.bert_model.load_weights(self.model_save_path)
            print('The last model has been loaded')
        else:
            print('Retraining...')
        self.history = self.bert_model.fit([train_inp, train_mask], train_label, batch_size=self.batch_size,
                                           epochs=self.epochs,
                                           validation_data=([test_inp, test_mask], test_label),
                                           callbacks=self.callbacks)


class Prediction(RealEstateTextClassification):
    def __init__(self, ):
        super().__init__()
        self.model_file = Path(self.model_save_path)
        if self.model_file.exists():
            self.bert_model.load_weights(self.model_save_path)
        else:
            print('There is no trained model')

    def predict(self, text):
        if not self.model_file.exists:
            print('There is no trained model')
            return 'No model'
        doc = self.preprocess_doc(text)
        bert_inp = self.data_transform(doc)
        input_ids = [bert_inp['input_ids']]
        attention_masks = [bert_inp['attention_mask']]
        input_ids = np.asarray(input_ids)
        attention_masks = np.array(attention_masks)
        preds = self.bert_model.predict([input_ids, attention_masks], batch_size=self.batch_size)
        pred_labels = np.argmax(preds.logits, axis=1)
        if pred_labels[0] == 0:
            return 'Unrelated'
        else:
            return 'Related'

