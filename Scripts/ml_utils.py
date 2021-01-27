
import pandas as pd  # data wrangling
import numpy as np  # data wrangling

import re

from rake_nltk import Rake  # subject extraction library
import yake  # subject extraction library

import pickle  # saving and loading objects

from joblib import load  # saving and loading model objects
# from sklearn.decomposition import PCA

from .preprocessor import TwitterPreprocessor  # twitter preprocessor

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # sentiment analysis packages
from textblob import TextBlob  # sentiment analysis packages

from .TDSAUtils import load_tdsa  # custom functions
from .errors import TargetError  # custom functions


def load_models():
    models = {
        # ALL
        'lr_all': load('../model_zoo/GSA/lr_all.joblib'),
        'sgc_all': load('../model_zoo/GSA/sgc_all.joblib'),
        'svc_all': load('../model_zoo/GSA/svc_all.joblib'),
        'rf_all': load('../model_zoo/GSA/rf_all.joblib'),
        'nb_all': load('../model_zoo/GSA/nb_all.joblib'),
    }
    return models


def load_ensembles():
    ensembles = {
        'rf': load('../model_zoo/GSA/ens_rf.joblib'),
        'lr': load('../model_zoo/GSA/ens_lr.joblib'),
        'rfv2': load('../model_zoo/GSA/ens_rf_v2.joblib'),
        'lrv2': load('../model_zoo/GSA/ens_lr_v2.joblib'),
        'final': load('../model_zoo/GSA/final.joblib'),
        'finalv2': load('../model_zoo/GSA/finalv2.joblib')
    }
    return ensembles


def load_others():
    misc = {}
    with open('../model_zoo/GSA/count_vec.pickle', 'rb') as handle:
        misc['count_vec'] = pickle.load(handle)
    with open('../model_zoo/GSA/tfidf_vec.pickle', 'rb') as handle:
        misc['tfidf_vec'] = pickle.load(handle)
    # with open('../model_zoo/GSA/pca_new.pickle', 'rb') as handle:
    #     misc['pca'] = pickle.load(handle)
    return misc


MODELS = load_models()
ENSEMBLES = load_ensembles()
MISC = load_others()
TDSA = load_tdsa()


class MLUtils:
    def __init__(self, text, n=2, ensemble='lr', n_keywords=4, version=1):
        self.text = text
        self.n_keywords = n_keywords
        self.n = n
        self.rake = Rake()
        self.yake = yake.KeywordExtractor(n=n)
        self.version = version
        self.ensemble = ensemble

    def __prepare_text_SA(self):
        processor = TwitterPreprocessor(text=self.text)
        processor.ml_preprocess()
        return processor.text

    def __prepare_text_SE(self):
        processor = TwitterPreprocessor(text=self.text)
        processor.se_preprocess()
        return processor.text

    def __ml_predict(self, model):
        text = self.__prepare_text_SA()
        text_df = model.predict_proba([text])
        text_df = pd.DataFrame(text_df)
        return text_df

    def __get_predictions(self):
        predictions = []
        for label, model in MODELS.items():
            y_pred = self.__ml_predict(model)
            predictions.append(y_pred)
        return predictions

    def __vectorize_text(self):
        text = self.__prepare_text_SA()
        tfidf_val = MISC['tfidf_vec'].transform(MISC['count_vec'].transform([text]))
        tfidf_val = pd.DataFrame(tfidf_val.toarray())
        return tfidf_val

    def __prepare_input(self):
        predictions = self.__get_predictions()
        tfidf_val = self.__vectorize_text()
        predictions.insert(0, tfidf_val)
        if self.version == 2:
            predictions.append(pd.DataFrame(np.array([self.__vader()])))
            predictions.append(pd.DataFrame(np.array([self.__text_blob()])))
        val = pd.concat(predictions, axis=1)
        return val

    def __pca(self):
        pred = self.__prepare_input()
        pca = pd.DataFrame(MISC['pca'].transform(pred))
        return pca

    def __vader(self):
        analyzer = SentimentIntensityAnalyzer()
        sent = analyzer.polarity_scores(self.text)
        return sent['compound']

    def __text_blob(self):
        sent = TextBlob(self.text)
        return sent.sentiment.polarity

    def __predict(self):
        # pca = self.__pca()
        val = self.__prepare_input()
        if self.version == 2:
            ensemble = self.ensemble + 'v2'
            y_hat = ENSEMBLES[ensemble].predict(val)
        else:
            y_hat = ENSEMBLES[self.ensemble].predict(val)
        return y_hat[0]

    @staticmethod
    def __process_result(result):
        if int(result) == -1:
            return 'negative'
        if int(result) == 1:
            return 'positive'
        if int(result) == 0:
            return 'neutral'

    def __getSentiment(self):
        y_hat = self.__predict()
        return y_hat

    def __get_span(self, target):
        span = [match.span() for match in re.finditer(target.lower(), self.text.lower())]
        if len(span) > 1:
            span = [span[0]]
        return span

    def __prepare_tdsa(self, target):
        data = [{
            'text': self.text,
            'target': target,
            'spans': self.__get_span(target)
        }]
        return data

    def __predict_tdsa(self, target):
        data = self.__prepare_tdsa(target)
        try:
            result = TDSA.predict(data)
        except Exception:
            raise TargetError('Target value not in query text!')
        return result[0]

    def extractKeywords(self, extractor='yake'):
        text = self.__prepare_text_SE()
        if extractor == 'rake':
            self.rake.extract_keywords_from_text(text)
            return self.rake.get_ranked_phrases()[:self.n_keywords]
        if extractor == 'yake':
            return [kw for kw, s in self.yake.extract_keywords(text)[:self.n_keywords]]

    def getSentiment(self, target=None):
        if not target:
            result = self.__getSentiment()
            return self.__process_result(result=result)
        if target:
            result = self.__predict_tdsa(target)
            return self.__process_result(result=result)

    def getSentimentValue(self, target=None):
        if not target:
            result = self.__getSentiment()
            return result
        if target:
            result = self.__predict_tdsa(target)
            return result
