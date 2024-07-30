import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin
import joblib
import paddle

train_fids = paddle.load('../data_preprocess/input_data/train_fids.pdparams')
test_fids = paddle.load('../data_preprocess/input_data/test_fids.pdparams')
val_fids = paddle.load('../data_preprocess/input_data/val_fids.pdparams')


class DataFrameImputer(TransformerMixin):
    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        fill = []
        for c in X:
            if X[c].dtype == np.dtype('O'):
                try:
                    fill.append(X[c].value_counts().index[0])
                except:
                    if c in self.numeric_cols:
                        fill.append(0)
                    else:
                        fill.append(False)

            else:
                fill.append(X[c].mean())
        self.fill = pd.Series(fill, index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def process(numeric_cols, df):

    fit_imputer = DataFrameImputer(numeric_cols).fit(df)
    df = fit_imputer.transform(df)

    fit_scaler = get_scaler().fit(df[numeric_cols])
    df[numeric_cols] = fit_scaler.transform(df[numeric_cols])
    df = df.drop(columns=['fid'])
    return df


def get_scaler():
    return MinMaxScaler()


features = pd.read_csv('features.tsv',sep='\t')

features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.dropna(axis=1, how='all', inplace=True)




float_feature_names = [
    feature_name for (feature_name, dtype) in features.dtypes.items() if dtype in [ np.float32, np.float64]]
integer_feature_names = [
    feature_name for (feature_name, dtype) in features.dtypes.items() if dtype in [np.int32,np.int64]]
numeric_cols = float_feature_names + integer_feature_names
non_numeric_cols = [
    feature_name for (feature_name, dtype) in features.dtypes.items() if feature_name not in numeric_cols]

train_df = features[features['fid'].isin(train_fids)]
test_df = features[features['fid'].isin(test_fids)]
val_df = features[features['fid'].isin(val_fids)]

train_processed_features = process(numeric_cols, train_df)
test_processed_features = process(numeric_cols, test_df)
val_processed_features = process(numeric_cols, val_df)

print(train_processed_features.shape)

kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(train_processed_features)

train_labels = kmeans.labels_
test_labels = kmeans.predict(test_processed_features)
val_labels = kmeans.predict(val_processed_features)

joblib.dump(kmeans, 'kmeans_model.pkl')
paddle.save(train_labels, 'train_labels.pdparams')
paddle.save(test_labels, 'test_labels.pdparams')
paddle.save(val_labels, 'val_labels.pdparams')
