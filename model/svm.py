import pandas as pd
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class SVMModel:
    def _prepare_data(self, df, train_valid_date):
        df.dropna(inplace=True)
        df_train = df.loc[:train_valid_date]

        # 用来回测的字段
        fields = list(df.columns)
        fields.remove('label')
        fields.remove('code')
        fields.remove('date')
        self.df_valid = df.loc[train_valid_date:]
        self.features = fields
        df_feature = df_train[fields]
        df_label = df_train['label']

        X_train, X_valid, y_train, y_valid = train_test_split(df_feature, df_label, test_size=0.2)
        return X_train, X_valid, y_train, y_valid

    def fit(self, df, train_valid_date, **kwargs):
        X_train, X_valid, y_train, y_valid = self._prepare_data(df, train_valid_date)
        self.model = svm.SVR(gamma='scale')
        self.model.fit(X_train, y_train)
        self.df = df

    def predict(self):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        y_pred = pd.Series(self.model.predict(self.df[self.features].values), index=self.df.index)
        print('R2可决系数', r2_score(self.df['label'], y_pred))
        return y_pred
