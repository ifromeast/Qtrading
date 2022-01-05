
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class LGBModel:
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
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)

        params = {"objective": 'mse', "verbosity": -1}
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=50,
            verbose_eval=True,
            **kwargs
        )
        self.df = df

    def predict(self):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        y_pred = pd.Series(self.model.predict(self.df[self.features].values), index=self.df.index)
        print('R2可决系数', r2_score(self.df['label'], y_pred))
        return y_pred
