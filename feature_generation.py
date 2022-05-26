import pandas as pd

class FeatureGen:
    def __init__(self, df):
        self.df = df

    def ordinal_cat(self, feature_list, dict_list):
        df2 = self.df
        for i in range(len(feature_list)):
            feature_name = feature_list[i]
            dict = dict_list[i]
            df2 = df2.replace({feature_name: dict})
        return df2


    def one_hot(self, feature_list):
        df = self.df
        for feature in feature_list:
            if feature in df.columns:
                tempdf = pd.get_dummies(self.df[feature], prefix = feature)

                df = pd.merge(
                    left = df,
                    right = tempdf,
                    left_index = True,
                    right_index = True,
                )
                df = df.drop(columns= feature)

        return df


    def keep_feats(self, index_list):
        df = self.df
        new_df = df.iloc[:, index_list]
        return new_df







