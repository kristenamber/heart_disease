import pandas as pd

class DataInfo:
    def __init__(self, df):
        self.df = df

    def get_feature_names(self):
        return list(self.df.columns)


    def get_feature_types_nuls(self):
        return self.df.info()


    def get_min_max(self, feature_name):
        min_val = self.df[feature_name].min()
        max_val = self.df[feature_name].max()
        return min_val, max_val


    def get_unique(self, feature_name):
        unique_vals = self.df[feature_name].unique()
        return unique_vals





