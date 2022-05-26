import matplotlib.pyplot as plt
import seaborn as sns

class FeatureDist:
    def __init__(self, df):
        self.df = df

    def get_boxplot(self, feature_name):
        sns.boxplot(self.df[feature_name])
        plt.show()

    def get_hist(self, feature_name):
        plt.hist(self.df[feature_name])
        plt.xlabel(feature_name)
        plt.show()


