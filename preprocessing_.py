# Script for cleaning, organizing and setting the dataset in order to be used
import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency
from scipy.stats import chi2

class Preprocessing_:
    def __init__(self,d):
        self.d = d        # self.d = data_.values
        self.rows = self.d.shape[0]
        self.cols = self.d.shape[1]


    def del_cols(self):
        todelete = ["Unnamed: 0","publisher_id","domain","placement_type","mf_user_id"]
        for val in todelete:
            print(val)
            if val in self.d.columns:
                self.d.pop(val)
        self.cols = self.d.shape[1]                             #update number of cols after deleting
        self.d.dropna(inplace=True,axis=0)

    def categorize_(self):
        print('number of columns: ',self.cols)
        # self.d.apply(pd.astype("category"),1)     #1 means by columns
        for g in range(0, self.cols-1): #So I don't categorize the Click variable
            self.d.iloc[g] = self.d.iloc[g].astype("category")

    def grouping_(self):
        self.d['device_os'].unique()
        self.d['device_os'] = np.where(self.d['device_os'] == 'BlackBerry', 'Linux&others', self.d['device_os'])
        self.d['device_os'] = np.where(self.d['device_os'] == 'ChromeOS', 'Linux&others', self.d['device_os'])
        self.d['device_os'] = np.where(self.d['device_os'] == 'Linux', 'Linux&others', self.d['device_os'])
        self.d['device_os'] = np.where(self.d['device_os'] == 'NetCast', 'Linux&others', self.d['device_os'])
        self.d['device_os'] = np.where(self.d['device_os'] == 'OSX', 'iOS_general', self.d['device_os'])
        self.d['device_os'] = np.where(self.d['device_os'] == 'iOS', 'iOS_general', self.d['device_os'])
        self.d['device_os'].unique()

    def process_(self):
        self.del_cols()
        self.categorize_()
        self.grouping_()

    def final_(self):
        vas = list(self.d.columns)
        self.d = pd.get_dummies(self.d, columns=vas[0:len(vas) - 1])

    def independ_(self,datos):
        c = list(datos.columns)
        for g in range(0, len(c)):
            for t in range(g+1, len(c)):
                table = pd.crosstab( datos.iloc[g], datos.iloc[t] )
                stat, p, dof, expected = chi2_contingency(table)
                # interpret test-statistic
                prob = 0.95
                critical = chi2.ppf(prob, dof)
                if abs(stat) >= critical:
                    print('Dependent (reject H0)     -    ',c[g],'  &  ',c[t])
                else:
                    print('Independent (fail to reject H0)     -    ',c[g],'  &  ',c[t])