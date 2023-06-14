import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class Encode_Normalize_Multicol:
    def __init__(self, df, max):
        self.df = df
        self.max = max
        self.obj_list = []
        self.numeric_list = []
        for i  in self.df.dtypes.index:
            if i == 't1':
                continue
            elif self.df.dtypes[i] in ['int64', 'int32','float64', 'float32']:
                self.numeric_list.append(i)
            else:
                self.obj_list.append(i)        
        
    def encode(self):
        #Add dummy variables for all object type variables (just make now)
        dummies = pd.get_dummies(self.df[self.obj_list])
        self.df.drop(columns = self.obj_list, inplace = True)
        self.df = self.df.join(dummies)
        print('encoded')
#         return(self.df)
        
    def normalize(self):
        self.scaler = MinMaxScaler()
        self.cols = []
        for i in self.df.dtypes.index:
            if i != 't1':
                self.cols.append(i)
        self.df[self.cols] = self.scaler.fit_transform(self.df[self.cols])
        print('normalized')
        return(self.df)
    
    def multicol(self):
        corr_max = self.max
        while True:
            corr = self.df.drop('t1', axis=1).corr().abs()  # Ignore 't1' when calculating correlation
            high_corr_combinations = corr[corr > corr_max].stack().index.tolist()
            mlt = list(set(map(tuple, map(sorted, [(i[0], i[1]) for i in high_corr_combinations if i[0] != i[1]]))))
            print(f'------------\nmulticollinear features:\n{mlt}')

            for i in mlt:
                new_name = str(i).replace("'", "").replace('(', '').replace(')', '').replace(', ', '_')
                pca = PCA(n_components=1).fit(self.df[list(i)])
                new_feature = pca.transform(self.df[list(i)])
                self.df[new_name] = new_feature

            for i in mlt:
                for j in i:
                    self.df.drop(str(j), axis=1, inplace=True, errors='ignore')

            corr_max = self.max
            corr = self.df.drop('t1', axis=1).corr().abs()  # Ignore 't1' when calculating correlation
            high_corr_combinations = corr[corr > corr_max].stack().index.tolist()
            mlt2 = list(set(map(tuple, map(sorted, [(i[0], i[1]) for i in high_corr_combinations if i[0] != i[1]]))))
            print(f'....\nMulticollinear features combined.')

            if len(mlt2) == 0:
                break

            print('------------')
        return(self.df)

    def encode_normalize_multicol(self):
        self.encode()
#         self.normalize()
#         self.multicol()
        return(self.df)