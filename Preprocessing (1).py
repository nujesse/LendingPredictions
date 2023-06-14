import pandas as pd
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from fuzzywuzzy import fuzz
import re


class Preprocessing:    
    def __init__(self, df, imputer=None):
        self.df = df
        self.imputer = imputer
        
    #Renaming columns for convenience
    def renaming(self):
        dt = self.df.dtypes
        self.names = dt.index
        self.new_names = ['stock','bal', 'repair', 'npay','pct', 'tr_yr', 'tr_mk', 'tr_odo',\
                     'tr_val', 'tr_lien', 'tr_holder', 'yr', 'mk', 'odo','dob', 'mf', 'dayslate',\
                     'dti','jointincome', 'income', 'depend','netincome', 'modela']
        self.namesDict = dict(zip(self.names,self.new_names))
        self.df = self.df.rename(columns=self.namesDict)
    
    #Correcting spelling mistakes in Lien Holder values, then simplifying
    def new_holder(self):
        #Correcting spelling mistakes and simplifying Lien Holder values
        holder_mapping = {
            'Paid off': 'paid',
            'Paid off!!': 'paid',
            'paid off': 'paid',
            'Chicagp Acceptance': 'CA',
            'CHICAGO ACCEPTANCE': 'CA',
            'CHICAGO ACCEPTANCE LLC': 'CA',
            'Chicago Acceptance': 'CA',
            'CHICGAO ACCEPTANCE': 'CA',
            'Chicago acceptance': 'CA',
            'CHICAGO Acceptance llc': 'CA',
            'CHIGCAO ACCEPTANCE': 'CA',
            'Chicago Aceptance': 'CA',
            'Wheels of Chicago': 'WOC'
        }

        fuzzy_mappings = {
            'chicago acceptance': 'CA',
            'chicago acceptance llc': 'CA',
            'chicago acceptance inc': 'CA',
            'wheels of chicago': 'WOC',
            'Paid off': 'paid'
        }

        newHolder = []
        
        punctuation_pattern = r'[^\w\s]'
        for i in self.df['tr_holder']:
            i_lower_unpunc = re.sub(punctuation_pattern, '', str(i).lower())
            if i_lower_unpunc in fuzzy_mappings:
                newHolder.append(fuzzy_mappings[i_lower_unpunc])
            else:
                best_match = None
                best_ratio = 0
                for key in fuzzy_mappings:
                    ratio = fuzz.ratio(i_lower_unpunc, key)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = key

                if best_match and best_ratio >= 80:  # Adjust the similarity threshold as needed
                    newHolder.append(fuzzy_mappings[best_match])
                elif i in holder_mapping:
                    newHolder.append(holder_mapping[i])
                else:
                    newHolder.append('OTHER')    
        self.df.tr_holder = newHolder
        
    def set_targets(self):
    #set target variable 
        self.df['t1'] = np.where(self.df['tr_holder'].isin(['CA', 'WOC', 'paid']), 1, 0)

    #drop irrelevant variables all around
#         self.df.drop(['repair', 'npay', 'pct'], axis = 1, inplace = True)

#         #drop potentially irrelevant features FOR NOW
#         self.df.drop(['dayslate','dti',	'jointincome', 'income', 'depend', 'netincome'], axis = 1, inplace = True)
    #change these variables to float type
    def numeric_to_float(self):
        self.df[['bal','repair','tr_val','tr_lien']] = self.df[['bal','repair','tr_val','tr_lien']].replace(',','',regex=True)
        self.df[['bal','repair','tr_val','tr_lien']] = self.df[['bal','repair','tr_val','tr_lien']].astype('float64')
        
    #Vehicles over 10 years old are exempt from government reporting. By the manager's account, 
    #the average traded vehicle which is Exempt has around 150,000 miles
    def tr_odo(self):
        #impute
        self.df['tr_odo'] = self.df['tr_odo'].replace('EXEMPT','150000.0')

        #make tr_odo a float
        self.df['tr_odo'] = self.df['tr_odo'].astype('float64')
#         self.df['tr_odo'].fillna(self.df['tr_odo'].median(),inplace = True)
    
    #make odo a float
    def odo(self):
        #impute
        self.df['odo'] = self.df['odo'].astype('str')

        self.df['odo'] = self.df['odo'].str.lower()
        self.df['odo'] = self.df['odo'].replace('/','').replace('.','')
        self.df['odo'] = self.df['odo'].\
        replace('exempt','150000').replace('exempy','150000').replace('unknwon','150000')\
        .replace('nr','150000').replace('n/r','150000').replace('unkwnon','150000')

        self.df['odo'] = self.df['odo'].astype('float64')
        self.df['odo'].fillna(self.df['odo'].median(),inplace = True)

    
    def impute_non_trade(self, train_data=None):
        cols_with_missing = ['tr_yr', 'tr_odo', 'tr_lien']

        # replace 0 values in tr_lien with NaN
        self.df.loc[self.df['tr_lien'] == 0, 'tr_lien'] = np.nan

        if train_data is None:
            train_data = self.df.dropna(subset=cols_with_missing, axis=0)

        # select the columns to use as features for the decision tree
        features = ['yr', 'odo', 'bal']

        # create the decision tree
        tree = DecisionTreeRegressor(random_state=0)
        imputer = IterativeImputer(estimator=tree, random_state=0)

        # fit the imputer on the training data and transform both the training and test data
        train_data.loc[:, features + cols_with_missing] = imputer.fit_transform(
            train_data.loc[:, features + cols_with_missing])
        self.df[features + cols_with_missing] = imputer.transform(self.df[features + cols_with_missing])

        # create new columns
        self.df['diff_yr'] = self.df['yr'] - self.df['tr_yr']
        self.df['diff_odo'] = self.df['odo'] - self.df['tr_odo']
        self.df['diff_bal'] = self.df['bal'] - self.df['tr_lien']

    
    def tr(self, user):
        if user == 1:
            #Create mask for rows where 't1' is 1
            mask = self.df['t1'] == 1

            #Replace certain columns for rows where 't1' is 1
            self.df.loc[mask, 'bal'] = self.df.loc[mask, 'tr_lien']
            self.df.loc[mask, 'yr'] = self.df.loc[mask, 'tr_yr']
            self.df.loc[mask, 'mk'] = self.df.loc[mask, 'tr_mk']
            self.df.loc[mask, 'odo'] = self.df.loc[mask, 'tr_odo']

            #Drop trade columns
#             self.df.drop(['tr_yr','tr_mk','tr_odo','tr_val','tr_lien','tr_holder'], axis = 1, inplace = True)
            self.df.drop(['tr_yr','tr_mk','tr_odo','tr_lien','tr_holder'], axis = 1, inplace = True)
            print('\n-----\n\ntrade vars replaced and dropped')

        elif user == 0:
            #Drop trade columns
#             self.df.drop(['tr_yr','tr_mk','tr_odo','tr_val','tr_lien','tr_holder', 't1'], axis = 1, inplace = True)
            self.df.drop(['tr_yr','tr_mk','tr_odo','tr_lien','tr_holder', 't1'], axis = 1, inplace = True)
            print('\n-----\n\ntrade vars not replaced, but dropped (along with t1)')

        self.df.mk = self.df.mk.str.lower()

    #drop the rows where interest % is 0; these are wholesale deals
    def drop_wholesale(self):
        self.df.drop(self.df[self.df['pct'] == 0].index, inplace=True)    
    
    #Drop all cash deals, since they did not take a loan
    def drop_cash_deals(self):
        self.df.drop(self.df[self.df['bal'] == 0].index, inplace = True)    
    
    
    #Since negative repair value is a typo and means the same as positive, let's change everything to be positive
    def abs_repair(self):
        self.df['repair'] = self.df['repair'].abs()
      
    #Rather than imputing DOB with the median, which may be a sub-par strategy, 
    #I may want to use a decision tree to impute this data in the future.
    #Currently, the DOB is being restricted by out DB provider
    def dob(self):
#         #change dob variable to datetime and select just the year
#         self.df['dob'] = pd.to_datetime(self.df['dob'])
#         self.df['dob'] = self.df['dob'].dt.year

#         #Make flag variable for the rows I'm about to impute
#         self.df['dob_flag'] = np.where(self.df['dob'].isna(),1,0)
#         self.df['dob_flag'] = self.df['dob_flag'].astype('object')

#         #impute missing dobs as the median dob
#         self.df['dob'].fillna(self.df['dob'].median(),inplace = True)
        self.df.drop('dob', axis = 1, inplace = True)
        
    def mf(self):
        #make an 'other' category from the n/a values
        self.df['mf'] = np.where(self.df['mf'].isna(), 'empty', self.df['mf'])
    
    #Since n/a dependents means zero dependents, we impute n/a with 0   
    def depend(self):
        self.df['depend'].fillna(0,inplace = True)
                    
    def days_late(self):   
        self.df['dayslate'] = self.df['dayslate'].astype('float')
        self.df['dayslate'] = np.where(self.df['dayslate'] < 0, 'early', np.where(self.df['dayslate'] > 0, 'late', 'on time'))
        self.df['dayslate'] = self.df['dayslate'].astype('object')

    #drop stock
    def drop_stock(self):
        self.df.drop('stock', axis = 1, inplace = True)
    
    #drop irrelevant
    def drop_extras(self):
        self.df.drop(['tr_val', 'dayslate', 'modela', 'repair', 'npay', 'pct', 'dti', 'jointincome', 'income', 'depend', 'netincome'], axis = 1, inplace = True)

    def truncate_outliers(self):
        self.numeric_list = []
        for i  in self.df.dtypes.index:
            if i == 't1':
                continue
            elif self.df.dtypes[i] in ['int64', 'int32','float64', 'float32']:
                self.numeric_list.append(i)                
        z_val = 5
        for i in self.numeric_list:
            mean = self.df[i].mean()
            sd = self.df[i].std()
            min_val = mean - z_val*sd
            max_val = mean + z_val*sd

            min_mask = self.df[i] <= min_val 
            max_mask = self.df[i] >= max_val 

            min_rows = self.df.loc[min_mask].index.tolist()
            max_rows = self.df.loc[max_mask].index.tolist()
            self.df.loc[min_rows, i] = min_val
            self.df.loc[max_rows, i] = max_val
        print('\n-----\n\noutliers truncated\n\n--------\n')
                      

    def process(self):
        len_before_processing = len(self.df)
        self.renaming()
        self.new_holder()
        self.set_targets()
        self.numeric_to_float()
        self.tr_odo()
        self.odo()
        self.impute_non_trade()

        # Reset trades (1) or don't (0)
        keep_trades = int(input("Would you like to reset the trades?\n\n1=True\n\n0=False\n\n"))
        self.tr(keep_trades)

        self.drop_wholesale()
        self.drop_cash_deals()
        self.abs_repair()
        self.dob()
        self.mf()
        self.depend()
        self.days_late()
        # self.drop_stock()
        self.drop_extras()
        print('Variables fixed.')
        len_after_processing = len(self.df)
        percentage_deleted = round(100 * (len_before_processing - len_after_processing) / len_before_processing, 1)
        print(f'\n-----\n\n{percentage_deleted}% of the values ({len_before_processing - len_after_processing} values) were wholesale and cash deals (deleted).\n')
        self.truncate_outliers()
        print('\n')
        return self.df
