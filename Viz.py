#viz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Preprocessing import Preprocessing

#make it pretty
sns.set()

class Visualization:
    def __init__(self, df):
        self.df = df
        self.obj_list = []
        self.numeric_list = []
        for i  in self.df.dtypes.index:
            if i == 't1':
                continue
            elif self.df.dtypes[i] in ['int64', 'int32','float64', 'float32']:
                self.numeric_list.append(i)
            else:
                self.obj_list.append(i)        
                
    def pie_charts(self):
        for i in self.obj_list:
            try:
                g1 = self.df.groupby(i)['t1'].mean()
                x = self.df[i].value_counts(dropna = True)
                data = pd.DataFrame({'Target propensity per class': g1,
                              'Total members in each class': x}) 
                print(data.sort_values('Target propensity per class', ascending = False))
            except Exception as e:
                print(f"Error encountered for variable {i}: {e}")
                continue
            theLabels = x.axes[0].tolist()
            theSlices = list(x)

            newLabels = []
            newSlices = []
            otherSlices = []

            for j in zip(theSlices, theLabels):
                if j[0] < 0.04 * sum(theSlices):
                    otherSlices.append(j[0])
                else:
                    newLabels.append(j[1])
                    newSlices.append(j[0])

            if len(otherSlices) > 0:
                newLabels.append('Other')
                newSlices.append(sum(otherSlices))

            plt.pie(newSlices, labels = newLabels, startangle = 90, autopct = '%1.0f%%')
            plt.title(i)
            plt.show()
            print('---------------------------------------------\n\n')
            
    def hists(self):
        for i in self.numeric_list:
            plt.hist(self.df[i], bins = 20)
            plt.xlabel(i)
            plt.show()

    def all_viz(self):
        self.pie_charts()
        self.hists()