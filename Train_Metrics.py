import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             make_scorer,
                             r2_score,
                             roc_curve, 
                             auc,
                             precision_recall_curve)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
    
class Train_Metrics:
    def __init__(self, model_class, param_grid, cv, scoring, X_train, X_test, y_train, y_test, random_state):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        
    def getTreeVars(self, TREE, varNames):
        tree_ = TREE.tree_
        varName = [varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        parameter_list = [varNames[i] for i in set(tree_.feature) if i != _tree.TREE_UNDEFINED]
        return parameter_list

    def grid_search_train_metrics(self):
        grid_search = GridSearchCV(self.model_class(random_state = self.random_state), param_grid=self.param_grid, cv=self.cv, \
                                   scoring=self.scoring, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        print('Best hyperparameters for GridSearchCV: ', grid_search.best_params_, 'Score:', round(grid_search.best_score_,2))
        
        # Handle 'random_state' argument correctly
        best_params = grid_search.best_params_
        if 'random_state' in best_params:
            del best_params['random_state']
            
        #because new makes are in this data, there are extra dummy variables. We need to remove the rows and columns containing these data
        extra_vars_1 = []
        extra_vars_2 = []
        for i in list(self.X_train.dtypes.index):
          if i not in list(self.X_test.dtypes.index):
            if i != 'stock':
              extra_vars_1.append(i)
        for i in list(self.X_test.dtypes.index):
          if i not in list(self.X_train.dtypes.index):
            if i != 'stock':
              extra_vars_2.append(i)

#         #mask to identify rows where columns have a value of 1
#         mask1 = (self.X_train[extra_vars_1] == 1).any(axis=1)
#         mask2 = (self.X_test[extra_vars_2] == 1).any(axis=1)

#         #drop the rows where the mask is True
#         self.X_train = self.X_train.loc[~mask1]
#         self.X_test = self.X_test.loc[~mask2]

        #drop extra_vars columns
        self.X_train.drop(extra_vars_1, axis=1, inplace=True)
        self.X_test.drop(extra_vars_2, axis=1, inplace=True)

                        
        clf = self.model_class(random_state = self.random_state, **grid_search.best_params_)
        clf.fit(self.X_train, self.y_train)

        #predict
        y_pred_train = clf.predict(self.X_train)
        accuracy_train = accuracy_score(self.y_train, y_pred_train)
        f1_train = f1_score(self.y_train, y_pred_train)

        y_pred_test = clf.predict(self.X_test)
        accuracy_test = accuracy_score(self.y_test, y_pred_test)
        f1_test = f1_score(self.y_test, y_pred_test)

        y_proba = clf.predict_proba(self.X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)

        #Accuracy Scores
        print('\n\n', self.model_class.__name__, "Train accuracy:", round(accuracy_train,3))
        print('-----------------\n', self.model_class.__name__, "Train F1-score:", round(f1_train, 3))
        print('-----------------\n', self.model_class.__name__, "Test accuracy:", round(accuracy_test, 3))
        print('-----------------\n', self.model_class.__name__, "Test F1-score:", round(f1_test, 3),'\n-----------------')

        # ROC AUC
        plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel(f'False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Test Data - {self.model_class.__name__}')
        plt.legend(loc='lower right')
        plt.show()

        #Confusion Matrix
        cm = confusion_matrix(self.y_train, y_pred_train, labels=clf.classes_, normalize = 'true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        disp.plot()
        plt.title(f'Confusion Matrix (TRAINING) for {self.model_class.__name__}')
        
        if self.model_class.__name__ == 'DecisionTreeClassifier':
            #Export graphviz tree
            feature_cols = list(self.X_train.columns.values)
            tree.export_graphviz(clf, out_file = 'tree_file.txt', \
                                 feature_names = feature_cols, class_names = ['0','1'], filled = True, rounded = True, impurity = False)
            print('-----------------------------\nExported GraphViz file \'tree_file.txt\'')
            vars_selected = self.getTreeVars(clf, feature_cols)
            print(f'\n\nDecision Tree Selected These Variables:\n', vars_selected)
        if self.model_class.__name__ == 'LogisticRegression':
            #See the coefficient weights
            lrlist = str(clf.coef_.tolist()).replace('[','').replace(']','').split(',')
            lr_coefs = pd.DataFrame([self.X_train.dtypes.index, lrlist]).T
            lr_coefs.columns = ['Variable', 'Weight']
            print(lr_coefs)
        return clf
            
           

        

