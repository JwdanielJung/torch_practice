import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['target'] = np.where(train['defects'] == False, 0, 1)

# experiment

# combination (n_estimators, learning_rate, max_depth, subsample, reg_alpha)

combi = []

# increase max_depth
combi.append([100, 0.2, 3, 0.5, 0.1])
combi.append([100, 0.1, 4, 0.5, 0.05])
combi.append([100, 0.1, 5, 0.5, 0.1]) 
combi.append([100, 0.1, 7, 0.5, 0.1]) 
combi.append([100, 0.1, 10, 0.5, 0.1]) 
combi.append([100, 0.1, 12, 0.5, 0.1]) 

# increase n_estimators
combi.append([200, 0.2, 4, 0.5, 0.1]) 

# increasing learning_rate
combi.append([100, 0.3, 4, 0.5, 0.1]) 

# decreasing learning_rate
combi.append([100, 0.1, 4, 0.5, 0.1]) 
combi.append([100, 0.05, 4, 0.5, 0.1]) 

# without subsample
combi.append([100, 0.1, 4, 0, 0.1]) 

# decrease reg_lambda
combi.append([100, 0.1, 4, 0.5, 0.05]) 

result_list = []
max_acc = 0
max_combi = []

for c in combi:
  model = XGBClassifier(n_estimators=c[0], # number of tree
                        learning_rate=c[1], 
                        max_depth=c[2], 
                        subsample=c[3], 
                      reg_alpha=c[4], 
                      random_state = 23)

  score = cross_val_score(model , train.iloc[:,1:-2] , train.iloc[:,-2] , scoring='accuracy',cv=3)
  acc = np.round(np.mean(score), 4)
  
  score = cross_val_score(model , train.iloc[:,1:-2] , train.iloc[:,-2] , scoring='f1',cv=3)
  f1 = np.round(np.mean(score), 4)
  
  c.append(acc)
  c.append(f1)
  result_list.append(c)
  if acc > max_acc:
        max_acc = acc
        max_combi = c
  print(f'acc: {acc}')
  print(f'f1: {f1}')

print(max_acc)
print(max_combi)

# Q2 answer
# max_combi = [100, 0.05, 4, 0.5, 0.1, 0.8148, 0.4906]
# has highest accuracy & F1-score
columns = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'reg_alpha','acc','f1-score']
result_df = pd.DataFrame(combi,columns = columns)
print(result_df)

# model fitting
model = XGBClassifier(n_estimators=max_combi[0], # number of tree
                        learning_rate=max_combi[1], 
                        max_depth=max_combi[2], 
                        subsample=max_combi[3], 
                      reg_alpha=max_combi[4], 
                      random_state = 23)

model.fit(train.iloc[:,1:-2], train.iloc[:,-1])
final = model.predict(test.iloc[:,1:])

# Q3 answer
result = pd.DataFrame(final)
print(result.shape)
result.to_csv('2023-21408_pred.csv', index=False)