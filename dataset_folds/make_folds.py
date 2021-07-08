import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
    df = pd.read_csv("../labels/trainLabels15.csv")
    df['fold'] = 0
    skf = StratifiedKFold(n_splits=5,shuffle = True, random_state = 42)
    for i,(train_index, test_index) in enumerate(skf.split(df['image'].values,df['level'].values)):
        df.iloc[test_index, -1] = i
    df = pd.get_dummies(df,columns = ['level'])
    df.to_csv("trainFolds15.csv")

