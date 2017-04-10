import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import mean_absolute_error


def y_trainn2(file_):
    target = []    
    file2_=file_[:21]
    file2_=str(file2_)+'.csv'
    df = pd.read_csv(file2_ , header=0) 
    target.append(len(df.index))
    return target
def load_test():
    X_test = []
    test_id = []
    y_test = []
    print('Read test images')
    folders = ['test_set']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('./', fld, '*.png')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            test_id.append(fl)
            y_test.append(y_trainn2(fl))
            
    test_target = np.array(y_test, dtype=np.uint8)

    #train_target = np_utils.to_categorical(train_target, 8)

    return test_target, test_id

test_target, test_id=load_test()

path = "./"
all_files = glob.glob(os.path.join(path, "*.csv")) #make list of paths

dfs = []
for file in all_files:
    dfs.append(pd.read_csv(file))
big_frame = pd.concat(dfs, axis=1)

big_frame['mean']=(big_frame.select_dtypes(include=['float'])).mean(axis=1)
big_frame=big_frame.drop('prediction',axis=1)
big_frame=big_frame.T.groupby(level=0).first().T


score = mean_absolute_error(test_target, big_frame['mean'])
print('Holdout set score is: ', score)
