from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import copy
import csv

def OverSampler(drop_duplicate=True):
    # Read the train dataset
    t = pd.read_csv('./Dataset/train.csv')
    X=t['text']
    yc=t['category']
    ys=t['stance']

    # Write duplicates into an external file to observe
    t[X.duplicated()].to_csv('./Dataset/duplicates.csv')

    # Remove all duplicates
    if(drop_duplicate):
        t.drop_duplicates(subset="text", keep=False, inplace=True)

    Xc_copy=copy.deepcopy(X)
    Xs_copy=copy.deepcopy(X)
    # reshape it instead 0f (n,) to be (n,1) as randomoversampler needs 2d array
    Xc_copy=Xc_copy.values.reshape(-1, 1)
    Xs_copy=Xs_copy.values.reshape(-1, 1)

    # Oversampling since the data we have is not abundant and is imbalanced
    ros = RandomOverSampler(random_state=0,sampling_strategy='auto')
    Xc_resampled, yc_resampled = ros.fit_resample(Xc_copy, yc)
    Xs_resampled, ys_resampled = ros.fit_resample(Xs_copy, ys)

    # instead of having [[sentence],[sentence]] we only need [sentence,sentence]
    Xc_resampled=Xc_resampled[:,0]
    Xs_resampled=Xs_resampled[:,0]

    # convert to pandas as this is the format needed by the preprocessor 
    Xc_resampled=pd.Series(Xc_resampled)
    Xs_resampled=pd.Series(Xs_resampled)

    # showing how oversampling affects the class balance
    print("The classification class distribution before sampling: ",sorted(Counter(yc).items()))
    print("The classification class distribution after sampling: ",sorted(Counter(yc_resampled).items()))

    print("The stance class distribution before sampling : ",sorted(Counter(ys).items()))
    print("The stance class distribution after sampling : ",sorted(Counter(ys_resampled).items()))

    print("The number of duplicates before sampling: ",len(t[t.duplicated()]))
    print("The number of duplicates for classification after sampling: ",len(Xc_resampled[Xc_resampled.duplicated()]))
    print("The number of duplicates for stance after sampling: ",len(Xs_resampled[Xs_resampled.duplicated()]))

    # write in a csv file
    # convert in a row form
    classification_data = zip(Xc_resampled,yc_resampled)
    stance_data = zip(Xs_resampled,ys_resampled)


    with open('./Dataset/classification_train_sample1.csv', "w",encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('text','category'))
        for row in classification_data:
            writer.writerow(row)

    with open('./Dataset/stance_train_sample1.csv', "w",encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('text','stance'))
        for row in stance_data:
            writer.writerow(row)


    print("size before sampling = ",len(X))
    print("size after sampling classification =",len(Xc_resampled))
    print("size after sampling stance = ",len(Xs_resampled))

OverSampler(drop_duplicate=True)