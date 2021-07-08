import json
import pandas as pd


f = open("resnet200d_ROC_LOSS.json")
resnet200 = json.load(f)

df = pd.DataFrame(resnet200)
print(df)
#
# train_loss = resnet200['train_loss']
# val_loss = resnet200['val_loss']
# val_roc = resnet200['val_roc']
#
# for i in range(len(val_roc)):
#     print( i)