import pandas as pd
import os, sys
import config
from preprocessor import PreProcessor


csvname = 'df_data.csv'
df_data = pd.read_csv(os.path.join('tmp2', csvname), index_col=0)

print('df_data')
       
csvname = 'attributes.csv'
df_attr = pd.read_csv(os.path.join('../data', csvname))

preproc = PreProcessor(config)      
df_attr = preproc.clean_text(df_attr)

print(df_attr)

with open('data.text','wt') as f:
    for col in ['q','t','d']:
        f.write(' '.join(df_data[col].tolist()))
        f.write(' ')
    f.write(' '.join(df_attr['value'].tolist()))
