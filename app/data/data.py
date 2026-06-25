import pandas as pd 
import numpy as np
import os

# Get the directory where this file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'creditcard.csv')#Kindly download the dataset from the link  given in the  documentation  and save it as creditcard.csv inorder to get this running  we are sorry to not upload the dataset directly due to its  large size trained on 30 features and 284k rows  

data=pd.read_csv(csv_path)
print(data.shape)
print(data[data['Class']==0].shape)
not_noice=data[data['Class']==0].sample(n=492,random_state=42)
noice=data[data['Class']==1]
print(noice.shape)  
inp=pd.concat([noice,not_noice],ignore_index=True)

def load_data():
    input_data=inp.drop(columns=['Class'])
    return input_data
    
    

    



    
    
