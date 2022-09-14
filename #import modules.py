#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
#import dataset
df = pd.read_csv("data/train_data_titanic.csv")

df.head()
df.info()
plt.figure(figsize=(15,5))
sns.distplot(df['Age'],bins=40)
#Remove the columns model will not use
df.drop(['Name','Ticket'],axis=1,inplace=True)

sns.pairplot(df[['Survived','Fare']], dropna=True)
#data observing
df.groupby('Survived').mean()
#data observing
df['SibSp'].value_counts()
df['Parch'].value_counts()
df['Sex'].value_counts()
#Handle missing values
