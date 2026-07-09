import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df =pd.read_csv('perceptron\placement (1).csv')
print(df.head())
print(df.shape)
sns.scatterplot(x=df['cgpa'], y=df['resume_score'], hue = df['placed'])
plt.show()

x= df.iloc[:,0:2]
y = df.iloc[:,-1]


from sklearn.linear_model import Perceptron
p = Perceptron()
p.fit(x,y)
print(p.coef_)
print(p.intercept_)
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x.values, y.values, clf=p, legend=2)
plt.show()
