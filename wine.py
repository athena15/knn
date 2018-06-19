import knn
import numpy as np
import pandas as pd
import requests
from io import StringIO

# Adding a note so I can commit changes

# Import wine classification data from UC Irvine's website
col_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
             'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
             'Proline']
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
s = requests.get(url).text
df = pd.read_csv(StringIO(s), names=col_names)


# Creating a column denoting whether a wine is above the median alcohol percentage
def high_alc(x):
	if x >= 13.05:
		return 1
	else:
		return 0


df['high_alcohol'] = df['Alcohol'].apply(high_alc)

# Run k-nearest neighbors classification to predict whether a wine will be classified as 'high alcohol' or not.
knn.nearest_neighbors_workflow(df, 'high_alcohol')
