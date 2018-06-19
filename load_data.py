import knn
import pandas as pd

df = pd.read_csv('KNN_Project_Data')
knn.nearest_neighbors_workflow(df)