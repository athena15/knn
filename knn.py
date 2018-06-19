import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def nearest_neighbors_workflow(df, target_col_name):
	"""Takes a Pandas dataframe and target column name, and performs feature scaling and K Nearest Neighbors classification.
	Then outputs a plot of the error rate for k_neighbors sizes between 1 and 40."""
	scaler = StandardScaler()
	scaler.fit(df.drop(target_col_name, axis=1))
	scaled_features = scaler.transform(df.drop(target_col_name, axis=1))

	# Split the data into train and test data
	X_train, X_test, y_train, y_test = train_test_split(scaled_features, df[target_col_name], test_size=0.30)

	error_rate = []

	# Loop through k-values between 1 and 40 and append the error rate to a list
	for i in range(1, 40):
		knn = KNeighborsClassifier(n_neighbors=i)
		knn.fit(X_train, y_train)
		pred_i = knn.predict(X_test)
		error_rate.append(np.mean(pred_i != y_test))

	# Plot the error rate against the k-value to visually determine which k-value to choose
	plt.scatter(x=list(range(1, 40)), y=error_rate)
	plt.title('Error Rate vs. K Value')
	plt.xlabel('K')
	plt.ylabel('Error Rate')
	plt.show()

	# Choose a k-value and then run reports to determine accuracy of model
	k_val = int(input('Enter a k-value to run confusion matrix & classification report: '))
	knn = KNeighborsClassifier(n_neighbors=k_val)
	knn.fit(X_train, y_train)
	predictions = knn.predict(X_test)
	print(confusion_matrix(y_test, predictions))
	print(classification_report(y_test, predictions))
