import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def convertAge(age):
	#split number and unit
	age_info = age.split()
	mult = 1
	#units are week/month/year -> converting to weeks
	if(age_info[0] == 'm'):
		mult = 4
	elif(age_info[0] == 'y'):
		mult = 52
	return int(age_info[0]) * mult


def preprocessing():
	features = ['OutcomeType', 'AnimalType', 'SexuponOutcome', 'Breed', 'Color', 'AgeuponOutcome']
	outcomes = ['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer']
	try:
		training_data = pd.read_csv('train.csv', usecols = features)
		#features[0] is the label and therefore not in testing_data
		testing_data = pd.read_csv('test.csv', usecols = features[1:])
	except OSError:
		print("Cannot find training or testing file")
	

	#drop any row with missing features
	training_data = training_data.dropna(axis = 0, how='any')
	testing_data = testing_data.dropna(axis = 0, how= 'any')
	#get labels from training data and drop from dataframe
	training_answers = training_data['OutcomeType']
	training_data = training_data.drop('OutcomeType', 1)

	#convert labels into integers
	training_answers = training_answers.apply(lambda x : outcomes.index(x))

	#convert AgeuponOutcome into integer of number of weeks old
	training_data['AgeuponOutcome'] = training_data['AgeuponOutcome'].apply(lambda x: convertAge(x))
	testing_data['AgeuponOutcome'] = testing_data['AgeuponOutcome'].apply(lambda x : convertAge(x))

	#convert categorical features (all except age) into binary features
	binary_training_data = pd.get_dummies(training_data)
	binary_testing_data = pd.get_dummies(testing_data)

	#save data for later use
	np.save('train_animal_data.npy', binary_training_data)
	np.save('test_animal_data.npy', binary_testing_data)
	np.save('train_answers.npy', training_answers)
	

if __name__ == '__main__':
	preprocessing()