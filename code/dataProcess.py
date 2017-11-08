import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def loadData(trainPath, testPath):
	train = pd.read_csv(trainPath)
	test = pd.read_csv(testPath)

	# 将PassengerId, Name, Ticket扔掉，这些没有意义
	titanic_df = train.drop(['PassengerId','Name','Ticket'], axis=1)
	test_df = test.drop(['PassengerId','Name','Ticket'],axis=1)

	# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
	# So, we can classify passengers as males, females, and child
	titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
	test_df['Person'] = test_df[['Age','Sex']].apply(get_person,axis=1)
	# No need to use Sex column since we created Person column
	titanic_df.drop(['Sex'],axis=1,inplace=True)
	test_df.drop(['Sex'],axis=1,inplace=True)

	# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
	person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
	person_dummies_titanic.columns = ['Child','Female','Male']
	person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

	person_dummies_test  = pd.get_dummies(test_df['Person'])
	person_dummies_test.columns = ['Child','Female','Male']
	person_dummies_test.drop(['Male'], axis=1, inplace=True)

	titanic_df = titanic_df.join(person_dummies_titanic)
	test_df    = test_df.join(person_dummies_test)

	titanic_df.drop(['Person'],axis=1,inplace=True)
	test_df.drop(['Person'],axis=1,inplace=True)


	# get average, std, and number of NaN values in titanic_df
	average_age_titanic   = titanic_df["Age"].mean()
	std_age_titanic       = titanic_df["Age"].std()
	count_nan_age_titanic = titanic_df["Age"].isnull().sum()

	# get average, std, and number of NaN values in test_df
	average_age_test   = test_df["Age"].mean()
	std_age_test       = test_df["Age"].std()
	count_nan_age_test = test_df["Age"].isnull().sum()

	# generate random numbers between (mean - std) & (mean + std)
	rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
	rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

	# fill NaN values in Age column with random values generated
	titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
	test_df["Age"][np.isnan(test_df["Age"])] = rand_2

	# convert from float to int
	titanic_df['Age'] = titanic_df['Age'].astype(int)
	test_df['Age']    = test_df['Age'].astype(int)

	# Instead of having two columns Parch & SibSp, 
	# we can have only one column represent if the passenger had any family member aboard or not,
	# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
	titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
	titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
	titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

	test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
	test_df['Family'].loc[test_df['Family'] > 0] = 1
	test_df['Family'].loc[test_df['Family'] == 0] = 0

	# drop Parch & SibSp
	titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
	test_df    = test_df.drop(['SibSp','Parch'], axis=1)

	# only for test_df, since there is a missing "Fare" values
	test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

	# convert from float to int
	titanic_df['Fare'] = titanic_df['Fare'].astype(int)
	test_df['Fare']    = test_df['Fare'].astype(int)

	#Cabin
	titanic_df['Has_Cabin'] = titanic_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
	test_df['Has_Cabin'] = test_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
	titanic_df = titanic_df.drop('Cabin',axis=1)
	test_df = test_df.drop('Cabin',axis=1)

	# Mapping Embarked
	# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
	titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
	titanic_df['Embarked'] = titanic_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
	test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

	x_train = titanic_df.drop('Survived',axis=1).values
	y_train = titanic_df['Survived'].values
	x_test = test_df.values

	return x_train,y_train,x_test



def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

if __name__ == '__main__':
	trainPath = '../data/train.csv'
	testPath = '../data/test.csv'
	x_train, y_train,x_test = loadData(trainPath,testPath)
	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
