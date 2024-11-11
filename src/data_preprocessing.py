from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

# Load the training dataset
train_data = pd.read_csv('../data/raw_data/train.csv')

# Fill missing values for 'Age' with the mean
imputer = SimpleImputer(strategy='mean')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])

# Fill missing values for 'Embarked' with the most frequent value
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Encode categorical features using LabelEncoder
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])

# Drop unnecessary columns
train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Split the data into features and target
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed data
train_data.to_csv('../data/processed_data/train_cleaned.csv', index=False)
