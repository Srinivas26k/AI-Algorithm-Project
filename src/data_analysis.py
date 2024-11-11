import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_data = pd.read_csv('../data/raw_data/train.csv')
test_data = pd.read_csv('../data/raw_data/test.csv')

# Display the first few rows of the training data
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum())

# Visualize the distribution of the target variable 'Survived'
sns.countplot(x='Survived', data=train_data)
plt.title('Survival Distribution')
plt.show()

# Visualize distribution of passenger class
sns.countplot(x='Pclass', data=train_data)
plt.title('Passenger Class Distribution')
plt.show()

# Correlation heatmap
corr_matrix = train_data.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()
