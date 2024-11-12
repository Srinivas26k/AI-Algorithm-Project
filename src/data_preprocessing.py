import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data (Update these paths)
train_data = pd.read_csv(r'E:\AI-Algorithm-Project\data\raw_data\train.csv')
test_data = pd.read_csv(r'E:\AI-Algorithm-Project\data\raw_data\test.csv')

# Basic data cleaning
train_data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)  # Drop columns with too many missing values
train_data.dropna(subset=['Embarked'], inplace=True)        # Drop rows with missing Embarked

# Separate target from features
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Preprocessing pipeline for numeric features
numeric_features = ['Age', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Convert processed data to a DataFrame
processed_columns = preprocessor.get_feature_names_out()
X_processed_df = pd.DataFrame(X_processed, columns=processed_columns)

# Save processed data (Update this path too)
X_processed_df['Survived'] = y.reset_index(drop=True)
X_processed_df.to_csv(r'E:\AI-Algorithm-Project\data\processed_data\train_cleaned.csv', index=False)
