def feature_engineering(df):
    # Binning 'Age' into categories
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

    # Log-transform 'Fare'
    df['Fare_log'] = np.log1p(df['Fare'])

    # Interaction term: Sex & Pclass
    df['Sex_Pclass_interaction'] = df['Sex'] * df['Pclass']

    # Return transformed DataFrame
    return df
