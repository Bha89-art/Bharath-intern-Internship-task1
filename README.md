import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Loading the Titanic dataset
titanic_info = pd.read_csv('Titanic-Dataset.csv')
print(titanic_info.head())
# general information
print("Dataset info:")
print(titanic_info.info())
# Checking for missing values in each column

missing_data = titanic_info.isnull().sum()
print(f"Missing values in each column:\n{missing_data}")
# Visualizing the distribution of the target variable named 'Survived'
plt.figure(figsize=(4, 3))
sns.countplot(x='Survived', data=titanic_info)
plt.title('Survival Distribution')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
# plotting histogram for numerical features
titanic_info.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()
# Handling missing values
# For 'Age', i wilbe using the median value

titanic_info['Age'].fillna(titanic_info['Age'].median(), inplace=True)
# For 'Embarked', i will use the most common value

titanic_info['Embarked'].fillna(titanic_info['Embarked'].mode()[0], inplace=True)
#Check and drop 'Cabin' if it exists

if 'Cabin' in titanic_info.columns:
    print("Dropping 'Cabin' column.")
    titanic_info.drop(columns='Cabin', inplace=True)
else:
    print("'Cabin' column not found in the dataset.")



# Converting categorical variables to numerical using LabelEncoder

le = LabelEncoder()
titanic_info['Sex'] = le.fit_transform(titanic_info['Sex'])
titanic_info['Embarked'] = le.fit_transform(titanic_info['Embarked'])


print("Data after preprocessing:")
print(titanic_info.head())

# Feature Engineering
# Adding a new feature 'FamilySize'

titanic_info['FamilySize'] = titanic_info['SibSp'] + titanic_info['Parch'] + 1

# Create a new feature 'IsAlone'
# If there's more than 1 in FamilySize, they're not alone
titanic_info['IsAlone'] = 1  # Assuming everyone is alone
titanic_info.loc[titanic_info['FamilySize'] > 1, 'IsAlone'] = 0  # Correct assumption for those not alone
# Dropping columns that won't be used for prediction
# Dropping 'Name', 'Ticket', 'SibSp', 'Parch' - they won't help my model
titanic_info.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch'], inplace=True)

print("Data after feature engineering:")
print(titanic_info.head())
# Model Building
# Separating features and target variable

features = titanic_info.drop(columns='Survived')
target = titanic_info['Survived']

# Splitting the data into training and testing sets
# Time to train-test split! Always good practice to set a random state.
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initializing the RandomForestClassifier
# Going with a trusty Random Forest here
rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfc_model.fit(x_train, y_train)
# Predictions on the test set

y_pred = rfc_model.predict(x_test)

# Evaluating the model
# Moment of truth - let's check the accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
co_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{co_matrix}')
print(f'Classification Report:\n{report}')
