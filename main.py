import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

data = pd.read_csv("Data.csv")

# *checking for missing values
print(data.isnull().sum())

# *Descriptive Statistics
print(data.describe())

numeric_cols = [
    "HomePage",
    "HomePage_Duration",
    "LandingPage",
    "LandingPage_Duration",
    "ProductDescriptionPage",
    "ProductDescriptionPage_Duration",
    "GoogleMetric:Bounce_Rates",
    "GoogleMetric:Exit_Rates",
    "GoogleMetric:Page_Values",
]

# *histogram
plt.figure(figsize=(16, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# *EDA
categorical_cols = [
    "SeasonalPurchase",
    "Month_SeasonalPurchase",
    "OS",
    "SearchEngine",
    "Zone",
    "VisitorType",
    "Gender",
    "Cookies",
    "Education",
    "Marital_Status",
    "WeekendPurchase",
    "Made_Purchase",
]


plt.figure(figsize=(16, 20))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(4, 3, i)
    sns.countplot(data[col])
    plt.title(col)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# *Countplots

plt.figure(figsize=(16, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=data["Made_Purchase"], y=data[col])
    plt.title(col)
plt.tight_layout()
plt.show()
# *Boxplots

plt.figure(figsize=(10, 8))
sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
# *heatmap

X = data.drop("Made_Purchase", axis=1)
y = data["Made_Purchase"]
# *Splitting data into features and target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# *train-test split

X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)
# *performing one-hot encoding on categorical variables

missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]
# *Ensuring that the columns in the training and testing datasets match after one-hot encoding

imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(X_train_encoded)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_imputed, y_train)
# *train Logistic Regression model

y_pred = model.predict(X_test_encoded)
# *making predictions
# TODO having a few issues with NaN values and equalizing values

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# *evaluating the model

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

y_prob = model.predict_proba(X_test_encoded)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
