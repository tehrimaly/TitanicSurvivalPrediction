import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# 1. LOAD DATASET
# ------------------------
df = pd.read_csv("train.csv")

# ------------------------
# 2. CLEAN DATA
# ------------------------

# Fill missing Age with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin (too many missing values)
df.drop(columns=["Cabin"], inplace=True)

# ------------------------
# 3. ENCODE CATEGORICAL DATA
# ------------------------
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ------------------------
# 4. SELECT FEATURES
# ------------------------
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

# ------------------------
# 5. SPLIT DATA
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ------------------------
# 6. TRAIN MODEL
# ------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------
# 7. PREDICT
# ------------------------
y_pred = model.predict(X_test)

# ------------------------
# 8. EVALUATION
# ------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))