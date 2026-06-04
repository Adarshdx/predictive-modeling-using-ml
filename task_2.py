# Predictive Modeling Using Machine Learning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import joblib


data = {
    "Age": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61,
            18, 21, 24, 28, 30, 35, 40, 42, 48, 50],

    "Salary": [25000, 30000, 80000, 90000, 75000,
               100000, 110000, 120000, 130000, 125000,
               20000, 24000, 28000, 35000, 40000,
               50000, 65000, 70000, 85000, 95000],

    "Purchased": [0, 0, 1, 1, 1,
                  1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())



X = df[["Age", "Salary"]]
y = df["Purchased"]


imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()


y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()



joblib.dump(model, "predictive_model.pkl")

print("\nModel Saved Successfully!")
print("Files Generated:")
print("1. confusion_matrix.png")
print("2. roc_curve.png")
print("3. predictive_model.pkl")