import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/employee_attrition.csv")


# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

df = df.drop([
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
], axis=1)


# ==========================================
# 3. CONVERT TARGET
# ==========================================

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# ==========================================
# 5. IDENTIFY CATEGORICAL COLUMNS
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns


print("Categorical columns:")
print(categorical_columns.tolist())

print("\nNumerical columns:")
print(numerical_columns.tolist())


# ==========================================
# 6. CREATE PREPROCESSOR
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================
# 8. LOGISTIC REGRESSION
# ==========================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# Train model
logistic_model.fit(X_train, y_train)


# Predictions
y_pred = logistic_model.predict(X_test)
y_prob = logistic_model.predict_proba(X_test)[:, 1]


# Evaluation
print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_prob), 4)
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 9. RANDOM FOREST
# ==========================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                max_depth=None
            )
        )
    ]
)


# Train model
random_forest_model.fit(X_train, y_train)


# Predictions
rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]


# Evaluation
print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, rf_pred), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, rf_prob), 4)
)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
from xgboost import XGBClassifier


# ==========================================
# 10. XGBOOST
# ==========================================

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        )
    ]
)


# Train model
xgb_model.fit(X_train, y_train)


# Predictions
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]


# Evaluation
print("\n==============================")
print("XGBOOST")
print("==============================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, xgb_pred), 4)
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, xgb_prob), 4)
)

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))
# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, xgb_prob)
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results.round(4))
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["ROC-AUC"]
)

plt.title("Model Comparison - ROC-AUC")
plt.xlabel("Model")
plt.ylabel("ROC-AUC")

plt.ylim(0, 1)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()
# ==========================================
# 12. XGBOOST FEATURE IMPORTANCE
# ==========================================

# Get the fitted preprocessor
fitted_preprocessor = xgb_model.named_steps["preprocessor"]

# Get feature names after one-hot encoding
feature_names = fitted_preprocessor.get_feature_names_out()

# Get XGBoost classifier
xgb_classifier = xgb_model.named_steps["classifier"]

# Get feature importance
importance = xgb_classifier.feature_importances_

# Create dataframe
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n==============================")
print("TOP 15 IMPORTANT FEATURES")
print("==============================")

print(feature_importance.head(15))
# ==========================================
# 13. FEATURE IMPORTANCE GRAPH
# ==========================================

top_features = feature_importance.head(10)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("Top 10 Features Influencing Employee Attrition")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 14. DETAILED MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, xgb_pred)
    ],

    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, rf_pred),
        precision_score(y_test, xgb_pred)
    ],

    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, rf_pred),
        recall_score(y_test, xgb_pred)
    ],

    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, rf_pred),
        f1_score(y_test, xgb_pred)
    ],

    "ROC-AUC": [
        roc_auc_score(y_test, y_prob),
        roc_auc_score(y_test, rf_prob),
        roc_auc_score(y_test, xgb_prob)
    ]
})

print("\n==============================")
print("DETAILED MODEL COMPARISON")
print("==============================")

print(comparison.round(4))
import joblib

# Save XGBoost model
joblib.dump(
    xgb_model,
    "employee_attrition_xgboost.pkl"
)

print("\nXGBoost model saved successfully!")