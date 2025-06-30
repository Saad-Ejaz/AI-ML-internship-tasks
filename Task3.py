# 📦 Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 📥 Load data from processed.cleveland.data
column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv('processed.cleveland.data', names=column_names)

# 🧹 Clean data: handle '?', convert types, drop missing
df.replace('?', pd.NA, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# 🎯 Convert multi-class target to binary (0 = no disease, 1 = disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 🔍 EDA: visualize correlation
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 📊 Target class distribution
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution')
plt.show()

# 🧪 Train-test split
X = df.drop('target', axis=1)
y = df['target']

# 🔄 Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# 🤖 Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

# 🌲 Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

# 📈 Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    print(f"{name} Accuracy: {acc:.2f}")
    print(f"{name} ROC-AUC: {auc:.2f}")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 📊 Evaluate models
evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)
evaluate_model("Decision Tree", y_test, y_pred_tree, y_prob_tree)

# 📉 ROC Curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)

plt.plot(fpr_log, tpr_log, label=f'Logistic (AUC={roc_auc_score(y_test, y_prob_log):.2f})')
plt.plot(fpr_tree, tpr_tree, label=f'Tree (AUC={roc_auc_score(y_test, y_prob_tree):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

# 📌 Feature Importance from Decision Tree
importance = pd.Series(tree_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
sns.barplot(x=importance, y=importance.index)
plt.title('Top Features (Decision Tree)')
plt.show()