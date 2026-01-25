import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def evaluate_model(model, X_test, y_test, feature_names, model_name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 {model_name} Performance:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # Feature Importance Plot
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df)
    plt.title(f"{model_name} - Feature Importance")
    plt.tight_layout()
    plt.savefig(f"results/{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.close()

    return acc, prec, rec, f1


# ==================== DIABETES MODEL ====================
df_diabetes = pd.read_csv("data/diabetes.csv")
X_d = df_diabetes.drop("Outcome", axis=1)
y_d = df_diabetes["Outcome"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

diabetes_model = RandomForestClassifier(random_state=42)
diabetes_model.fit(X_train_d, y_train_d)

evaluate_model(diabetes_model, X_test_d, y_test_d, X_d.columns, "Diabetes Model")

with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump(diabetes_model, f)


# ==================== HEART DISEASE MODEL ====================
df_heart = pd.read_csv("data/heart.csv")
X_h = df_heart.drop("target", axis=1)
y_h = df_heart["target"]

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train_h, y_train_h)

evaluate_model(heart_model, X_test_h, y_test_h, X_h.columns, "Heart Disease Model")

with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(heart_model, f)


# ==================== BRAIN STROKE MODEL ====================
df_brain = pd.read_csv("data/brain_stroke.csv")
df_brain.columns = df_brain.columns.str.strip()

le = LabelEncoder()
for col in df_brain.select_dtypes(include="object").columns:
    df_brain[col] = le.fit_transform(df_brain[col])

df_brain = df_brain.fillna(0)

X_bs = df_brain.drop("stroke", axis=1)
y_bs = df_brain["stroke"]

X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(X_bs, y_bs, test_size=0.2, random_state=42)

brain_model = RandomForestClassifier(random_state=42)
brain_model.fit(X_train_bs, y_train_bs)

evaluate_model(brain_model, X_test_bs, y_test_bs, X_bs.columns, "Brain Stroke Model")

with open("models/brainstroke_model.pkl", "wb") as f:
    pickle.dump(brain_model, f)
