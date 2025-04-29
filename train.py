import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load data 
train_data = pd.read_csv('data/Cancer_Train_Preprocessed.csv')
val_data = pd.read_csv('data/Cancer_Validation_Preprocessed.csv')
test_data = pd.read_csv('data/Cancer_Test_Preprocessed.csv')

# Seperate target from test, train and validation datasets
X_train = train_data.drop(columns=['diagnosis'])
y_train = train_data['diagnosis']

X_val = val_data.drop(columns=['diagnosis'])
y_val = val_data['diagnosis']

X_test = test_data.drop(columns=['diagnosis'])
y_test = test_data['diagnosis']



# LogisticRegression train
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

# validation on LogisticRegression 
y_val_pred_logreg = logreg.predict(X_val)
# testing on LogisticRegression  
y_test_pred_logreg = logreg.predict(X_test)



# RandomForest train 
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Validation on RandomForest
y_val_pred_rf = rf.predict(X_val)
# testing on RandomForest
y_test_pred_rf = rf.predict(X_test)



# Evaluation 
def evaluate_model(name, y_true, y_pred, filename_prefix):
    print(f"\n{name} Evaluation Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_confusion_matrix.png")
    plt.close()



# Comparison
evaluate_model("Logistic Regression (Validation)", y_val, y_val_pred_logreg, "logreg_val")
evaluate_model("Random Forest (Validation)", y_val, y_val_pred_rf, "rf_val")

evaluate_model("Logistic Regression (Test)", y_test, y_test_pred_logreg, "logreg_test")
evaluate_model("Random Forest (Test)", y_test, y_test_pred_rf, "rf_test")

model_names = ['Logistic Regression', 'Random Forest']
test_accuracies = [accuracy_score(y_test, y_test_pred_logreg), accuracy_score(y_test, y_test_pred_rf)]

plt.figure(figsize=(8,6))
sns.barplot(x=model_names, y=test_accuracies)
plt.title("Model Test Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.05)
for i, acc in enumerate(test_accuracies):
    plt.text(i, acc + 0.005, f"{acc:.2f}", ha='center')
plt.tight_layout()
plt.savefig("model_test_accuracy_comparison.png")
plt.close()


joblib.dump(logreg, 'logistic_regression_model.pkl')
joblib.dump(rf, 'random_forest_model.pkl')
