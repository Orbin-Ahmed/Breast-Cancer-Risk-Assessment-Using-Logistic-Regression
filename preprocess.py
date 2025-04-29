import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Cancer_Data.csv')

data_clean = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

X = data_clean.drop(columns=['diagnosis'])
y = data_clean['diagnosis']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

train_data = pd.DataFrame(X_train, columns=X.columns)
train_data['diagnosis'] = y_train
train_data.to_csv('Cancer_Train_Preprocessed.csv', index=False)

val_data = pd.DataFrame(X_val, columns=X.columns)
val_data['diagnosis'] = y_val
val_data.to_csv('Cancer_Validation_Preprocessed.csv', index=False)

test_data = pd.DataFrame(X_test, columns=X.columns)
test_data['diagnosis'] = y_test
train_data.to_csv('Cancer_Test_Preprocessed.csv', index=False)

print("Preprocessing completed! Ready for model training.")
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")
