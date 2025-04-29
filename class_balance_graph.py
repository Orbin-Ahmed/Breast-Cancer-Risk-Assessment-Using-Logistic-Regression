import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('Cancer_Train_Preprocessed.csv')

y_train = train_data['diagnosis']
class_counts = y_train.value_counts()

plt.figure(figsize=(8,6))
class_counts.plot(kind='bar')

plt.title('Class Distribution After Applying SMOTE')
plt.xlabel('Class (0 = Benign, 1 = Malignant)')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)

for index, value in enumerate(class_counts):
    plt.text(index, value + 5, str(value), ha='center', fontsize=12)

plt.tight_layout()
plt.show()