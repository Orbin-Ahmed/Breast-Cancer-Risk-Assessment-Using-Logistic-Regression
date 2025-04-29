import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Stony Brook/520/Cancer_Data.csv")

sns.countplot(x='diagnosis', data=df, palette='Set2')
plt.title('Diagnosis Distribution (Malignant vs. Benign)')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.savefig('diagnosis_distribution.png')
plt.close()

diagnosis_counts = df['diagnosis'].value_counts()
labels = diagnosis_counts.index
sizes = diagnosis_counts.values
colors = ['#ff9999','#66b3ff']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.axis('equal')
plt.title('Diagnosis Proportion')
plt.savefig('diagnosis_pie_chart.png')
plt.close()

df = df.drop(['id'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

corr_matrix = df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

top_corr_features = corr_matrix['diagnosis'].drop('diagnosis').sort_values(ascending=False)

plt.figure(figsize=(14, 6))
top_corr_features.head(10).plot(kind='barh', color='teal')
plt.title('Top 10 Features Positively Correlated with Malignant Diagnosis')
plt.xlabel('Correlation Coefficient')
plt.gca().invert_yaxis()
plt.savefig('top_correlated_features.png')
plt.close()
