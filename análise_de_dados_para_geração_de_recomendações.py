Análise de Dados para Geração de Recomendações

Original file is located at https://colab.research.google.com/drive/13-ALKzCs4HxZxv6KfO-Xsy304js0MeY1


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/content/online_shoppers_intention.csv')

print(data.head())

print(data.info())

print(data.describe())

data_encoded = pd.get_dummies(data, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)

plt.figure(figsize=(10,8))
sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação após codificação')
plt.show()

corr_matrix = data_encoded.corr()

print(corr_matrix)

print("\nCorrelação com a variável 'Revenue':\n")
print(corr_matrix['Revenue'].sort_values(ascending=False))

print(data_encoded.isnull().sum())

data_encoded = data_encoded.dropna()

print("Após remoção de valores ausentes:")
print(data_encoded.isnull().sum())

numeric_cols = ['Administrative', 'Informational', 'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

Q1 = data_encoded[numeric_cols].quantile(0.25)
Q3 = data_encoded[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

data_cleaned = data_encoded[~((data_encoded[numeric_cols] < (Q1 - 1.5 * IQR)) | (data_encoded[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Tamanho original: {data_encoded.shape}, Tamanho após remoção de outliers: {data_cleaned.shape}")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data_cleaned[numeric_cols] = scaler.fit_transform(data_cleaned[numeric_cols])

print(data_cleaned[numeric_cols].head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = data_cleaned.drop('Revenue', axis=1)
y = data_cleaned['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Acurácia: ", accuracy_score(y_test, y_pred))
print("Relatório de Classificação: \n", classification_report(y_test, y_pred))

import joblib

joblib.dump(rf, 'modelo_random_forest.pkl')

print("Modelo exportado com sucesso!")

modelo_carregado = joblib.load('modelo_random_forest.pkl')

y_pred_novo = modelo_carregado.predict(X_test)

print("Acurácia do modelo carregado: ", accuracy_score(y_test, y_pred_novo))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X = data_cleaned.drop('Revenue', axis=1)
y = data_cleaned['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)

y_pred_weighted = rf_weighted.predict(X_test)

print("Acurácia (ajustada): ", accuracy_score(y_test, y_pred_weighted))
print("Relatório de Classificação (ajustado): \n", classification_report(y_test, y_pred_weighted))