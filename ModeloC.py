import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant
import numpy as np

# Definir el nombre del archivo CSV y cargar los datos
file_path = 'datosEGS.csv'
df = pd.read_csv(file_path)

# Manejar valores faltantes
df.fillna(df.mean(), inplace=True)

# Preprocesar los datos
label_encoder = LabelEncoder()
df['Nivel de Riesgo'] = label_encoder.fit_transform(df['Nivel de Riesgo'])
X = df[['Gasto', 'Ingreso']]
y = df['Nivel de Riesgo']

# Normalización de datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo de red neuronal
model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu', bias_initializer=Constant(value=0.1)))  # Capa de entrada con 2 nodos (gasto e ingreso)
model.add(Dense(16, activation='relu'))  # Capa oculta
model.add(Dense(3, activation='softmax'))  # Capa de salida con 3 nodos (Crítico, Alerta, Aceptable)

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=3, batch_size=16, validation_split=0.2)

# Evaluar el modelo

# Precisión (Accuracy)
# Obtener las predicciones como probabilidades
y_pred_prob = model.predict(X_test)

# Convertir las probabilidades a clases
y_pred = [np.argmax(pred) for pred in y_pred_prob]
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

# Precisión, Sensibilidad y Especificidad
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
specificity = recall_score(y_test, y_pred, labels=[0, 1, 2], average=None)
print(f"Precisión: {precision}, Sensibilidad: {recall}, Especificidad: {specificity}")

# Obtener las predicciones como probabilidades
y_pred_prob = model.predict(X_test)

# Convertir las etiquetas multiclase en binarias
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Calcular la curva ROC y el AUC-ROC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):  # 3 clases en este ejemplo
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Calcular el promedio del AUC-ROC
mean_roc_auc = sum(roc_auc.values()) / len(roc_auc)
print(f"Área bajo la curva ROC (AUC-ROC) promedio: {mean_roc_auc}")