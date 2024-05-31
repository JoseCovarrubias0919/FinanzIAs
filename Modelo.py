import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

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
model.add(Dense(32, input_dim=2, activation='relu'))  # Capa de entrada con 2 nodos (gasto e ingreso)
model.add(Dense(16, activation='relu'))  # Capa oculta
model.add(Dense(3, activation='softmax'))  # Capa de salida con 3 nodos (Crítico, Alerta, Aceptable)

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

# Guardar el modelo entrenado en un archivo
model.save('modeloE.h5')