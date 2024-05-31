import pandas as pd
import numpy as np
from keras.models import load_model

# Cargar el archivo CSV con las categorías y montos de gastos
df = pd.read_csv('datos.csv')

# Cargar los modelos entrenados
modelo_primarias = load_model('modeloEGP.h5')
modelo_secundarias = load_model('modeloEGS.h5')

# Definir las categorías primarias y secundarias
categorias_primarias = ['Escuela', 'Salud', 'Comida']
categorias_secundarias = [c for c in df.columns if c not in ['Ingreso'] + categorias_primarias]

# Función para asignar niveles de riesgo basados en las predicciones numéricas
def asignar_nivel(Aceptable, Alerta, Critico):
    if Critico == 1:
        return "Crítico"
    elif Alerta == 1:
        return "Alerta"
    elif Aceptable == 1:
        return "Aceptable"

# Iterar sobre las filas del DataFrame y hacer predicciones
for index, row in df.iterrows():
    ingreso = row['Ingreso']

    # Predecir con el modelo de categorías primarias
    predicciones_primarias = []
    for categoria in categorias_primarias:
        gastos = row[categoria]
        entrada = np.array([[gastos, ingreso]])
        pred = modelo_primarias.predict(entrada)
        nivel = asignar_nivel(pred[0][0], pred[0][1], pred[0][2])  # Umbral de riesgo para primarias
        predicciones_primarias.append((categoria, nivel))

    # Predecir con el modelo de categorías secundarias
    predicciones_secundarias = []
    for categoria in categorias_secundarias:
        gastos = row[categoria]
        entrada = np.array([[gastos, ingreso]])
        pred = modelo_secundarias.predict(entrada)
        nivel = asignar_nivel(pred[0][0], pred[0][1], pred[0][2])  # Umbral de riesgo para secundarias
        predicciones_secundarias.append((categoria, nivel))

    # Imprimir las predicciones para cada categoría
    print(f'Predicciones para fila {index}:')
    print('Categorías Primarias:')
    for categoria, nivel in predicciones_primarias:
        print(f'{categoria}: {nivel}')
    print('Categorías Secundarias:')
    for categoria, nivel in predicciones_secundarias:
        print(f'{categoria}: {nivel}')
    print('\n')
