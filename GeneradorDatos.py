import pandas as pd
import random

# Función para generar datos de prueba
def generate_data(num_samples):
    data = {'Gasto': [], 'Ingreso': [], 'Nivel de Riesgo': []}
    for _ in range(num_samples):
        ingreso = random.randint(1000, 5000)  # Ingreso aleatorio entre 1000 y 5000
        gasto = random.randint(0, 1800)  # Gasto aleatorio entre 200 y 4000
        porcentaje_gasto = (gasto / ingreso) * 100
        if porcentaje_gasto > 35:
            nivel_riesgo = 'Crítico'
        elif porcentaje_gasto > 20:
            nivel_riesgo = 'Alerta'
        else:
            nivel_riesgo = 'Aceptable'
        data['Gasto'].append(gasto)
        data['Ingreso'].append(ingreso)
        data['Nivel de Riesgo'].append(nivel_riesgo)
    return pd.DataFrame(data)

# Generar datos de prueba
num_samples = 100000  # Número de muestras de datos de prueba
df_test = generate_data(num_samples)

# Guardar los datos de prueba en un archivo CSV
df_test.to_csv('datosEG.csv', index=False)