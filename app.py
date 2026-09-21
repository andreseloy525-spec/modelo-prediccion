"""
API REST para el Modelo de Calidad de Agua
Este archivo es el que se subirá a Render para servir el modelo (.pkl) a internet.
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado 
try:
    modelo = joblib.load('modelo_calidad_agua.pkl')
    print("Modelo cargado exitosamente.")
except Exception as e:
    modelo = None
    print(f"Error al cargar el modelo: {e}")

# Esta es la ruta externa
@app.route('/api/predecir', methods=['POST'])
def predecir_calidad():
    if not modelo:
        return jsonify({"error": "El modelo no está disponible en el servidor"}), 500

    try:
        # 1. Recibir los datos en formato JSON desde la plataforma web
        datos_entrada = request.get_json()
        
        # 2. Extraer el pH y Amonio
        ph = float(datos_entrada['pH'])
        amonio = float(datos_entrada['Amonio'])
        
        # 3. Empaquetarlos para la Inteligencia Artificial
        nueva_lectura = pd.DataFrame({'pH': [ph], 'Amonio': [amonio]})
        
        # 4. Obtener la predicción
        alerta = modelo.predict(nueva_lectura)[0]
        
        # 5. Devolver la respuesta a la plataforma web
        return jsonify({
            "status": "success",
            "pH_recibido": ph,
            "Amonio_recibido": amonio,
            "prediccion": alerta
        }), 200

    except Exception as e:
        return jsonify({"error": f"Datos inválidos o faltantes. Detalle: {str(e)}"}), 400

# Ruta de prueba para saber si la API está encendida en Render
@app.route('/', methods=['GET'])
def index():
    return jsonify({"mensaje": "API de Predicción Acuícola GAM en línea."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)