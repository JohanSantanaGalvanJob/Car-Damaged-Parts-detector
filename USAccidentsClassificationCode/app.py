import numpy as np
import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Path del modelo preentrenado
MODEL_PATH = 'models/accidents_model.pkl'
DATASET_LOCALIDAD = '../USAccidentsDatasetForDeploying.csv'
df = pd.read_csv(DATASET_LOCALIDAD)
print(df.shape)
df = df.dropna(axis=0)
array_state = pd.factorize(df['State'])[1]
array_county= pd.factorize(df['County'])[1]
array_city = pd.factorize(df['City'])[1]
array_street= pd.factorize(df['Street'])[1]
array_wind=pd.factorize(df['Wind_Direction'])[1]
array_weather=pd.factorize(df['Weather_Condition'])[1]

# Creamos diccionarios vacíos para almacenar los mapeos
map_state = {}
map_county = {}
map_city = {}
map_street = {}
map_wind = {}
map_weather = {}

# Iteramos sobre los arrays y asignamos un número único a cada valor
for i, state in enumerate(array_state):
    map_state[state] = i

for i, county in enumerate(array_county):
    map_county[county] = i

for i, city in enumerate(array_city):
    map_city[city] = i

for i, street in enumerate(array_street):
    map_street[street] = i

for i, wind in enumerate(array_wind):
    map_wind[wind] = i

for i, weather in enumerate(array_weather):
    map_weather[weather] = i

# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1, -1)
    preds = model.predict(x)
    return preds

def transform_predict(predict):
    if predict == 0:
        return "1: Muy Leve."
    elif predict == 1:
        return "2: Leve."
    elif predict == 2:
        return "3: Moderado."
    elif predict == 3:
        return "4: Grave."
    else:
        return "???"

def main():
    model = ''

    # Se carga el modelo
    if model == '':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)

    # Título
    html_temp = """
    <h1 style="color:#00000;text-align:center;">PREDICCIÓN DE SEVERIDAD DE ACCIDENTES EEUU</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    # Dividir en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        Distance = st.number_input("Distancia (mi):", key='distance')
        State = st.selectbox("Estado:", array_state, key='state').upper()
        County = st.selectbox("Condado:",array_county, key='county')
        City = st.text_input("Ciudad:",key='city')
        Street = st.text_input("Calle:",key='street')
        Wind_Direction = st.selectbox("Dirección del Viento:", array_wind, key='wind_direction')
        Junction = st.selectbox("Intersección:", ["No", "Sí"], key='junction')

    with col2:
        Temperature = st.number_input("Temperatura (F):", key='temperature')
        Pressure = st.number_input("Presión (in):", key='pressure')
        Weather_Condition = st.selectbox("Condición Climática:", array_weather, key='weather_condition')
        Crossing = st.selectbox("Cruzando:", ["No", "Sí"], key='crossing')
        Stop = st.selectbox("Parada:", ["No", "Sí"], key='stop')
        Traffic_Signal = st.selectbox("Semáforo:", ["No", "Sí"], key='traffic_signal')

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"):
        x_in = [
            float(st.session_state['distance']),
            map_state[st.session_state['state'].upper()],
            map_county[st.session_state['county'].title()],
            st.session_state['city'].title(),
            st.session_state['street'].title(),
            float(st.session_state['temperature']),
            float(st.session_state['pressure']),
            map_wind[st.session_state['wind_direction'].title()],
            map_weather[st.session_state['weather_condition'].title()],
            int(st.session_state['crossing'] == 'Sí'),
            int(st.session_state['junction'] == 'Sí'),
            int(st.session_state['stop'] == 'Sí'),
            int(st.session_state['traffic_signal'] == 'Sí')
        ]
        predictS = model_prediction(x_in, model)
        st.success('El nivel de severidad medido en el accidente es: {}'.format(transform_predict(predictS[0])).upper())

if __name__ == '__main__':
    main()
