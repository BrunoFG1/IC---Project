import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Rui -> Estratégia RN + CE (Redes Neuronais + Computação Evolucionária)

VISION = 3
OBS_SHAPE = (2 * VISION + 1, 2 * VISION + 1, 1)
NUM_ACTIONS = 7

def convNN():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=OBS_SHAPE),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(NUM_ACTIONS, activation='linear')
    ])
    return model


def policy(obs, agent_id):
    obs_array = np.array(obs) / 255.0  #normalizar os dados para a rede funcionar como deve ser
    return random.randint(0, 6)


