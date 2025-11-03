import random
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Concatenate

# Rui -> Estratégia RN + CE (Redes Neuronais + Computação Evolucionária)

VISION = 3
OBS_SHAPE = (2 * VISION + 1, 2 * VISION + 1, 1)
NUM_ACTIONS = 7
STUN_TIME = 10.0

def modelNN():
    
    """Neural network with multiple inputs (in this case 3 inputs, for the obs, holding and stun)"""
    
    obs_input = Input(shape=OBS_SHAPE, name='obs_input')
    x_vis = Conv2D(8, (3,3), activation='relu')(obs_input)
    x_vis = Flatten()(x_vis) 
    
    hold_input = Input(shape=(1,), name='hold_input')
    x_hold = Dense(4, activation='relu')(hold_input)
    
    stun_input = Input(shape=(1,), name='stun_input')
    x_stun = Dense(4, activation='relu')(stun_input)

    #Here we concat the 3 inputs in a single vector
    concat = Concatenate()([x_vis, x_hold, x_stun])

    hid_layer = Dense(16, activation='relu')(concat)
    output = Dense(NUM_ACTIONS, activation='linear')(hid_layer)
    model = Model(inputs=[obs_input, hold_input, stun_input], outputs=output)
    return model

glob_model = modelNN()

def set_model_weights(chromo):
    global glob_model

def get_model_chromo_shape():
    global glob_model
    tot_weights = 0
    # each i is correspondent to the weights of each layer in the neural network
    for i in glob_model.get_weights():
        tot_weights += i.size
    return tot_weights        

def policy(obs, agent_id):
    global glob_model

    obs_array = np.array(obs) / 250.0  # data normalization
    return random.randint(0, 6)

# def load_weights(chromo_weights, agent_idx):



