import random
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Concatenate

# Rui -> Estratégia RN + CE (Redes Neuronais + Computação Evolucionária) AGENTE 1

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
    """
    For each weight layer in glob_model, its shape and size are discovered.
    Then, a slice with that size is cut from the chromosome, it is reshaped 
    to have that shape, and its used to replace the weights of that layer 
    in the model.
    
    """
    global glob_model
    new_weights = []
    cur_idx = 0
    # each i is correspondent to the weights of each layer in the neural network
    for i in glob_model.get_weights():
        size = i.size
        shape = i.shape
        # we here get a slice from the chromossome and we give it the layer shape 
        w_slice = chromo[cur_idx : cur_idx + size]
        new_weights.append(w_slice.reshape(shape))
        cur_idx += size
    # in the end we have a list of arrays of each chromossome slice reshaped to match the model´s layer shape 
    glob_model.set_weights(new_weights)
    

def get_model_chromo_shape():
    """
    Here, we will get the size (length) of the chromossome's vector.
    
    """
    global glob_model
    tot_weights = 0
    # each i is correspondent to the weights of each layer in the neural network
    for i in glob_model.get_weights():
        tot_weights += i.size
    return tot_weights        

def policy(obs, agent_id):
    """
    The policy function will use the 'glob_model' which is the modelNN() function
    and the dictionary 'obs' to decide the next action
    
    """
    global glob_model

    vis = np.array(obs["grid"]) / 250.0  # data normalization
    vis = vis.reshape(1, OBS_SHAPE[0], OBS_SHAPE[1], 1)

    hold = np.array(obs["holding"])
    hold = hold.reshape(1, 1)

    stun = np.array(obs["stun"]) / STUN_TIME
    stun = stun.reshape(1, 1)

    #  Action predict
    act_pref = glob_model([vis, hold, stun], training=False).numpy()

    # Action Choice
    cho_act = np.argmax(act_pref[0])

    return int(cho_act)

def load_weights(chromo_weights, agent_idx):
    """
    Load the weights from the .npy file and 
    that file is used in this agent
    
    """
    try:
        chromo = np.load(chromo_weights)
        agent_len = get_model_chromo_shape()
        # agent 1 will take the first half from the weights list
        if agent_idx == 1:
            c1 = chromo[0 : agent_len]
            set_model_weights(c1)
        # agent 2 will take the second half from the weights list
        elif agent_idx == 2:
            c2 = chromo[agent_len :]
            set_model_weights(c2)
        print(f"Agent {agent_idx} loaded the {chromo_weights} weights")
    except FileNotFoundError:
        print(f"File {chromo_weights} not found, currently using random weights")
    except Exception as e:
        print(f"Failed to load the weights: {e}, using random weights")

load_weights("best_chrome_team.npy", 1)



