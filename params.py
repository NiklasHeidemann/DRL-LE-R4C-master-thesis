

def visible_positions_13(x_y):
    return [
            (x_y[0]+1,x_y[1]+1),(x_y[0]+1,x_y[1]),(x_y[0]+1,x_y[1]-1),
            (x_y[0],x_y[1]+1),(x_y[0],x_y[1]),(x_y[0],x_y[1]-1),
            (x_y[0]-1,x_y[1]+1),(x_y[0]-1,x_y[1]),(x_y[0]-1,x_y[1]-1),
            (x_y[0]+2,x_y[1]),(x_y[0]-2,x_y[1]),(x_y[0],x_y[1]+2),(x_y[0],x_y[1]-2)
        ]

visible_positions_5 =         lambda x_y: [
            (x_y[0]+1,x_y[1]),
            (x_y[0],x_y[1]+1),(x_y[0],x_y[1]),(x_y[0],x_y[1]-1),
            (x_y[0]-1,x_y[1]),
        ]

# network
BATCH_SIZE =128
LAYER_SIZE = 32
LEARNING_RATE = 0.001
LSTM_SIZE = 32
NUMBER_OF_BIG_LAYERS = 2
RECURRENT = True
SELF_PLAY = True
TIME_STEPS = 1
ALPHA = 0.05
GAMMA = 0.99
TARGET_ENTROPY = 1.

#environment
WORLD_GENERATOR = "choice" # multi or random or choice
GRID_SIZE_RANGE = (4,8)
MAX_TIME_STEP = 30
NUMBER_OF_AGENTS = 2
NUMBER_OF_OBJECTS_TO_PLACE_RANGE = (0.08,0.2)
MAX_OBJECT_COLOR_RANGE = 4
POS_REWARD = 2
NEG_REWARD = -0.0

#input
NUMBER_COMMUNICATION_CHANNELS = 1
SIZE_VOCABULARY = MAX_OBJECT_COLOR_RANGE
VISIBLE_POSITIONS = visible_positions_13

#buffer
MAX_REPLAY_BUFFER_SIZE = 10000
PRE_SAMPLING_STEPS = 10

#training
ENVIRONMENT_STEPS_PER_TRAINING = 500
TRAININGS_PER_TRAINING = 8
EPOCHS = 1000000
SEED=10
PARALLEL_ENVS = 6
FROM_SAVE = False

ACTIONS = ["HOLD", "LEFT", "RIGHT", "UP", "DOWN"]
