from domain import visible_positions_13

# network
BATCH_SIZE =128
LAYER_SIZE = 32
LEARNING_RATE = 0.001
LSTM_SIZE = 32
NUMBER_OF_BIG_LAYERS = 2
RECURRENT = True
SELF_PLAY = True
TIME_STEPS = 10
ALPHA = 0.06
GAMMA = 0.99
TARGET_ENTROPY = 1.

#environment
WORLD_GENERATOR = "random" # multi or random or choice
GRID_SIZE_RANGE = (4,8)
MAX_TIME_STEP = 30
NUMBER_OF_AGENTS = 3
COMMUNISM = True
AGENT_DROPOUT_PROBS = 0.5 if NUMBER_OF_AGENTS == 3 else 0 # meaning with 0.5 probabilty the third agent is not placed
NUMBER_OF_OBJECTS_TO_PLACE_RANGE = (0.08,0.15)
OBJECT_COLOR_RANGE = (1,5)
POS_REWARD = 2
NEG_REWARD = -0.1
XENIA_LOCK = True
XENIA_PERMANENCE = False

#input
NUMBER_COMMUNICATION_CHANNELS = 1
SIZE_VOCABULARY = OBJECT_COLOR_RANGE[1]
VISIBLE_POSITIONS = visible_positions_13

#buffer
MAX_REPLAY_BUFFER_SIZE = 10000
PRE_SAMPLING_STEPS = 1000

#training
ENVIRONMENT_STEPS_PER_TRAINING = 500
TRAININGS_PER_TRAINING = 8
EPOCHS = 300000
SEED=12
ENV_PARALLEL = 32
FROM_SAVE = True
RENDER = False

