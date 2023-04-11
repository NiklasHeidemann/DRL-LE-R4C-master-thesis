
visible_positions_13 =         lambda x_y: [
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
LSTM_SIZE = 8
NUMBER_OF_BIG_LAYERS = 1
RECURRENT = False
SELF_PLAY = False
TIME_STEPS = 1
ALPHA = 0.00

#environment
MAX_GRID_SIZE_RANGE = 3
MAX_TIME_STEP = 30
NUMBER_OF_AGENTS = 2
NUMBER_OF_OBJECTS_TO_PLACE_RANGE = (0.12,0.2)

#input
NUMBER_COMMUNICATION_CHANNELS = 0
SIZE_VOCABULARY = 4
VISIBLE_POSITIONS = visible_positions_13

#buffer
MAX_REPLAY_BUFFER_SIZE = 10000
PRE_SAMPLING_STEPS = 10#000

#training
ENVIRONMENT_STEPS_PER_TRAINING = 500
TRAININGS_PER_TRAINING = 8
EPOCHS = 100000
SEED=3
FROM_SAVE = False

# TODO smoother lines in plot
# TODO save and load
ACTIONS = ["HOLD", "LEFT", "RIGHT", "UP", "DOWN"]
