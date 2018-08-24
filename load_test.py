import numpy as np

# import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM
from keras.optimizers import Adam

# keras-rl agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# trader environment
from TraderEnv import OhlcvEnv
# custom normalizer
from util import NormalizerProcessor


def main():
    # OPTIONS
    ENV_NAME = 'OHLCV-v0'
    TIME_STEP = 30
    WINDOW_LENGTH = TIME_STEP
    ADDITIONAL_STATE = 4

    # Get the environment and extract the number of actions.
    PATH_TRAIN = "./data/train/"
    PATH_TEST = "./data/test/"
    env = OhlcvEnv(TIME_STEP, path=PATH_TRAIN)
    env_test = OhlcvEnv(TIME_STEP, path=PATH_TEST)

    # random seed
    np.random.seed(123)
    env.seed(123)

    nb_actions = env.action_space.n

    model = Sequential()
    model.add(CuDNNLSTM(64, input_shape=env.shape, return_sequences=True))
    model.add(CuDNNLSTM(64))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=50000, window_length=TIME_STEP)
    # policy = BoltzmannQPolicy()
    policy = EpsGreedyQPolicy()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=200,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy,
                   processor=NormalizerProcessor())
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    ### now only test
    dqn.load_weights("./model/duel_dqn_OHLCV-v0_weights_49112166LS_184_297_4.033341265853485.h5f")
    # validate
    info = dqn.test(env_test, nb_episodes=1, visualize=False)
    n_long, n_short, total_reward, portfolio = info['n_trades']['long'], info['n_trades']['short'], info[
        'total_reward'], int(info['portfolio'])
    np.array([info]).dump(
        './model/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.info'.format(ENV_NAME, portfolio, n_long, n_short,
                                                                     total_reward))
    dqn.save_weights(
        './info/duel_dqn_{0}_weights_{1}LS_{2}_{3}_{4}.h5f'.format(ENV_NAME, portfolio, n_long, n_short, total_reward),
        overwrite=True)

if __name__ == '__main__':
    main()

