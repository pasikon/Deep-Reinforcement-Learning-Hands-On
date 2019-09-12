import numpy as np
import gym
from collections import namedtuple

# keras modules
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense, Activation, Softmax
from keras.optimizers import Adam

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()

    while True:
        obs_v = np.array([obs])

        # NN prediction
        act_probs = np.squeeze(model.predict(obs_v))

        # calculate softmax from pure network output
        # act_probs = softmax(np.squeeze(act_probs))
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda step: step.reward, batch))
    reward_bound = np.percentile(rewards,
                                 percentile)  # bierze percentyl czyli odleglosc min -> max rewards w 100 krokach
    # max jest 200 bo env jest done jak sie wywroci albo rewards jest 200?
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:  # percentyl najlepszych wynikow
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    return train_obs, train_act, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = Sequential()

    model.add(Dense(units=128, input_dim=obs_size))
    model.add(Activation('relu'))

    model.add(Dense(units=n_actions))

    model.add(Softmax())

    # print summary to double check the network
    model.summary()
    # create a nice image of the network model
    plot_model(model, to_file='01_cartpole_K.png', show_shapes=True)

    model.compile(loss='sparse_categorical_crossentropy', # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
                  optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])

    for iter_no, batch in enumerate(iterate_batches(env, model, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        print(reward_m)

        # do softmaxa wchodza pojedyncze poniewaz uzyto sparse_categorical_crossentropy
        # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        model.fit(y=np.asarray(acts_v), x=np.asarray(obs_v), epochs=30, verbose=0)
        if reward_m > 199:
            print("Solved!")
            break
