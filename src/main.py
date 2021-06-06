import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from DQN import DQNAgent
from idle import IdleGame
from random import randint
from typing import List
import random
import statistics
import torch.optim as optim
import torch
import datetime
import distutils.util
from collections import defaultdict

DEVICE = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'

#################################
#   Define parameters manually  #
#################################
def define_parameters():
    params = dict()
    # Neural Network
    params["epsilon_decay_linear"] = 1 / 100
    params["learning_rate"] = 0.00013629
    params["first_layer_size"] = 200  # neurons in the first layer
    params["second_layer_size"] = 20  # neurons in the second layer
    params["third_layer_size"] = 50  # neurons in the third layer
    params["episodes"] = 105
    params["memory_size"] = 200000
    params["batch_size"] = 200000
    # Settings
    params["weights_path"] = "weights/weights.h5"
    params["train"] = True
    params["test"] = False
    params["plot_score"] = True
    params["log_path"] = (
        "logs/scores_" + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + ".txt"
    )
    return params


def get_record(score: float, record: float):
    if score >= record:
        return score
    else:
        return record


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13, 8))
    fit_reg = False if train == False else True
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        # color="#36688D",
        x_jitter=0.1,
        scatter_kws={"color": "#36688D"},
        label="Data",
        fit_reg=fit_reg,
        line_kws={"color": "#F49F05"},
    )
    # Plot the average line
    y_mean = [np.mean(array_score)] * len(array_counter)
    ax.plot(array_counter, y_mean, label="Mean", linestyle="--")
    ax.legend(loc="upper right")
    ax.set(xlabel="# games", ylabel="score")
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)


def test(params: dict):
    params["load_weights"] = True
    params["train"] = False
    params["test"] = False
    score, mean, stdev = run(params)
    return score, mean, stdev


def run(params: dict):
    """
    Run the DQN algorithm, based on the parameters previously set.
    """
    # state (input) is equal to the number of actions, minus 1 (do nothing)
    # plus 3 (time, money, all time high money)
    input_size: int = len(IdleGame().actions) + 1
    output_size: int = len(IdleGame().actions)
    agent: DQNAgent = DQNAgent(params, input_size, output_size)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(
        agent.parameters(), weight_decay=0, lr=params["learning_rate"]
    )
    counter_games: int = 0
    score_plot: List[float] = []
    counter_plot: List[float] = []
    record: float = 0
    total_score: float = 0
    while counter_games < params["episodes"]:
        zero_counts: float = 0
        # Initialize classes
        idle_game: IdleGame = IdleGame()
        counter: defaultdict = defaultdict(lambda: 0)

        if not params["train"]:
            agent.epsilon = 0.01
        else:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = max(1 - (counter_games * params["epsilon_decay_linear"]), 0)
            # if counter_games < 1000:
            #     agent.epsilon = 0.01
            # else:
            #     agent.epsilon = 0

        # Perform first move
        agent.memory.clear()
        agent.reward = 0

        steps: int = 0  # steps since the last positive reward
        max_steps: int = 10000
        while steps < max_steps:

            # get old state
            state_old = np.asarray(idle_game.get_state())

            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < agent.epsilon:
                move_index = idle_game.random_action()
                final_move = idle_game.one_hot_actions(move_index)
            else:
                # predict action based on the old state
                with torch.no_grad():
                    state_old_tensor = torch.tensor(
                        state_old.reshape((1, input_size)), dtype=torch.float32
                    ).to(DEVICE)
                    prediction = agent(state_old_tensor).detach().cpu().numpy()[0]
                    valid_actions = idle_game.valid_actions()
                    # I bet there is a much more pythonic way of doing this
                    for i in range(len(prediction)):
                        if i not in valid_actions:
                            prediction[i] = 0
                    move_index = np.argmax(prediction)
                    final_move = np.eye(input_size)[move_index]
                    # print(move_index, prediction, valid_actions)

            # perform new move and get new state

            # print(final_move)
            # action_index = 0
            # action = idle_game.one_hot_actions(action_index)
            counter[move_index] = counter[move_index] + 1
            idle_game.execute_action(move_index)

            simulation_steps: int = 1  # random.randint(1, max_steps // 1000)
            # print(simulation_steps)

            idle_game.step(simulation_steps)

            state_new = idle_game.get_state()

            # set reward for the new state
            reward = idle_game.get_reward()

            if params["train"]:
                agent.set_reward(reward)
                # store the new data into a long term memory
                agent.remember(state_old, move_index, reward, state_new)

            # record = get_record(game.score, record)
            steps += simulation_steps
        if params["train"]:
            agent.replay_new(agent.memory, params["batch_size"])
        counter_games += 1
        total_score = idle_game.get_reward()
        agent.upsample()
        print(
            f"Game {counter_games}      Score: {total_score}      State: {idle_game.get_state()}    Mistakes: {idle_game.mistakes}      Moves: {counter}       Epsilon: {agent.epsilon}"
        )
        score_plot.append(total_score)
        counter_plot.append(counter_games)
    mean, stdev = get_mean_stdev(score_plot)
    if params["train"]:
        model_weights = agent.state_dict()
        torch.save(model_weights, params["weights_path"])
    if params["plot_score"]:
        plot_seaborn(counter_plot, score_plot, params["train"])
    return total_score, mean, stdev


if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    params = define_parameters()
    if params["train"]:
        print("Training...")
        params["load_weights"] = False  # when training, the network is not pre-trained
        run(params)
    if params["test"]:
        print("Testing...")
        params["train"] = False
        params["load_weights"] = True
        run(params)
