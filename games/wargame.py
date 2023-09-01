import datetime
import math
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (3, 21, 21)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(10 * 21 * 21 * 21 * 21))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 3  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 100  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = gym.make("CartPole-v1")
        if seed is not None:
            self.env.seed(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done, _ = self.env.step(action)
        return numpy.array([[observation]]), reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return numpy.array([[self.env.reset()]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Push cart to the left",
            1: "Push cart to the right",
        }
        return f"{action_number}. {actions[action_number]}"

class Player:
    '''只感知敌人状态'''
    def __init__(self, _id, _team, _type, _dis, _lives, _HP, _dam, _axis):
        self.id = _id
        self.team = _team # 敌方队伍 1 我方队伍 0
        self.type = _type # 步兵 0 炮兵 1 特种兵 2
        self.dis = _dis # 攻击距离
        self.lives = _lives # 可复活次数
        self.HP = _HP # 生命值
        self.dam = _dam # 攻击伤害
        self.axis = _axis # 坐标

    def get_info(self, who): # TODO 好像用不到
        '''who'''
        if who.team != self.team:
            return self._axis
        #TODO

class WarGame:
    def __init__(self):
        self.board_size = 20
        self.players_num = 20
        self.cur_player = 0
        self.players_list = [] # [0:10] 我方队伍  [10:] 敌方队伍
        self.max_health = 100
        self.max_lives = 3
        self.board = numpy.zeros((self.board_size * self.board_size), dtype="int32")
        self.act_num = 0 # 记录已经活动的士兵 每回合每个队伍内的每个角色可以活动10步
        self.step = 0 # 记录该team已经活动的步数

        # 创建障碍物
        self.place_obstacles()
        self.init_characters()
        
        
    def place_obstacles(self):
        obstacles_num = numpy.random.randint(0, 300)
        for _ in range(obstacles_num):
            row, col = numpy.random.randint(self.board_size, size = 2)
            self.board[row][col] = -1

    def random_empty_cell(self): # 随机一个位置
        while True:
            row, col = numpy.random.randint(self.board_size, size = 2)
            if self.board_size[row][col] == 0:
                return row, col

    def init_characters(self):
        for _id in range(1, self.players_num + 1): 
            _row, _col = self.random_empty_cell()
            if _id <= 10:
                team = 0
            else:
                team = 1
            if _id % 10 <= 2: # 2个特种兵
                self.players_list.append(Player(_id, team, 2, 1, 2, 100, 11, [_row, _col]))
            elif _id % 10 <= 5: # 3个炮兵
                self.players_list.append(Player(_id, team, 1, 1, 2, 100, 12, [_row, _col]))
            else:
                self.players_list.append(Player(_id, _team=team, _type=0, _dis=1, _lives=2, _HP=100, _dam=10, _axis=[_row, _col]))

    def to_play(self):
        # 遍历搜索
        # team轮流走 考虑队伍有减员的情况
        # 每回合每个队伍内的每个角色可以活动10步

        while(1): # 死亡跳过
            self.cur_player += 1
            if self.cur_player % 10 == 0: # 一个队伍走完一步
                #TODO 每走完10个单位（1步）刷新一次地图（把地图上的暂存路径都删除）
                self.steps += 1
                if self.steps == 10: 
                    # 走完10步 切换队伍 刷新step
                    self.steps = 0
                    self.cur_player =  self.cur_player % self.players_num: # 切换到第一个
                else:
                    # 走完1步 重新回到该队伍第一个士兵
                    self.cur_player = self.cur_player - self.players_num/2 # 10个角色
                    self.steps = 0
            if self.players_list[self.cur_player].lives >= 0: # TODO 需要确认0条命时是死是活
                break
        return self.cur_player
    
    def reset(self):
        self.__init__()
    
    # def step(self):
        
