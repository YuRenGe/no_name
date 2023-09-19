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
        self.action_space = list(range(9))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 3  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 2000  # Maximum number of moves if game is not finished before
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
        self.training_steps = 500000  # Total number of training steps (ie weights update according to a batch)
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
        self.env = WarGame()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(9))
    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()
    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        print(self.env.render())
        input("Press enter to take a step ")

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        cur_chess = self.env.cur_chess
        cur_player = self.env.cur_player

        actions = { # 坐标变换
            0:(0, 0), # 不动
            1:(1, 0), # 东
            2:(1, -1), # 东南
            3:(0, -1), # 南
            4:(-1, -1), # 西南
            5:(-1, 0), # 西
            6:(-1, 1), # 西北
            7:(0, 1), # 北
            8:(1, 1), # 东北
        }
        return f"Turns:{self.env.turn} steps:{self.env.steps} cur_player:{cur_player}\n chess:{cur_chess} move:{actions[action_number]}"

class Chess:
    '''只感知敌人状态'''
    def __init__(self, _id, _team, _type, _dis, _lives, _HP, _dam, _pos):
        self.id = _id
        self.team = _team # 敌方队伍 1 我方队伍 0
        self.type = _type # 步兵 0 炮兵 1 特种兵 2
        self.dis = _dis # 攻击距离
        self.lives = _lives # 可复活次数
        self.HP = _HP # 生命值
        self.dam = _dam # 攻击伤害
        self.pos = _pos # 坐标
        self.attack_pos = set() # 本回合已攻击过的坐标
    def get_info(self, who): # TODO 好像用不到
        '''who'''
        if who.team != self.team:
            return self._axis
    def update_pos(self, new_pos):
        self.pos[0], self.pos[1] = new_pos[0], new_pos[1]

class WarGame:
    def __init__(self):
        self.OBSTACLE_AXIS = 0
        self.TEAM1_AXIS = 1
        self.TEAM2_AXIS = 2
        self.MAX_TURN = 100
        self.team_member = [10, 10] # 一开始存活10人
        self.board_size = 21
        self.chess_num = 20
        self.cur_player = 0
        self.cur_chess = 0 # to_play + 1  1~20
        self.chess_list = [] # [0:10] 我方队伍  [10:] 敌方队伍
        self.board = []
        # self.render_change_count = 0 # for test
        for i in range(3):
            t = []
            for j in range(self.board_size):
                t.append([0] * self.board_size)
            self.board.append(t)
        self.board_render = []
        for j in range(self.board_size):
            self.board_render.append([0] * self.board_size)# 记录士兵移动
        self.act_num = 0 # 记录已经活动的士兵 每回合每个队伍内的每个角色可以活动10步
        self.steps = 0 # 记录一方玩家已经活动的步数
        self.turn = 0 # 记录回合数
        
        # 创建障碍物
        self.place_obstacles()
        self.init_characters()
        # 移动方式
        self.direction_move = { # 坐标变换
            0:(0, 0), # 不动
            1:(1, 0), # 南
            2:(1, -1), # 西南
            3:(0, -1), # 西
            4:(-1, -1), # 西北
            5:(-1, 0), # 北
            6:(-1, 1), # 东北
            7:(0, 1), # 东
            8:(1, 1), # 东南
        }
        # 分数定义
        self.normalReward = {
            "MEET_OBSTACLE":-100,
            "ATTACK_ONEENEMY":50,
            "ATTACK_ONEENEMY_ADDING":50, # 集火
            "MEET_ALI":-200
        }

    def place_obstacles(self):
        obstacles_num = numpy.random.randint(0, 300)
        for _ in range(obstacles_num):
            row, col = numpy.random.randint(self.board_size, size = 2)
            self.board[self.OBSTACLE_AXIS][row][col] = 1
            self.board_render[row][col] = -1

    def random_empty_cell(self): # 随机一个空位置
        while True:
            row, col = numpy.random.randint(self.board_size, size = 2)
            if self.board[self.OBSTACLE_AXIS][row][col] == 0 and \
                self.board[self.TEAM1_AXIS][row][col] == 0 and \
                self.board[self.TEAM2_AXIS][row][col] == 0:
                return row, col

    def init_characters(self):
        for _id in range(1, self.chess_num + 1): 
            _row, _col = self.random_empty_cell()
            if _id < self.chess_num/2:
                team = 1
                self.board[self.TEAM1_AXIS][_row][_col] = 1
            else:
                team = 2
                self.board[self.TEAM2_AXIS][_row][_col] = 1
            self.board_render[_row][_col] = _id
            if _id % 10 < 2: # 2个特种兵
                self.chess_list.append(Chess(_id, team, 2, 1, 2, 100, 11, [_row, _col]))
            elif _id % 10 < 5: # 3个炮兵
                self.chess_list.append(Chess(_id, team, 1, 1, 2, 100, 12, [_row, _col]))
            else:
                self.chess_list.append(Chess(_id, _team=team, _type=0, _dis=1, _lives=2, _HP=100, _dam=10, _pos=[_row, _col]))

    def to_play(self):
        # 遍历搜索
        # team轮流走 考虑队伍有减员的情况
        # 每回合每个队伍内的每个角色可以活动10步
        # 训练时角色状态实时更新（实际输入时敌方玩家只有回合初始位置信息）  
        # 实际游戏中一方角色不感知另一方角色血量及复活位置 如果复活相撞应该会报异常（随机性）
        while(1): # 死亡跳过
            self.cur_player = 0
            self.cur_chess += 1   
            if self.cur_chess == self.chess_num + 1 or self.cur_chess==int(self.chess_num/2) + 1: # 一个队伍走完一步
                self.steps += 1
                if self.steps == 10: 
                    # 走完10步 切换队伍 刷新step 刷新攻击列表 
                    for chess in self.chess_list:
                        chess.attack_pos = set()
                    self.steps = 0
                    # 切换到下一队第一个
                    if self.cur_chess == self.chess_num + 1:
                        self.cur_chess = 1
                    else:
                        self.cur_chess = int(self.chess_num/2) +1
                    if self.cur_chess == 1:
                        self.turn += 1
                else:
                    # 走完1步 重新回到该队伍第一个士兵
                    if self.cur_chess == self.chess_num + 1:
                        self.cur_chess = int(self.chess_num/2) +1
                    else:
                        self.cur_chess = 1
            if self.chess_list[self.cur_chess-1].lives >= 0: # TODO 需要确认0条命时是死是活
                if self.cur_chess <= 10:
                    self.cur_player = 0
                else:
                    self.cur_player = 1
                break
        return self.cur_player
    
    def reset(self):
        self.__init__()
        return numpy.array(self.board, dtype=numpy.float32)
    def expert_action(self):
        ##随机走
        cur_chess_exp = self.chess_list[self.cur_chess-1]
        legal_actions, illegal_actions = self.legal_actions(cur_chess_exp.pos)
        ## 
        action = numpy.random.choice(legal_actions)
        return action

    def legal_actions(self, pos):
        '''
        返回合法的行为 以及不合法的行为及后果
        '''
        MEET_OBSTACLE = 0
        legal_actions = [0]
        illegal_actions = {}
        for action in range(1, 9):
            after_row, after_col = self.apply_move(pos, action)
            if after_row < 0 or \
                after_row >= self.board_size or \
                after_col < 0 or \
                after_col >= self.board_size: # 越界
                illegal_actions[action] = self.normalReward["MEET_OBSTACLE"]
            elif self.board[self.OBSTACLE_AXIS][after_row][after_col] == 1 :
                illegal_actions[action] = self.normalReward["MEET_OBSTACLE"]
            elif self.board[self.TEAM1_AXIS][after_row][after_col] == 1 or \
                self.board[self.TEAM2_AXIS][after_row][after_col] == 1: # 碰撞
                illegal_actions[action] = self.normalReward["MEET_ALI"]
            else:
                legal_actions.append(action)
        return legal_actions, illegal_actions
    def apply_move(self, pos, action):
        return [pos[0] + self.direction_move[action][0], pos[1] + self.direction_move[action][1]]
    
    def step(self, action):
        ret_reward = 0
        if self.cur_chess <= 10 :
            cur_team = self.TEAM1_AXIS
            ene_team = self.TEAM2_AXIS
        else:
            cur_team = self.TEAM2_AXIS
            ene_team = self.TEAM1_AXIS
        if self.turn == self.MAX_TURN: # 100回合结束
            # 生命值差值绝对值 鼓励进攻
            ret_reward += (self.team_member[cur_team - 1] - self.team_member[ene_team - 1]) * 200
            return self.board, ret_reward, True
        cur_chess_exp = self.chess_list[self.cur_chess-1]
        legal_actions, illegal_actions = self.legal_actions(cur_chess_exp.pos)
        after_row, after_col = self.apply_move(cur_chess_exp.pos, action)
        
        if action in legal_actions: # 如果合法 地图变 位置变
            self.board[cur_team][cur_chess_exp.pos[0]][cur_chess_exp.pos[1]] = 0
            self.board[cur_team][after_row][after_col] = 1
            self.board_render[cur_chess_exp.pos[0]][cur_chess_exp.pos[1]] = 0
            self.board_render[after_row][after_col] = cur_chess_exp.id # 移动
            # print(cur_chess_exp.pos[0],cur_chess_exp.pos[1],after_row, after_col,cur_chess_exp.id)
            cur_chess_exp.update_pos([after_row, after_col])
            attack_reward = self.get_attack_reward(cur_chess_exp)
            ret_reward += attack_reward
        else: # 不合法 扣分
            ret_reward -= illegal_actions[action]
        
        if self.complete_destroy(cur_team): # 完全消灭
            ret_reward += self.team_member[cur_team - 1] * 200
            return self.board, ret_reward, True
        return numpy.array(self.board, dtype=numpy.float32), ret_reward, False 
    
    def get_attack_reward(self, cur_chess):
        cur_pos = cur_chess.pos
        if cur_chess.team == 1 :
            enemy_team = self.TEAM2_AXIS
        else:
            enemy_team = self.TEAM1_AXIS
        attack_reward = 0
        # TODO 集火
        for fire_range in self.direction_move:
            if fire_range == 0:
                continue
            fire_pos_x = cur_pos[0] + self.direction_move[fire_range][0]
            fire_pos_y = cur_pos[1] + self.direction_move[fire_range][1]
            if fire_pos_x < 0 or fire_pos_x >= self.board_size:
                continue
            if fire_pos_y < 0 or fire_pos_y >= self.board_size:
                continue
            fire_pos = (fire_pos_x, fire_pos_y)
            if fire_pos not in cur_chess.attack_pos and self.board[enemy_team][fire_pos_x][fire_pos_y] == 1:
                cur_chess.attack_pos.add(fire_pos)
                attack_reward += 50
                # 更新被攻击方信息
                enemy_id = self.board_render[fire_pos_x][fire_pos_y]
                enemy_exp = self.chess_list[enemy_id-1]
                enemy_exp.HP -= cur_chess.dam
                if enemy_exp.HP <= 0:
                    # 死亡
                    enemy_exp.lives -= 1
                    self.board[enemy_exp.team][enemy_exp.pos[0]][enemy_exp.pos[1]] = 0
                    self.board_render[enemy_exp.pos[0]][enemy_exp.pos[1]] = 0
                    # 复活
                    if enemy_exp.lives >= 0:
                        rst_pos_x, rst_pos_y = self.random_empty_cell()
                        enemy_exp.pos = [rst_pos_x, rst_pos_y]
                        self.board[enemy_exp.team][rst_pos_x][rst_pos_y] = 1
                        self.board_render[rst_pos_x][rst_pos_y] = enemy_exp.id
                    else:
                        self.team_member[enemy_exp.team - 1] -= 1
        return attack_reward
    
    def complete_destroy(self, cur_team):
        if cur_team == 1:
            enemy_team_index = 1
        else:
            enemy_team_index = 0
        if self.team_member[enemy_team_index] == 0: # 1->1(team2) 2->0(team1)
            return True
        else:
            return False

    def render(self):
        # TODO
        return numpy.array(self.board_render)
