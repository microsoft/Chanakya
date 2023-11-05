import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import random
from collections import deque
import pickle

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

import torch
import torch.nn.functional as F
import torch.nn as nn

np.set_printoptions(precision=3)

class MSEMultiple(nn.Module):
    def __init__(self):
        super(MSEMultiple,self).__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, preds, targets):
        loss = 0.0
        for i in range(len(preds)):
            loss += self.mse(preds[i], targets[i])
        return loss

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out_sizes, n_layers, old_style=False):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.d_in = d_in
        self.old_style = old_style
        self.linear_in = nn.Sequential( 
            nn.Linear(d_in, d_hidden),
            nn.ReLU()
        )
        self.linear_layers = nn.ModuleList(
                [ nn.Sequential(
                    nn.Linear(d_hidden, d_hidden), 
                    nn.ReLU()
                    ) for i in range(n_layers - 2)
                ])
        if self.old_style:
            self.linear_out = nn.ModuleList([ 
                nn.Sequential(
                    nn.Linear(d_hidden, d_out),
                ) for d_out in d_out_sizes 
                ])
        else:
            self.linear_out = nn.Linear(d_hidden, sum(d_out_sizes))

    def forward(self, X):
        X = X.view(-1, self.d_in)
        X = self.linear_in(X)
        for i in range(len(self.linear_layers)):
            X = self.linear_layers[i](X)
        if self.old_style:
            output = [ linear_out_i(X) for linear_out_i in self.linear_out ]
        else:
            output = self.linear_out(X)
        return output

class EpsilonGreedy:
    '''
    Create an EpsilonGreedy agent
    Functions:
        pull: select the arm according to the EpsilonGreedy algorithm; returns arm
        reset: reset the EpsilonGreedy agent 
    '''
    def __init__(self, agent_id, no_arms, epsilon, min_epsilon, epsilon_decay):
        self.agent_id = agent_id
        self.no_arms = no_arms
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.info = {}
        self.round = 0
        self.score = 0
        self.history = [[0] for _ in range(self.no_arms)]
    
    def get_dict(self):
        d = {
            "agent_id" : self.agent_id,
            "no_arms" : self.no_arms,
            "epsilon" : self.epsilon,
            "min_epsilon" : self.min_epsilon,
            "epsilon_decay" : self.epsilon_decay,
            "info" : copy.deepcopy(self.info),
            "round" : self.round,
            "score" : self.score,
            "history" : copy.deepcopy(self.history)
        }
        return d
    
    def set_dict(self, d):
        self.agent_id = d["agent_id"]
        self.no_arms = d["no_arms"]
        self.epsilon = d["epsilon"]
        self.min_epsilon = d["min_epsilon"]
        self.epsilon_decay = d["epsilon_decay"]
        self.info = copy.deepcopy(d["info"])
        self.round = self.round
        self.score = self.score
        self.history = copy.deepcopy(d["history"])

    def pull(self, values, test_phase):
        self.round += 1
        sample_for_rl = np.random.binomial(1, self.epsilon) if not test_phase else 0
        if sample_for_rl == 0:
            #select greedily
            arm = np.argmax(values)
        else:   
            #select randomly
            arm = np.random.choice([ x for x in range(len(values)) ])               
            # arm = np.random.randint(len(values))
        f = [(self.history[action][-1] + int(arm==action)) for action in range(self.no_arms)]        
        [self.history[action].append(f[action]) for action in range(self.no_arms)]     

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return arm     
    
    # def reset(self):
    #     self.score = 0
    #     self.round = 0  
    #     self.history = [[0] for _ in range(self.no_arms)]

    def plot(self, out_folder, prefix=None, action_set=None):
        try:
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
            x_axis = list(range(len(self.history[0])))
            actions_frequency = self.history
            [plt.plot(x_axis, actions_frequency[arm]) for arm in range(self.no_arms)]
            plt.xlabel('Timesteps', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.title('Frequency of actions by Agent {}'.format(self.agent_id), fontsize=20)
            if action_set is None:
                plt.legend([('Action ' + str(i)) for i in range(self.no_arms)], loc='lower right')   
            else:
                plt.legend(action_set, loc='lower right')
            #plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
            if prefix is None:
                file_name = os.path.join(out_folder, 'Agent_'+ str(self.agent_id)+'.png')
            else:
                file_name = os.path.join(out_folder, '{}.png'.format(prefix))
            plt.savefig(file_name, format='png')
            plt.close()
        except:
            print("Plot Error")

class SingleUCB:
    '''
    Create an UCB agent
    Functions:
        pull: select the arm according to the UCB algorithm; returns arm
        reset: reset the UCB agent i.e. delete the stastical data
        bonus: calculate the hoffeding temporal bonus
    '''
    def __init__(self, agent_id, no_arms, c):
        self.agent_id = agent_id
        self.no_arms = no_arms
        self.c = c
        self.info = {}
        for arm in range(self.no_arms):
            self.info[arm] = {}
            self.info[arm]['no_visited'] = 0
            self.info[arm]['is_visited'] = False
        self.history = []
        self.round = 0
        self.score = 0

    def get_dict(self):
        d = {
            "agent_id" : self.agent_id,
            "no_arms" : self.no_arms,
            "c" : self.c,
            "info" : copy.deepcopy(self.info),
            "round" : self.round,
            "score" : self.score,
            "history" : copy.deepcopy(self.history)
        }
        return d
    
    def set_dict(self, d):
        self.agent_id = d["agent_id"]
        self.no_arms = d["no_arms"]
        self.c = d["c"]
        self.info = copy.deepcopy(d["info"])
        self.round = self.round
        self.score = self.score
        self.history = copy.deepcopy(d["history"])

    def pull(self, values, test_phase):
        self.round += 1
        
        for arm in range(self.no_arms):
            if not self.info[arm]['is_visited']:
                self.info[arm]['is_visited'] = True
                self.info[arm]['no_visited'] += 1
                self.history.append([self.info[action]['no_visited'] for action in range(self.no_arms)])
                return arm
        bonus = self.bonus() if not test_phase else 0
        decision_values = values + bonus
        arm = np.argmax(decision_values)
        self.info[arm]['no_visited'] += 1

        self.history.append([self.info[action]['no_visited'] for action in range(self.no_arms)])
        
        return arm
    
    # def reset(self):
    #     self.info = {}
    #     for arm in range(self.no_arms):
    #         self.info[arm] = {}
    #         self.info[arm]['no_visited'] = 0
    #         self.info[arm]['is_visited'] = False
    #     self.round = 0    
    #     self.history = []
    
    def bonus(self):
        bonus = []
        all_arm_visited = all([self.info[arm]['is_visited'] for arm in range(self.no_arms)])
        if all_arm_visited:
            for arm in range(self.no_arms):
                bonus.append(self.c * np.sqrt((2 * np.log(self.round)) / self.info[arm]['no_visited']))
        return np.array(bonus)

    def plot(self, out_folder, prefix=None, action_set=None):
        try:
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
            x_axis = list(range(len(self.history)))
            actions_frequency = [[frequency[arm] for frequency in self.history] for arm in range(self.no_arms)]
            [plt.plot(x_axis, actions_frequency[arm]) for arm in range(self.no_arms)]
            plt.xlabel('Timesteps', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.title('Frequency of actions by Agent {}'.format(self.agent_id), fontsize=20)
            if action_set is None:
                plt.legend([('Action ' + str(i)) for i in range(self.no_arms)], loc='lower right')   
            else:
                plt.legend(action_set, loc='lower right')
            #plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
            if prefix is None:
                file_name = os.path.join(out_folder, 'Agent_'+ str(self.agent_id)+'.png')
            else:
                file_name = os.path.join(out_folder, '{}.png'.format(prefix))
            plt.savefig(file_name, format='png')
            plt.close()
        except:
            print("Plot Error")

class MultiOutputMAB:
    '''
    Create MultiOutputMAB which can handle multiple decision points
    Fucntions: 
    state: takes 1-D array/list and return list of actions 
    remember: takes state, actions and reward, and add it to the memory
    update: update the model 
    '''
    def __init__(self, rl_config, memory_length=256, device="cuda:0", old_style=False):
        self.rl_config = rl_config
        self.state_size = self.rl_config["states"]["state_size"]

        self.action_space_sizes = [ len(acts) for key, acts in self.rl_config["actions"] ]
        self.batch_size = self.rl_config["rl"]["mlp"]["batch_size"]
        self.device = device
        self._make_agents(self.rl_config["rl"])
        self._make_model(self.rl_config["rl"]["mlp"])
        self.memory = deque(maxlen=memory_length)
        self.steps = 0
        self.sequence_reward_list = []
        self.reward_list = []
        self.state_archive = []
        self.actions_archive = [[] for i in range(len(self.action_space_sizes))]
        self.old_style = old_style
        self.default_mask = [True for i in range(len(self.action_space_sizes))]
        #self.update_frequency = 50

    def _make_model(self, mlp_config):
        self.model = MLP(self.state_size, mlp_config["hidden_layer_size"], self.action_space_sizes, n_layers=mlp_config["num_layers"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # self.criterion = MSEMultiple()
        self.criterion = nn.MSELoss()
        self.model.to(self.device)
        # self.model.apply(init_weights)
    
    def save(self, out_path, prefix):
        torch.save(self.model.state_dict(), os.path.join(out_path, prefix + ".qmodel.pth"))
        contextual_bandit = {}
        agents_dict = {}
        for agent in self.agents:
            agent_dict = agent.get_dict()
            agents_dict[agent_dict["agent_id"]] = copy.deepcopy(agent_dict)
        contextual_bandit['agents_dict'] = agents_dict
        contextual_bandit['sequence_reward_list'] = self.sequence_reward_list
        contextual_bandit['reward_list'] = self.reward_list
        contextual_bandit['state_archive'] = self.state_archive
        contextual_bandit['actions_archive'] = self.actions_archive
        with open(os.path.join(out_path, prefix + ".agents.pkl"), "wb") as f:
            pickle.dump(contextual_bandit, f)
        torch.cuda.synchronize()
    
    def load(self, out_path, prefix):
        self.model.load_state_dict(torch.load(os.path.join(out_path, prefix + ".qmodel.pth")))
        with open(os.path.join(out_path, prefix + ".agents.pkl"), "rb") as f:
            contextual_bandit = pickle.load(f)
        agents_dict = contextual_bandit['agents_dict']
        self.sequence_reward_list = contextual_bandit['sequence_reward_list']
        self.reward_list = contextual_bandit['reward_list']
        self.state_archive = contextual_bandit['state_archive']
        try:
            self.actions_archive = contextual_bandit['actions_archive']
        except:
            print("action_archive not found. Initilizing it")
            self.actions_archive = [[] for i in range(len(self.action_space_sizes))]

        for i in range(len(self.agents)):
            self.agents[i].set_dict(agents_dict[i])
        torch.cuda.synchronize()

    def _make_agents(self, algo_config):
        if algo_config["algo"] == 'ucb':
            self.agents = [SingleUCB(agent_id, action_size, algo_config["c"]) for agent_id, action_size in enumerate(self.action_space_sizes)]
        else:
            self.agents = [EpsilonGreedy(agent_id, action_size, algo_config["epsilon"], algo_config["min_epsilon"], algo_config["epsilon_decay"]) for agent_id, action_size in enumerate(self.action_space_sizes)]

    def act(self, state, test_phase=False):
        self.steps += 1
        if test_phase:
            self.state_archive.append(copy.deepcopy(state))
        with torch.no_grad():
            state = state.astype(np.float32)
            state = torch.from_numpy(state).to(self.device)
            state[torch.isnan(state)] = 0
            state[torch.isinf(state)] = 0
            values = self.model(state)
            #print("###")
            #print(self.steps)
            # # print(state)
            #print(values)
            #print("###")
            if self.old_style:
                values = [ values[0].cpu().numpy() for value in values ]
                actions = []
                for i, action_size in enumerate(self.action_space_sizes):
                    value_i = values[i]
                    action_i = self.agents[i].pull(value_i, test_phase=test_phase)
                    actions.append(action_i)
            else:
                values = values[0].cpu().numpy()
                actions = []
                visited_values = 0
                for i, action_size in enumerate(self.action_space_sizes):
                    value_i = values[visited_values : visited_values + action_size]
                    action_i = self.agents[i].pull(value_i, test_phase=test_phase)
                    actions.append(action_i)
                    visited_values += action_size
        if test_phase:
            visited_values = 0
            for i, action_size in enumerate(self.action_space_sizes):
                self.actions_archive[i].append(actions[i] + visited_values)
                visited_values += action_size
        return actions
    
    def remember(self, state, actions, reward):
        self.sequence_reward_list.append(reward)
        self.reward_list.append(reward)
        self.memory.append((state.astype(np.float32), actions, float(reward)))
        '''
        if self.steps % self.update_frequency:
            self.update()
        '''
    
    def update(self):
        batch_size = min(len(self.memory), self.batch_size)
        training_batch = random.sample(self.memory, batch_size)
        state_batch = torch.tensor([training_batch[i][0] for i in range(batch_size)]).to(self.device)
        actions_batch = [training_batch[i][1] for i in range(batch_size)]
        reward_batch = [training_batch[i][2] for i in range(batch_size)]
        
        self.optimizer.zero_grad()

        values = self.model(state_batch)

        if self.old_style:
            target_values = [torch.clone(value) for value in values]
            for idx in range(len(actions_batch)):
                for j, action_size in enumerate(self.action_space_sizes):
                    target_values[j][idx][actions_batch[idx][j]] = reward_batch[idx]

            loss = self.criterion(values, target_values)
        else:
            target_values = torch.clone(values)
            for idx in range(len(actions_batch)):
                visited_values = 0
                for j, action_size in enumerate(self.action_space_sizes):
                    target_values[idx, actions_batch[idx][j] + visited_values ] = reward_batch[idx]
                    visited_values += action_size
            loss = self.criterion(values, target_values)

        loss.backward()
        self.optimizer.step()

    def plot(self, out_folder, prefixes = None, action_sets = None):
        try:
            for i, agent in enumerate(self.agents):
                if prefixes is not None and action_sets is not None:
                    agent.plot(out_folder, prefix=prefixes[i], action_set=action_sets[i])
                elif prefixes is not None and action_sets is None:
                    agent.plot(out_folder, prefix=prefixes[i])
                elif prefixes is None and action_sets is not None:
                    agent.plot(out_folder, action_set=action_sets[i])
                else:
                    agent.plot(out_folder)
            num_bins = 20
            plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
            n, bins, patches = plt.hist(self.reward_list, num_bins, facecolor='blue', alpha=0.5)
            plt.xlabel('Rewards', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.title('Histogram of rewards at sequence', fontsize=20)
            if prefixes is None:
                file_name = os.path.join(out_folder, 'Rewards.png')
            else:
                file_name = os.path.join(out_folder, '{}_reward_latest.png'.format('_'.join(prefixes[0].split('_')[:-3])))
            plt.savefig(file_name, format='png')
            plt.close()
        except:
            print("Plot Error")

    def plot_rewards(self, out_folder, prefix, sequence=None, reset=True):
        try:
            out_folder = os.path.join(out_folder, prefix+'_rewards')
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            num_bins = 20
            plt.figure(num=None, figsize=(20,10), facecolor='w', edgecolor='k')
            n, bins, patches = plt.hist(self.sequence_reward_list, num_bins, facecolor='blue', alpha=0.5)
            plt.xlabel('Rewards', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.title('Histogram of rewards at sequence {}'.format(str(sequence)), fontsize=20)
            if sequence is None:
                file_name = os.path.join(out_folder, 'Rewards.png')
            else:
                file_name = os.path.join(out_folder, 'reward_{}.png'.format(str(sequence)))
            plt.savefig(file_name, format='png')
            plt.close()
            if reset:
                self.sequence_reward_list = []
        except:
            print("Plot Error")

    def features_importance(self, out_folder, model=None):
        feature_names = ['curr_scale', 'ada_scale', 
                            'class_info_0', 'class_info_1', 'class_info_2', 'class_info_3', 
                            'class_info_4', 'class_info_5', 'class_info_6', 'class_info_7', 
                            'conf_info_0', 'conf_info_1', 
                            'tb_lr_crop_0', 'tb_lr_crop_1', 'tb_lr_crop_2', 'tb_lr_crop_3']
        if model is None:
            model = self.model
        state_archive = np.array(self.state_archive)
        input_tensor = torch.from_numpy(state_archive).type(torch.FloatTensor).to(self.device)
        ig = IntegratedGradients(model)
        input_tensor.requires_grad_()
        for i in range(len(self.action_space_sizes)):
            agent_prefix = 'Agent-{}'.format(i)
            action_labels_i = np.array(self.actions_archive[i])
            action_labels_i = torch.from_numpy(action_labels_i).to(self.device)
            attr = ig.attribute(input_tensor, target=action_labels_i)
            attr = attr.cpu().detach().numpy()
            self._visualize_importances(out_folder, agent_prefix, feature_names, np.mean(attr, axis=0))
        
    # Helper method to print importances and visualize distribution
    def _visualize_importances(self, out_folder, agent_prefix, feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Context"):
        print(title, agent_prefix)
        for i in range(len(feature_names)):
            print(feature_names[i], ": ", '%.3f'%(importances[i]))
        x_pos = (np.arange(len(feature_names)))
        if plot:
            try:
                title = title + ' for ' + agent_prefix
                prefix = 'FeatureImportance_' + agent_prefix
                plt.figure(figsize=(12,6))
                plt.bar(x_pos, importances, align='center')
                plt.xticks(x_pos, feature_names, wrap=True, fontsize=5)
                plt.xlabel(axis_title)
                plt.title(title)
                file_name = os.path.join(out_folder, '{}.png'.format(prefix))
                plt.savefig(file_name, format='png')
                plt.close()   
            except:
                print("Plot Error Feature Importance")       
 

if __name__ == "__main__":
    state_size = 19 # curr_scale, ada_scale, conf_metric [2 * 1], class_metric [8 * 1], loc_extent [4 * 1] ##### + complexity [3 * 1]
    prop_choices = [50, 100, 200, 300, 500]
    det_scales_choices = [ (2000, 720), (2000, 640), (2000, 600), (2000, 560), (2000, 480), (2000, 360) ]
    model_names = ["fcos", "faster_rcnn"]

    rl_model = MultiOutputMAB(state_size, [ len(prop_choices), len(det_scales_choices), len(model_names) ], device="cpu")
    print(rl_model.model)
    for i in range(50):
        rl_model.act(np.random.normal(0, 0.1, state_size).astype(np.float32))
        rl_model.remember(
            np.random.normal(0, 0.1, state_size).astype(np.float32),
            [0, 0],
            np.random.normal(0.5, 0.4, 1)[0]
        )
    rl_model.update()
    print('Tested')
    rl_model.plot("plots")
    # vals = rl_model.act()
    # print(vals)
    '''
    This is demonstrate the use of this class
    #There are two decison points, for 1st action space size is 2 and of the 2nd action space size 3
    >>> action_space_sizes = [2,3]
    #State has two features
    >>> state_size = 2
    >>> from UCB import MultiOutputUCB
    #make the agent object
    >>> agent = MultiOutputUCB(state_size, action_space_sizes)
    #get state from the simulater and recive actions; say state = [1,2]
    >>> actions = agent.act([1,2])
    #let actions be [1,1] i.e select action 1 from 1st action space and select action 1 from 2st action space
    #Note: the action start from 0 not from 1
    #obtain reward and new state from the simulater; let reward be 0.5
    #Now push the transition to the agents memory
    >>> agent.remember([1,2], [1,1], 0.5)
    #now repet the above steps
    #after some time update the agent's model
    >>> agent.update()
    '''
