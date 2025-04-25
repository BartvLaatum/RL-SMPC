import random, os, tqdm
from functools import partial
import numpy as np
import torch
import casadi

# from SHARED.params import *
# from SHARED.setup import weather_data
# from SHARED.aux_functions import get_disturbance, print_metrics
from envs.lettuce_greenhouse import LettuceGreenhouse
from common.rl_utils import make_vec_env, normalize_state

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# from torch.utils.tensorboard import SummaryWriter

ALGS = {"sac": SAC, "ppo": PPO}

class neural_net(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim,hidden_layers) -> None:
        super(neural_net, self).__init__()
        self.hh = hidden_layers
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim) 
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = torch.nn.Linear(hidden_dim, 1)  
        # self.relu = torch.nn.ReLU()  # ReLU activation function
        self.act_fn = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        # if self.h == 2:
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def weights_reset(self):
        for layer in self.children():
            if hasattr(layer, 'weight'):
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_, output = self.data[idx]
        return torch.tensor(input_,dtype=torch.float32),output
                        
class value_function_TR():
    def __init__(
        self,
        input_dim,
        hidden_dim,
        learning_rate, 
        batch_size,
        hidden_layers = 2,
        reduced = False,
        env_id = "LettuceGreenhouse",
        algorithm = "sac",
        N = 3840,
        wandb_run = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.reduced = reduced
        
        self.neural_net = neural_net(input_dim, hidden_dim, hidden_layers)
        self.neural_net.weights_reset()

        self.optimizer = torch.optim.Adam(self.neural_net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=N
        )

        self.env_id = env_id
        self.trajectories = {}
        self.values = {} 
        self.obs_count = {}       
        i = 1
        log_path = "vf_logs_TR/"
        path = log_path+str(i)
        self.algorithm = algorithm
        self.wandb_run = wandb_run
        while os.path.exists(path):
            i+=1
            path = log_path+str(i)
            
        # self.writer = SummaryWriter(path)

    def learn(self, global_step = 0):
        
        for i, (inputs, outputs) in enumerate(self.data_loader_train, 0):
            inputs = inputs.to("cuda")
            outputs = outputs.to("cuda")
            obs =  inputs
            total_return =  outputs.unsqueeze(1).float()

            #optimizer
            self.optimizer.zero_grad()

            # Calculate TD error
            current_state_values = self.neural_net(obs)   

            #Calculating Loss
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(current_state_values, total_return) 

            # Logging loss for visualization using TensorFlow
            log_step = (global_step) + (i)
            # self.writer.add_scalar('Train/loss', loss.item(),global_step=log_step)
            # self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'],global_step=log_step)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()      
            
            if log_step % 10 == 0:
                self.validate(log_step)

        self.wandb_run.log({"train_loss": loss.item()})

    def evaluate_value(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        value = self.neural_net(obs_tensor).detach().numpy()[0][0]
        return value 
    
    def validate(self, global_step = 0):
        self.neural_net.eval()
        for i, (inputs, outputs) in enumerate(self.data_loader_validate,0):
            inputs = inputs.to("cuda")
            outputs = outputs.to("cuda")

            obs =  inputs
            total_return =  outputs.unsqueeze(1).float()  
            # Calculate TD error
            current_state_values = self.neural_net(obs).detach()
            #Calculating Loss
            mse_loss = torch.nn.MSELoss()
            loss = mse_loss(current_state_values, total_return) 
            # Logging loss for visualization using TensorFlow
            # self.writer.add_scalar('Train/validation', loss.item(), global_step=global_step)
            self.wandb_run.log({"val_loss": loss.item()})
            break
        self.neural_net.train()

    def prepare_data(self):
        split_ratio = 0.8

        all_data = list(self.values.items())
        random.shuffle(all_data)
        split_point = int(len(all_data) * split_ratio)
        
        train_values    = all_data[:split_point]
        validate_values = all_data[split_point:]

        self.train_data    = dict(train_values)
        self.validate_data = dict(validate_values)

        self.train_dataset     = MyDataset(self.train_data)
        self.validate_dataset  = MyDataset(self.validate_data)

        self.data_loader_train = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.data_loader_validate = torch.utils.data.DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, epochs):

        self.prepare_data()

        total_iterations = epochs*len(self.data_loader_train)
        self.scheduler.total_iters = total_iterations

        print(f"Total training iterations: {total_iterations}")
        self.neural_net.train()
        for i in tqdm.tqdm(range(epochs)):
            self.learn(global_step = i*len(self.data_loader_train))

        self.neural_net.eval()
        # self.writer.close() 

    # def sim_with_mpc(self,num_traj = 1, spread = 0.1):
    #     self.neural_net.eval()
        
    #     #Env Setup
    #     env = greenhouseEnv()
    #     env.random_starts = False
    #     env.stochastic = False
    #     env.using_mpc = True   
        
    #     #Controllet Setup
    #     N = (3600*1)//dT
    #     MPC = casadi.Function.load('MPC/MPC_controller_1hr')
    #     casadi.DM.set_precision(15)
    #     get_d = partial(get_disturbance,weather_data = weather_data,start_time=start_time,Np=N, dt=dT)
        
    #     #Logs for debugging
    #     obs_log = []
        
    #     for t in range(num_traj):
    #         if t == 0:
    #             #Initial condtions
    #             obs_now, info = env.reset()
    #             x_now = info["x"]
    #             u_opt = obs_now[4:7]
    #             kk = 0
    #         else:
    #             #Initial condtions
    #             kk = np.random.randint(0,max_steps-1)
                
    #             for (_, _),(_,x_next,x_now,time_k) in self.trajectories[0]:
    #                 if kk == time_k:
    #                     env.set_env_state(x_next,x_now,u0,kk)
    #                     break
                    
    #             obs_now,x_now,u_opt = env.sample_obs(spread)
                
    #         traj = [] #to store observation and reward
    #         for k in tqdm.tqdm(range(kk,max_steps)):
                
    #             #Get weather prediction
    #             d_now = get_d(k)  
                
    #             #Get optimal action and trajectories
    #             u_opt,u_traj,x_traj = MPC(x_now,d_now,u_opt)
    #             u_opt  = np.array(u_opt)      
    #             u_traj = np.array(u_traj)
    #             x_traj = np.array(x_traj)
    #             u_opt = np.clip(u_opt,u_min.reshape(3,1),u_max.reshape(3,1))
                
    #             #Step in env
    #             obs_next, reward, done, _,info = env.step(u_opt)
    #             reward = 0 if done else reward
                
    #             x_next = info["x"]                
                
    #             obs_log.append(obs_next)
    #             obs_now_norm = normalizeObs(obs_now,obs_min,obs_max)
                                
    #             traj.append(((obs_now_norm,reward),(obs_now,x_next,x_now,k)))                
                        
    #             obs_now = obs_next
    #             x_now = x_next
             
    #         self.trajectories[t] = traj
        
    #     for t in self.trajectories:
    #         cumulative_reward = 0
    #         traj = self.trajectories[t]
    #         for (obs, reward),(_,_,_,_) in reversed(traj):
    #             cumulative_reward += reward
    #             self.values[tuple(obs)] = cumulative_reward
    #     return obs_log, self.trajectories
    
    def sim_with_agent(self, env_params, num_traj=1, spread=0.1, model_path="", env_path="", stochastic=False):
        self.neural_net.eval()

        #Env Setup
        env = LettuceGreenhouse(**env_params)

        # env = make_vec_env(
        #     self.env_id, 
        #     env_params, 
        #     seed=666, 
        #     n_envs=1, 
        #     monitor_filename=None, 
        #     vec_norm_kwargs=None,
        #     eval_env=True
        # )


        # Creating trained normalized environment
        env_norm = LettuceGreenhouse(**env_params)
        env_norm = DummyVecEnv([lambda: env_norm])
        env_norm = VecNormalize(env_norm, norm_obs = True, norm_reward = False, clip_obs = 10., gamma=1)
        env_norm = env_norm.load(env_path, env_norm)
        env_norm.training = False

        #Controllet Setup
        agent = ALGS[self.algorithm].load(model_path, env=env_norm)

        # Logs for debugging
        obs_log = []

        print(f"Simulating {num_traj} trajectories")
        # simulate N trajectories
        for t in tqdm.tqdm(range(num_traj)):
            # at first simulation; start from initial conditions
            if t == 0:
                # initial condtions
                obs_now, _ = env.reset(seed=42)
                x_now = env.get_numpy_state()
                # u_opt = obs_now[4:7]
                kk = 0
            else:
                kk = np.random.randint(0, env.N-1)
                # Initial condtionsobs_now, info = env.reset()
                for (_, _),(_, obs_now, x_next, x_now, time_k) in self.trajectories[0]:
                    if kk == time_k:
                        env.set_env_state(x_next, x_now, obs_now[4:7], kk)
                        found = 1
                        break

                if found == 0:
                    print ("ERROR, point not found")
                    break

                obs_now, x_now, _ = env.sample_obs(spread)

            traj = [] #to store observation and reward
            for k in range(kk, env.N):

                # Get optimal action and trajectories
                obs_now_norm = env_norm.normalize_obs(obs_now)
                action, _ = agent.predict(obs_now_norm, deterministic=True)            
                # Step in env
                obs_next, reward, done, _,info = env.step(action)
                reward = 0 if done else reward
                x_next = env.get_state()

                obs_log.append(obs_now)

                if self.reduced:
                    obs2store = normalize_state(
                        x = np.array([x_now[0],obs_now[7]]),
                        x_min = np.array([env.x_min[0], 0]),
                        x_max = np.array([env.x_max[0], env.N]))
                else:
                    obs2store = obs_now_norm
                traj.append(((obs_now_norm, reward), (obs2store, obs_now, x_next, x_now, k)))                

                obs_now = obs_next
                x_now = x_next

            self.trajectories[t] = traj

        # calculates the cumulative reward for each observation; 
        # i.e., the target for the value function
        for t in self.trajectories:
            cumulative_reward = 0
            traj = self.trajectories[t]
            for (obs, reward),(obs2store,_,_,_,_) in reversed(traj):
                cumulative_reward += reward
                # stored_obs = tuple(obs)
                # stored_obs = tuple([obs[0], obs[7]]) #This is the normalized drymass state and normalized time step
                stored_obs = tuple(obs2store)
                if stored_obs in self.obs_count:
                    self.values[stored_obs] = (self.values[stored_obs] * self.obs_count[stored_obs] + cumulative_reward)/(self.obs_count[stored_obs] + 1)
                    self.obs_count[stored_obs] +=1
                else:
                    self.obs_count[stored_obs] = 1
                    self.values[stored_obs] = cumulative_reward
                
        self.dataset = MyDataset(self.values)
        return obs_log, self.trajectories
    
    # def test_with_mpc(self):
    #     self.neural_net.eval()
        
    #     #Env Setup
    #     env = LettuceGreenhouse()
    #     env.random_starts = False
    #     env.stochastic = False
    #     env.using_mpc = True   

    #     #Controllet Setup
    #     N = (3600*1)//dT
    #     MPC = casadi.Function.load('MPC/MPC_controller_1hr')
    #     casadi.DM.set_precision(15)
    #     get_d = partial(get_disturbance,weather_data = weather_data,start_time=start_time,Np=N, dt=dT)
        
    #     #Logs for showing
    #     Y_log, D_log, U_log = [],[],[]
    #     total_reward = 0
    #     cum_reward_log = []
    #     to_show_reward = []
    #     to_show_values = []
    #     value_log = []
    #     obs_log = []
        
    #     #Initial condtions
    #     obs_now, info = env.reset()
    #     x_now = info["x"]
    #     u_opt = obs_now[4:7]
        
    #     for k in tqdm.tqdm(range(max_steps)):
            
    #         #Get weather prediction
    #         d_now = get_d(k)  
            
    #         #Get optimal action and trajectories
    #         u_opt,u_traj,x_traj = MPC(x_now,d_now,u_opt)
    #         u_opt  = np.array(u_opt)      
    #         u_traj = np.array(u_traj)
    #         x_traj = np.array(x_traj)
    #         u_opt = np.clip(u_opt, u_min.reshape(3,1), u_max.reshape(3,1))
            
    #         #Step in env
    #         obs_next, reward, done, _,info = env.step(u_opt)
    #         x_next = info["x"]                
            
    #         obs_log.append(obs_next)
    #         obs_now_norm = normalizeObs(obs_now,obs_min,obs_max)
    #         obs_next_norm= normalizeObs(obs_next,obs_min,obs_max)

    #         value = self.evaluate_value(obs_next_norm)  

    #         total_reward += reward
    #         cum_reward_log.append(total_reward)
    #         value_log.append(value)
    #         Y_log.append(info['output'])
    #         D_log.append(obs_next[8:])
    #         U_log.append(obs_next[4:7])  
            
    #         obs_now = obs_next
    #         x_now = x_next
            
            
    #     U_log = np.array(U_log)
    #     Y_log = np.array(Y_log)
    #     D_log = np.array(D_log)
    #     to_show_reward = np.array(cum_reward_log)
    #     to_show_values = np.array(value_log)
        
    #     print_metrics(Y_log, U_log, D_log,vf = to_show_values,rewards=to_show_reward, day_range=(0,40))
 
    def test_with_agent(self, env_params, test_episodes=0, spread=0.5, startk=False, stochastic=False, model_path="", env_path=""):
        self.neural_net.eval()
        model_path = model_path
        env_path = env_path
        
        #Env Setup
        env = LettuceGreenhouse(**env_params)
        
        #Creating trained normalized environment
        env_norm = LettuceGreenhouse(**env_params)
        env_norm = DummyVecEnv([lambda: env_norm])
        env_norm = VecNormalize(env_norm, norm_obs = True, norm_reward = False, clip_obs = 10.,gamma=1)
        env_norm = env_norm.load(env_path,env_norm)
        env_norm.training = False
        
        #Controllet Setup
        agent = SAC.load(model_path,env=env_norm)
        
        traj = []
        
        test_episodes = test_episodes+1 if startk else test_episodes
        
        for e in range(test_episodes):
            #Logs for showing
            Y_log, D_log, U_log = [],[],[]
            total_reward = 0
            cum_reward_log = []
            to_show_reward = []
            to_show_values = []
            value_log = []
            obs_log = []
            
            #Initial condtions
            if e == 0:
                obs_now, info = env.reset()
                x_now = env.get_state()
                # u_opt = obs_now[4:7]
                kk = 0
                
            else:
                if startk != False:
                    kk = startk
                else:                
                    kk = np.random.randint(0, env.N-1)
                
                for (_, _),(_,_,x_next,x_now,time_k) in self.trajectories[0]:
                    if kk == time_k:
                        env.set_env_state(x_next, x_now, u0, kk)
                        break
                if startk == False:   
                    obs_now,x_now,u_opt = env.sample_obs(spread)
                
            for k in tqdm.tqdm(range(kk, env.N)):
                
                #Get optimal action and trajectories
                obs_now_norm = env_norm.normalize_obs(obs_now)
                action, _ = agent.predict(obs_now_norm, deterministic=True)            
                #Step in env
                obs_next, reward, done, _,info = env.step(action)
                reward = 0 if done else reward
                x_next = info["x"]                
                
                obs_log.append(obs_next)
                # obs_next_norm= env_norm.normalize_obs(obs_next)
                
                if self.reduced == True:
                    obs2store = normalize_state(
                        np.array([x_now[0], obs_now[7]]),
                        np.array([env.x_min[0], 0]),
                        np.array([env.x_max[0], env.N]))
                else:
                    obs2store = obs_now_norm
                value = self.evaluate_value(obs2store)  
                
                traj.append(((obs_now_norm,reward),(obs2store,obs_now,x_next,x_now,k)))  
                                    
                total_reward += reward
                cum_reward_log.append(total_reward)
                value_log.append(value)
                Y_log.append(info['output'])
                D_log.append(obs_next[8:])
                U_log.append(obs_next[4:7])  
                
                obs_now = obs_next
                x_now = x_next
            
            U_log = np.array(U_log)
            Y_log = np.array(Y_log)
            D_log = np.array(D_log)
            to_show_reward = np.array(cum_reward_log)
            to_show_values = np.array(value_log)
            
            if not self.trajectories:
                self.trajectories[e] = traj
                
            print_metrics(Y_log, U_log, D_log,vf = to_show_values,rewards=to_show_reward, day_range=(0,40))
            
        return Y_log,U_log,to_show_reward,to_show_values

# End of script