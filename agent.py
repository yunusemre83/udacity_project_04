import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    valid_Traffic_lights = ['red','green']
    
    def __init__(self, env, alpha=0.35, gamma=0.2):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here      
        self.state_action_table = pd.DataFrame(columns=('light','oncoming','left','planner_action','learner_action','value'))
        db_inx_counter=0;        
        for i_light  in self.valid_Traffic_lights:
            for i_oncoming in self.env.valid_actions:
                for i_left in self.env.valid_actions:
                    for i_planner_action in self.env.valid_actions:
                        for i_learner_action in self.env.valid_actions:
                          self.state_action_table.loc[db_inx_counter] = [i_light,i_oncoming,i_left,i_planner_action,i_learner_action,random.uniform(0,0.2)] 
                          db_inx_counter=db_inx_counter+1;
        self.state_action_table = self.state_action_table.fillna(value='none')                  
        self.next_waypoint = None
        self.valid_actions = self.env.valid_actions
        self.alpha = alpha
        self.gamma = gamma  
        self.initial_deadline = 50
        self.total_rewards = 0.0
        self.reward_hist = list()
        self.trial_num = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_waypoint = None    
        self.initial_deadline = self.env.agent_states[self]['deadline']
        self.reward_hist.append(self.total_rewards)        
        self.total_rewards = 0.0
        self.trial_num = self.trial_num + 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        if self.next_waypoint is None:
            self.next_waypoint='none'        
        inputs = self.env.sense(self)      
        deadline = self.env.get_deadline(self)
        
        for i_keys,i_values in inputs.items():
            if (i_values is None):
                inputs[i_keys]='none'
        
        # TODO: Update state
        subset_state_action_table = self.state_action_table[(self.state_action_table['light']==inputs['light']) \
        & (self.state_action_table['oncoming']==inputs['oncoming']) & (self.state_action_table['left']==inputs['left']) \
        & (self.state_action_table['planner_action']==self.next_waypoint) ] 

        # TODO: Select action according to your policy
        e_greedy = random.uniform(0,1)
        threshold = t/float(self.initial_deadline)        
        
        if(e_greedy > 2 * threshold):
            # Select a Random Action to Learn = Explore
            action = random.choice(self.env.valid_actions[0:])
            if action is None:
                action='none'
        else:
            # Select best Q(s,a) = Exploit
            action = subset_state_action_table['learner_action'].iloc[np.argmax(subset_state_action_table['value'].values)]         
           
        # Execute action and get reward
        if action == 'none':
            reward = self.env.act(self, None)
        else:     
            reward = self.env.act(self, action)

        self.total_rewards = self.total_rewards + reward
        # Find the max action for the next state
        max_qvalue_next_state = np.max(self.state_action_table['value'].values)

        # TODO: Learn policy based on state, action, reward
        index_qval = self.state_action_table[(self.state_action_table['light']==inputs['light']) \
        & (self.state_action_table['oncoming']==inputs['oncoming']) & (self.state_action_table['left']==inputs['left']) \
        & (self.state_action_table['planner_action']==self.next_waypoint) & (self.state_action_table['learner_action'] == action) ].index.values

        updated_qval = self.state_action_table.ix[index_qval,'value'].values
        
        updated_qval = updated_qval + self.alpha * (reward + self.gamma*max_qvalue_next_state - updated_qval)

        self.state_action_table.ix[index_qval,'value'] = updated_qval
        
        print ("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward))  # [debug]
        print ("Accumulated Reward: {}".format(self.total_rewards))

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    return [a.state_action_table, a.reward_hist]


if __name__ == '__main__':
    [state_action_table_learned,final_reward_history]= run()
    plt.plot(state_action_table_learned['value'].values)
