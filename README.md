####README
This repo serves as a minimal implmentation case for iCore project. 

IEEE_13_3p.py is the Python code for the three phase IEEE-13 bus built by the provided OpenDSS model.

safeDDPG.py is the model for our proposed method.

train_DDPG.py is the code for training or testing the model

The checkpoints are available at checkpoints/three-phase/13bus3p/safe-ddpg/three_single
(Checkpoints in three_single copy refers to linear controllers)

If you want to test the trained models, please go to the train_DDPG.py, change 
'''
status = 'test'
'''
Then run train_DDPG.py