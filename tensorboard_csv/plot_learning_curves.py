import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the data from CSV file
data_1_rew = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240427-1413_rew.csv')
data_1_bl = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240427-1413_bl.csv')
data_1_loss = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240427-1413_loss.csv')
data_1_val_loss = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240427-1413_val-los.csv')

data_2_rew = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240430-1432_rew.csv')
data_2_bl = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240430-1432_bl.csv')
data_2_loss = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240430-1432_loss.csv')
data_2_val_loss = pd.read_csv('tensorboard_csv/SVRPTWn10m1_240430-1432_val-los.csv')


data_1_rew['Episode'] = ((data_1_rew['Step'] - 1) // 1000) + 1
data_1_bl['Episode'] = ((data_1_bl['Step'] - 1) // 1000) + 1
data_1_loss['Episode'] = ((data_1_loss['Step'] - 1) // 1000) + 1
data_1_val_loss['Episode'] = ((data_1_val_loss['Step'] - 1) // 1000) + 1

data_2_rew['Episode'] = ((data_2_rew['Step'] - 1) // 1000) + 1
data_2_bl['Episode'] = ((data_2_bl['Step'] - 1) // 1000) + 1
data_2_loss['Episode'] = ((data_2_loss['Step'] - 1) // 1000) + 1
data_2_val_loss['Episode'] = ((data_2_val_loss['Step'] - 1) // 1000) + 1

# Setting theme
sns.axes_style()

custom = {"axes.edgecolor": "black"}
sns.set_style("whitegrid", rc = custom)
# sns.set_theme(style="whitegrid")
# sns.color_palette('bright')

##### First figures #####

data_1_rew['label'] = 'reward'
data_2_rew['label'] = 'reward'
data_1_bl['label'] = 'baseline value'
data_2_bl['label'] = 'baseline value'

data = pd.concat([data_1_rew, data_2_rew, data_1_bl, data_2_bl])

fig, ax = plt.subplots(1,2, figsize=(8, 3.7))

# Plotting
ax[0].tick_params(axis='both', which='major', labelsize=12)
sns.lineplot(data=data, x='Episode', y='Value', hue='label', ax=ax[0], palette=['brown', 'mediumblue'])

# Enhancing the plot
# plt.title('Learning Curve')

ax[0].set_xlabel('Episode', fontsize=16)
ax[0].set_ylabel('Values', fontsize=16)
ax[0].legend(fontsize=16)

# Display the plot
# fig.savefig("tensorboard_csv/learning_curves.png") 

# ##### Second figures #####

data_1_act_loss = data_1_loss
data_1_act_loss['Value'] = data_1_loss['Value'] - data_1_val_loss['Value']
data_2_act_loss = data_2_loss
data_2_act_loss['Value'] = data_2_loss['Value'] - data_2_val_loss['Value']

data_1_act_loss['label'] = 'actor loss'
data_2_act_loss['label'] = 'actor loss'
data_1_val_loss['label'] = 'critic loss'
data_2_val_loss['label'] = 'critic loss'

data = pd.concat([data_1_act_loss, data_2_act_loss, data_1_val_loss, data_2_val_loss])

# Plotting
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
ax[1].tick_params(axis='both', which='major', labelsize=12)
sns.lineplot(data=data, x='Episode', y='Value', hue='label', ax=ax[1])

# Enhancing the plot
# plt.title('Learning Curve')
ax[1].set_xlabel('Episode', fontsize=16)
ax[1].set_ylabel('Losses', fontsize=16)
ax[1].legend(fontsize=16)


# Display the plot
# plt.tight_layout()
# plt.show()
fig.tight_layout()
fig.savefig("tensorboard_csv/learning_curves.png", dpi=500) 