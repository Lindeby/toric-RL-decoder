import numpy as np
import matplotlib.pyplot as plt

path_data_dir           = "../data/"
path_actor_samples      = "sample_distribution_actor_13_Mar_2020-15:48:18.data"
path_learner_samples    = "sample_distribution_learner_13_Mar_2020-15:48:18.data"
separator_timestamp     = ":::::"
separator_data          = " "


# read actor data
data_actor       = []
timestamps_actor = []
with open(path_data_dir + path_actor_samples, 'r') as f:
    # header = f.readline()
    #_, info = l.split(separator_timestamp)
    # interval = float(info[info.find("interval=") + len("interval=")])
    # max      = float(info[info.find("max=") + len("max=")])
    lines = f.readlines()
    for l in lines:
        timestamp, data = l.split(separator_timestamp)
        data = np.fromstring(data, sep=separator_data)
        data_actor.append(data)
        timestamps_actor.append(timestamp)

# read learner data
data_learner         = []
timestamps_learner   = []
with open(path_data_dir + path_learner_samples, 'r') as f:
    # header = f.readline()
    #_, info = l.split(separator_timestamp)
    # interval = float(info[info.find("interval=") + len("interval=")])
    # max      = float(info[info.find("max=") + len("max=")])
    lines = f.readlines()
    for l in lines:
        timestamp, data = l.split(separator_timestamp)
        data = np.fromstring(data, sep=separator_data)
        data_learner.append(data)
        timestamps_learner.append(timestamp)


data_actor = np.array(data_actor)
data_actor = data_actor/np.max(data_actor)
data_actor_sum = data_actor_sum/np.max(data_actor_sum)

data_learner = np.array(data_learner)
data_learner = data_learner/np.max(data_learner)

fig, ax = plt.subplots(2,figsize=(20,8))

# Actor data
ax[0].plot(data_actor[ 0,:], label=timestamps_actor[0])
ax[0].plot(data_actor[ int(data_actor.shape[0]/2),:], label=timestamps_actor[int(data_actor.shape[0]/2)])
ax[0].plot(data_actor[-1,:], label=timestamps_actor[-1])


# Learner data
ax[1].plot(data_learner[ 0,:], label=timestamps_learner[ 0])
ax[1].plot(data_learner[ int(data_learner.shape[0]/2),:], label=timestamps_learner[int(data_learner.shape[0]/2)])
ax[1].plot(data_learner[-1,:], label=timestamps_learner[-1])


ax[0].set_title("Priority Distribution from Actors", fontsize=26)
ax[1].set_title("Priority Distribution from Learner Sampling", fontsize=26)
ax[0].set(xticks=np.arange(0,1000, 20), xticklabels=np.arange(0,100,2))
ax[1].set(xticks=np.arange(0,1000, 20), xticklabels=np.arange(0,100,2))
ax[0].tick_params(axis='x', labelsize=12)
ax[1].tick_params(axis='x', labelsize=12)
ax[0].set_xlabel(xlabel="Priority (Absolute TD-error)", fontsize=18)
ax[1].set_xlabel(xlabel="Priority (Absolute TD-error)", fontsize=18)

ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()
# # read learner data
# with open(path_data_dir + path_learner_samples, 'r') as f:
#     pass

