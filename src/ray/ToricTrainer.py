from ray.rllib.agents.dqn.apex import ApexTrainer, APEX_DEFAULT_CONFIG as APEX_CONFIG
from ray.rllib.utils import merge_dicts
import ToricPolicy

# TODO: Fix config
TORIC_DEFAULT_CONFIG = merge_dicts(
    DQN_CONFIG,  # see also the options in dqn.py, which are also supported
    {
        "optimizer": merge_dicts(
            DQN_CONFIG["optimizer"], {
                "max_weight_sync_delay": 400,
                "num_replay_buffer_shards": 4,
                "debug": False
            }),
        "n_step": 3,
        "num_gpus": 1,
        "num_workers": 32,
        "buffer_size": 2000000,
        "learning_starts": 50000,
        "train_batch_size": 512,
        "sample_batch_size": 50,
        "target_network_update_freq": 500000,
        "timesteps_per_iteration": 25000,
        "per_worker_exploration": True,
        "worker_side_prioritization": True,
        "min_iter_time_s": 30,
    },
)

TORIC_TRAINER_PROPERTIES = {
    "default_policy": ToricPolicy

}

ToricTrainer = ApexTrainer.with_updates("Toric", default_config=TORIC_DEFAULT_CONFIG, **TORIC_DEFAULT_CONFIG)

## Main would look like

# def env_creator(config):
#     return gym.make('toric-code-v0', config=config)

# env = env_creator({"size":SYSTEM_SIZE})

# config = {
#     "env":"toric-code-v0"
# }
# tune.run(ToricTrainer, config=config)

