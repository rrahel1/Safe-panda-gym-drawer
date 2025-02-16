import os

from gym.envs.registration import register

BASE_DIR = os.path.dirname(__file__)

with open(os.path.join(BASE_DIR, "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="PandaCubes{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaCubes",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaCleanPlate{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaCleanPlate",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaSponge{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaSponge",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaDrawer{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaDrawer",
            kwargs=kwargs,
            max_episode_steps=100000,
        )

        register(
            id="PandaCookSteak{}{}-v2".format(control_suffix, reward_suffix),
            entry_point="panda_gym.envs:PandaCookSteak",
            kwargs=kwargs,
            max_episode_steps=100000,
        )
