import numpy as np

from panda_gym.envs.core_multi_task import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.drawer import Drawer  # Ensure this import path matches your project structure
from panda_gym.pybullet import PyBullet


class PandaDrawer(RobotTaskEnv):
    """Drawer task with Panda robot.

    In this environment, a Panda robot interacts with a drawer-with-handle.
    The drawer is placed on a table with its base located in the positive y-direction,
    and the handle protrudes in the negative y-direction.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        debug (bool, optional): Enable debug visuals. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, debug: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        # Create the Panda robot. Adjust the base_position as needed.
        robot = Panda(
            sim,
            block_gripper=False,
            debug=debug,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
            body_name=""
        )
        # Instantiate the Drawer task. By default, it places the drawer with its base in the positive y direction.
        task = Drawer(sim, debug=debug)
        super().__init__([robot], task)

