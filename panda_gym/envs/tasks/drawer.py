from typing import Any, Dict
import os
import numpy as np
import pybullet as p

from panda_gym import BASE_DIR
from panda_gym.envs.core_multi_task import Task
from panda_gym.pybullet import PyBullet

def get_body_unique_id(sim: PyBullet, target_body_name: str) -> int:
    """
    Iterates through all bodies in the simulation and returns the unique ID
    for the body whose name matches target_body_name.
    """
    num_bodies = sim.physics_client.getNumBodies()
    for i in range(num_bodies):
        # Get the body ID from the physics client using the index
        body_id = sim.physics_client.getBodyUniqueId(i)
        # getBodyInfo returns a tuple; index 1 is the body name as a byte string.
        body_info = sim.physics_client.getBodyInfo(body_id)
        if body_info[1].decode("utf-8") == target_body_name:
            return body_id
    raise ValueError(f"Body '{target_body_name}' not found.")

def get_joint_index(sim: PyBullet, body_id: int, joint_name: str) -> int:
    """
    Returns the joint index for the given joint name on a specified body.
    """
    num_joints = sim.physics_client.getNumJoints(body_id)
    for i in range(num_joints):
        joint_info = sim.physics_client.getJointInfo(body_id, i)
        # joint_info[1] is the joint name as a byte string.
        if joint_info[1].decode("utf-8") == joint_name:
            return i
    raise ValueError(f"Joint '{joint_name}' not found for body ID {body_id}.")

class Drawer(Task):
    def __init__(
        self,
        sim: PyBullet,
        debug: bool = False,
        drawer_position: np.ndarray = None,
    ) -> None:
        super().__init__(sim)
        self.debug = debug
        # Default position: on a table (table height 0.4) with the drawer base at 0.5 in z.
        if drawer_position is None:
            drawer_position = np.array([0.0, 0.4, 0.1])
        self.drawer_position = drawer_position
        self.drawer_name = "drawer_with_handle"

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        # Create the ground plane and table (same as in the Sponge task).
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.2, width=1.5, height=0.4, x_offset=-0.3)

        # Load the drawer URDF.
        self.sim.loadURDF(
            body_name=self.drawer_name,
            fileName=os.path.join(BASE_DIR, "assets/drawer/drawer_with_handle.urdf"),
            basePosition=self.drawer_position,
            useFixedBase=True,
            #flags = self.sim.URDF_USE_SELF_COLLISION,
        )
        # Now search for the body by name to get its unique ID.
        self.drawer_body_id = get_body_unique_id(self.sim, self.drawer_name)

        # Set lateral rolling and spinning friction for the handle
        self.sim.changeDynamics(self.drawer_body_id, 1, 10.0, 2.0, 2.0)


        if self.debug:
            self._create_visuals()

    def _create_visuals(self) -> None:
        # Optionally add a debug sphere at the handle.
        try:
            handle_state = self.sim.get_link_state(self.drawer_name, "handle")
            handle_position = np.array(handle_state[0])
            self.sim.create_sphere(
                body_name="handle_debug_sphere",
                radius=0.03,
                mass=0.0,
                position=handle_position,
                rgba_color=np.array([1.0, 0.0, 0.0, 0.5]),
                ghost=True,
            )
        except Exception:
            pass

    def get_obs(self) -> Dict[str, Any]:
        """
        Returns an observation dictionary containing the current poses of key parts
        of the drawer.
        """
        obs = {}
        # Base (fixed outer shell)
        obs["base"] = {
            "position": np.array(self.sim.get_base_position(self.drawer_name)),
            "orientation": np.array(self.sim.get_base_orientation(self.drawer_name)),
        }
        # The sliding drawer link.
        try:
            drawer_position = self.sim.get_link_position(self.drawer_name, 0)
            drawer_orientation = self.sim.get_link_orientation(self.drawer_name, 0)
            obs["drawer"] = {
                "position": np.array(drawer_position),
                "orientation": np.array(drawer_orientation),
            }
        except Exception:
            obs["drawer"] = {"position": np.zeros(3), "orientation": np.array([0, 0, 10, 1])}
        # The handle.
        try:
            # get_joint_index(self.sim, self.drawer_body_id, "handle")
            handle_position = self.sim.get_link_position(self.drawer_name, 1)
            handle_orientation = self.sim.get_link_orientation(self.drawer_name, 1)
            obs["handle"] = {
                "position": np.array(handle_position),
                "orientation": np.array(handle_orientation),
            }
        except Exception:
            obs["handle"] = {"position": np.zeros(3), "orientation": np.array([0, 0, 0, 1])}
        return obs

    def reset(self) -> None:
        """
        Resets the drawer to its initial configuration. This sets the drawer base pose
        and resets the sliding joint ("drawer_slide") to 0 (closed).
        """
        self.sim.set_base_pose(self.drawer_name, self.drawer_position, np.array([0, 0, 0, 1]))
        # Use our helper to get the joint index for "drawer_slide"
        body_id = self.drawer_body_id
        joint_index = get_joint_index(self.sim, body_id, "drawer_slide")
        # Reset the joint state to 0 using PyBullet's native API.
        p.resetJointState(body_id, joint_index, 0.0)

        if self.debug:
            self._update_visuals()

    def _update_visuals(self) -> None:
        try:
            handle_state = self.sim.get_link_state(self.drawer_name, "handle")
            handle_position = np.array(handle_state[0])
            self.sim.set_base_pose("handle_debug_sphere", handle_position, np.array([0, 0, 0, 1]))
        except Exception:
            pass

    def get_achieved_goal(self) -> np.ndarray:
        # Define the achieved goal as the current state of the "drawer_slide" joint.
        joint_val = self.sim.get_joint_state(self.drawer_name, "drawer_slide")
        return np.array([joint_val])

    def _sample_goal(self) -> np.ndarray:
        # Sample a random goal opening (up to 0.15 m).
        return np.array([self.np_random.uniform(0.0, 0.15)])

    def is_success(self) -> bool:
        achieved = self.get_achieved_goal()
        goal = self._sample_goal()  # (In practice the goal is fixed for an episode.)
        return np.all(np.abs(achieved - goal) < 0.01)

    def compute_cost(self) -> float:
        return 0.0

    def compute_reward(self) -> float:
        achieved = self.get_achieved_goal()
        goal = self._sample_goal()
        return -np.linalg.norm(achieved - goal)


if __name__ == "__main__":
    import gym
    import panda_gym
    env = gym.make('PandaDrawer-v2')
