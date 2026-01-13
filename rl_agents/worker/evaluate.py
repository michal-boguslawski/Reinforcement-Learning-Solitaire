import os
import torch as T

from agent.factories import get_policy
from envs.factories import make_vec
from envs.utils import get_env_vec_details
from config.config import ExperimentConfig
from network.model import RLModel
from .utils import prepare_action_for_env


os.environ["MUJOCO_GL"] = "egl" if T.cuda.is_available() else "osmesa"


if __name__ == "__main__":
    # policy_name = "ppo"
    config_instance = ExperimentConfig()
    config = config_instance.get_config()
    experiment_name = config["experiment_name"]
    
    envs = make_vec(
        id=config["env_kwargs"]["id"],
        num_envs=1,
        training=False,
        record=True,
        video_folder=f"logs/{experiment_name}/videos",
    )

    env_details = get_env_vec_details(envs)

    network_config = config["network"]
    network_kwargs = network_config.get("kwargs", {})
    network = RLModel(
        input_shape=env_details.state_dim,
        num_actions=env_details.action_dim,
        low=env_details.action_low,
        high=env_details.action_high,
        **network_kwargs
    )

    policy_config = config["policy"]
    policy_type = policy_config.get("type", "a2c")
    policy_kwargs = policy_config.get("kwargs", {})

    agent = get_policy(
        policy_type=policy_type,
        network=network,
        action_space_type=env_details.action_space_type,
        policy_kwargs=policy_kwargs
    )

    agent.load_weights("/app/rl_agents/logs/CarRacing-PPO-GRU/model.pt")

    state, _ = envs.reset()
    core_state = None

    for _ in range(1000):
        action_output = agent.action(state=state, core_state=core_state, temperature=0.1)
        action = action_output.action

        env_action = prepare_action_for_env(action, env_details.action_space_type)
        state, reward, terminated, truncated, info = envs.step(action)
        core_state = action_output.core_state

    envs.close()

    # def _eval_record_video(self, num_step: int) -> None:
    #     env = make_env(
    #         env_config=self.env_config,
    #         record=True,
    #         video_folder = f"logs/{self.experiment_name}/videos/num_step_{num_step}",
    #     )
        
    #     # Initialize evaluation metrics
    #     total_reward = 0.
    #     step_count = 0
    #     state, _ = env.reset()
    #     done = False
    #     truncated = False
    #     terminated = False
    #     action_output = None
    #     core_state = None

    #     # Start env inference
    #     self.agent.eval_mode()
    #     while not done:
    #         state = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
    #         action_output = self.agent.action(state=state, core_state=core_state, training=False, temperature=1.)
    #         action = action_output.action.squeeze(0)
    #         action = action.item() if self.action_space_type == "discrete" else action.detach().cpu().numpy()
    #         next_state, reward, terminated, truncated, _ = env.step(
    #                 action
    #             )
    #         done = terminated or truncated
    #         total_reward += float(reward)
    #         state = next_state
    #         core_state = action_output.core_state
    #         step_count += 1
    #     env.close()
    #     self.agent.train_mode()
    #     logger.info(
    #         f"Step {num_step}: {step_count} steps, reward = {total_reward:.2f}, truncated = {truncated}, terminated = {terminated}"
    #     )
    #     if state.ndim < 3:
    #         logger.info("Evaluation stopped in state %s", state.round(2))
    #     if action_output and action_output.dist:
    #         try:
    #             if "covariance_matrix" in dir(action_output.dist.base_dist):  # type: ignore
    #                 cov = action_output.dist.base_dist.covariance_matrix[0].cpu().numpy()  # type: ignore
    #             else:
    #                 cov = action_output.dist.base_dist.stddev[0].cpu().numpy()  # type: ignore
    #             logger.info(f"Covariance matrix:\n{cov}")
    #         except AttributeError:
    #             pass