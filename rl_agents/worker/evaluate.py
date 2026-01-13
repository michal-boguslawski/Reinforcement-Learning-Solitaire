

    def _eval_record_video(self, num_step: int) -> None:
        env = make_env(
            env_config=self.env_config,
            record=True,
            video_folder = f"logs/{self.experiment_name}/videos/num_step_{num_step}",
        )
        
        # Initialize evaluation metrics
        total_reward = 0.
        step_count = 0
        state, _ = env.reset()
        done = False
        truncated = False
        terminated = False
        action_output = None
        core_state = None

        # Start env inference
        self.agent.eval_mode()
        while not done:
            state = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
            action_output = self.agent.action(state=state, core_state=core_state, training=False, temperature=1.)
            action = action_output.action.squeeze(0)
            action = action.item() if self.action_space_type == "discrete" else action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(
                    action
                )
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state
            core_state = action_output.core_state
            step_count += 1
        env.close()
        self.agent.train_mode()
        logger.info(
            f"Step {num_step}: {step_count} steps, reward = {total_reward:.2f}, truncated = {truncated}, terminated = {terminated}"
        )
        if state.ndim < 3:
            logger.info("Evaluation stopped in state %s", state.round(2))
        if action_output and action_output.dist:
            try:
                if "covariance_matrix" in dir(action_output.dist.base_dist):  # type: ignore
                    cov = action_output.dist.base_dist.covariance_matrix[0].cpu().numpy()  # type: ignore
                else:
                    cov = action_output.dist.base_dist.stddev[0].cpu().numpy()  # type: ignore
                logger.info(f"Covariance matrix:\n{cov}")
            except AttributeError:
                pass