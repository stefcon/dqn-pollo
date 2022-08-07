from gym import wrappers, Env
import os


class EnvWrapper(object):
    def __init__(self, gym_env, frame_num=1, steps=None, film_video=True, run_name='test'):
        self.env = gym_env
        self.frame_num = frame_num
        if steps is not None:
            self.env._max_episode_steps = steps
        if film_video:
            self.env = wrappers.RecordVideo(
                self.env, os.path.join('./video_folder/'+run_name, self.env.spec.id), 
                episode_trigger= lambda x: x % 50 == 0, new_step_api=True)

    def state_size(self):
        return self.env.observation_space.shape[0]

    def action_size(self):
        if hasattr(self.env.action_space, 'n'):
            return self.env.action_space.n
        else:
            return self.env.action_space.high.shape[0]

    def action_limit(self):
        if hasattr(self.env.action_space, 'high'):
            return self.env.action_space.high[0]
        else:
            return 1

    # Displays current state
    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        # Repeat the action across frame_num frames
        for _ in range(self.frame_num):
            # Deprecated; new return format: next_state, reward, done, truncated, info
            next_state, reward, done, truncated, _ = self.env.step(action)
        return next_state, reward, done or truncated