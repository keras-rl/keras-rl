import os

from rl.core import Agent


class PPOAgent(Agent):
    def __init__(self, actor, critic, memory, nb_actor=3, nb_steps=1000, epoch=5, **kwargs):
        super(Agent, self).__init__(**kwargs)

        # Parameters.
        self.nb_actor = nb_actor
        self.nb_steps = nb_steps
        self.epoch = epoch

        # Related objects.
        self.actor = actor
        self.critic = critic
        self.memory = memory

    def compile(self, optimizer, metrics=[]):
        # TODO
        pass

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.load_weights(actor_filepath)
        self.critic.load_weights(critic_filepath)

    def save_weights(self, filepath, overwrite=False):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.save_weights(critic_filepath, overwrite=overwrite)

    def forward(self, observation):
        # TODO
        pass

    @property
    def layers(self):
        return self.actor.layers[:] + self.critic.layers[:]

    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
        # TODO
