from numpy import *
from alife.rl.agent import Agent


def sigmoid(a):
    """ The sigmoid activation function """
    # Keep some exploration open, by clipping the values going into the sigmoid
    a = clip(a, -3, 3)
    return 1. / (1. + exp(-a))


class Reinforce(Agent):
    """
        A Policy-Search Method.
    """

    def __init__(self, obs_space, action_space):
        """
            Init.


            Parameters
            ----------

            obs_space : BugSpace
                observation space
            action_space : BugSpace
                action space

        """
        self.state_space = obs_space
        self.action_space = action_space

        # Set random weights (+ bias)
        self.dim_state = obs_space.shape[0]
        self.w = random.randn(self.dim_state + 1) * 0.1

        # The step size for the gradient
        self.alpha = 0.1

        # To store an episode
        T = 20  # length of episode
        self.episode = zeros((T, 1 + self.dim_state + 1))  # for storing (a,s,r)
        self.t = 0  # for counting

        # Discretize the action space
        # namely, we make two possible actions: turn left or right
        # (in both cases, move always at a fixed speed).
        speed = 3.
        self.discrete2continuous = array([
            [+pi / 4., speed],
            [-pi / 4., speed],
        ])

    def act(self, obs, reward, done=False):
        """
            Act.

            Parameters
            ----------

            obs : numpy array
                the state observation
            reward : float
                the reward obtained in this state

            Returns
            -------

            numpy array
                the action to take
        """
        # Save some info to a episode
        self.episode[self.t, 0:self.dim_state] = obs
        self.episode[self.t, -1] = reward

        # Add the bias term (for our model)
        x = ones(self.dim_state + 1)
        x[1:] = obs

        # End of episode ?
        T = len(self.episode)
        self.t = self.t + 1
        if self.t >= T:
            # =======================================
            # WE ARE AT THE END OF THE EPISODE 
            # for each t = 1,...,T of the episode
            #   TODO get an unbiased sample
            #   TODO calculate the gradient
            #   TODO ascend the gradient by alpha
            # =======================================
            self.t = 0

        # Sigmoid network simple linear reflex,
        p_action_1 = sigmoid(dot(x, self.w))  # probability of taking a=1 | obs
        a = (p_action_1 > random.rand()) * 1  # decide a=? stochastically

        # Save some info to a episode
        self.episode[self.t, self.dim_state] = a

        # Return the action to take
        return self.discrete2continuous[a]

    def __str__(self):
        """Return a string representation (e.g., a label) for this agent """
        # This will appear as label when we click on the bug in ALife
        return ("RF. alpha=%3.2f" % (self.alpha))

    def spawn_copy(self):
        """
            Spawn.


            Returns
            -------
            
            A new copy (child) of this agent, [optionally] based on this one (the parent).
        """
        b = Reinforce(self.state_space, self.action_space)
        b.alpha = self.alpha
        # (We could consider this the end of an episode, and do the gradient
        # update here -- but in this case we simply create a new instance).
        return b
