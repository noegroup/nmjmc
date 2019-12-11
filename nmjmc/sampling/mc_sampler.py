import numpy as np
from tqdm import tqdm, trange, tqdm_notebook


class MCSampler:
    def __init__(self, model, x0, dim, noise=0.01, stride=1, temperature=1.0):
        self.model = model
        self.x0 = x0
        self.dim = dim
        self.traj = []
        self.noise = noise
        self.stride = stride
        self.temperature = temperature
        self.beta = 1.0 / temperature

    def run(self, nsteps, reporter=None):
        x = self.x0
        traj = [self.x0]
        E = self.model.energy(self.x0.reshape(1, self.dim))
        steps = range(nsteps)
        if reporter is not None:
            if reporter == "notebook":
                steps = tqdm_notebook(steps)
            elif reporter == "script":
                steps = tqdm(steps)
            else:
                print("Invalid reporter. Using none.")
        for i in steps:
            y = x + self.noise * np.random.randn(self.dim)
            Enew = self.model.energy(y.reshape(1, self.dim))
            p_acc = np.exp(-self.beta * (Enew - E))
            if np.random.rand(1) < p_acc:
                x = y
                E = Enew
            if i % self.stride == 0:
                traj.append(x)
        self.traj = np.array(traj)

    def reset(self, x):
        self.x0 = x[0]
        self.traj = []
