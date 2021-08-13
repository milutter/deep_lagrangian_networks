import warnings
import numpy as np
import numpy.random as random
import torch
import jax.numpy as jnp

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


class ReplayMemory(object):
    def __init__(self, maximum_number_of_samples, minibatch_size, dim):

        # General Parameters:
        self._max_samples = maximum_number_of_samples
        self._minibatch_size = minibatch_size
        self._dim = dim
        self._data_idx = 0
        self._data_n = 0

        # Sampling:
        self._sampler_idx = 0
        self._order = None

        # Data Structure:
        self._data = []
        for i in range(len(dim)):
            self._data.append(np.empty((self._max_samples, ) + dim[i]))

    def __iter__(self):
        # Shuffle data and reset counter:
        self._order = np.random.permutation(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]

        # Reject Batches that have less samples:
        if batch_idx.size < self._minibatch_size:
            raise StopIteration()

        out = [x[batch_idx] for x in self._data]
        return out

    def add_samples(self, data):
        assert len(data) == len(self._data)

        # Add samples:
        add_idx = self._data_idx + np.arange(data[0].shape[0])
        add_idx = np.mod(add_idx, self._max_samples)

        for i in range(len(data)):
            self._data[i][add_idx] = data[i][:]

        # Update index:
        self._data_idx = np.mod(add_idx[-1] + 1, self._max_samples)
        self._data_n = min(self._data_n + data[0].shape[0], self._max_samples)

        # Clear excessive GPU Memory:
        del data

    def shuffle(self):
        self._order = np.random.permutation(self._data_idx)
        self._sampler_idx = 0

    def get_full_mem(self):
        out = [x[:self._data_n] for x in self._data]
        return out

    def not_empty(self):
        return self._data_n > 0


class PyTorchReplayMemory(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim, cuda):
        super(PyTorchReplayMemory, self).__init__(max_samples, minibatch_size, dim)

        self._cuda = cuda
        for i in range(len(dim)):
            self._data[i] = torch.empty((self._max_samples,) + dim[i])

            if self._cuda:
                self._data[i] = self._data[i].cuda()

    def add_samples(self, data):

        # Cast Input Data:
        tmp_data = []

        for i, x in enumerate(data):
            if isinstance(x, np.ndarray):
                x= torch.from_numpy(x).float()

            tmp_data.append(x.type_as(self._data[i]))
            # tmp_data[i] = tmp_data[i].type_as(self._data[i])

        # Add samples to the Replay Memory:
        super(PyTorchReplayMemory, self).add_samples(tmp_data)

class PyTorchTestMemory(PyTorchReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim, cuda):
        super(PyTorchTestMemory, self).__init__(max_samples, minibatch_size, dim, cuda)

    def __iter__(self):
        # Reset counter:
        self._order = np.arange(self._data_n)
        self._sampler_idx = 0
        return self

    def __next__(self):
        if self._order is None or self._sampler_idx >= self._order.size:
            raise StopIteration()

        tmp = self._sampler_idx
        self._sampler_idx += self._minibatch_size
        self._sampler_idx = min(self._sampler_idx, self._order.size)

        batch_idx = self._order[tmp:self._sampler_idx]
        out = [x[batch_idx] for x in self._data]
        return out


class RandomBuffer(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim_input, dim_output, enforce_max_batch_size=False):
        super(RandomBuffer, self).__init__(max_samples, minibatch_size, dim_input, dim_output)

        # Parameters:
        self._enforce_max_batch_size = enforce_max_batch_size

    def get_mini_batch(self):
        if self._data_n == 0 or (self._enforce_max_batch_size and self._data_n < self._minibatch_size):
            return None, None

        # Draw Random Mini-Batch
        idx = random.choice(self._data_n, min(self._minibatch_size, self._data_n))
        x_batch = np.array(self._x[idx], copy=True)
        y_batch = np.array(self._y[idx], copy=True)

        # Note Faster with indexing:
        # This should be faster with indexing, as less memory operation are used. However, this would significantly
        # increase implementation complexity. Therefore, this is currently not planned!

        # Remove Samples from Buffer:
        after_removal_x = np.delete(self._x, idx, 0)
        after_removal_y = np.delete(self._y, idx, 0)
        self._data_n -= idx.size

        if self._data_n > 0:
            self._x[0:self._data_n] = after_removal_x[0:self._data_n]
            self._y[0:self._data_n] = after_removal_y[0:self._data_n]

        return x_batch, y_batch

    def __next__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError


class RandomReplayMemory(ReplayMemory):
    def __init__(self, max_samples, minibatch_size, dim_input, dim_output):
        super(RandomReplayMemory, self).__init__(max_samples, minibatch_size, dim_input, dim_output)

    def add_samples(self, x, y):
        n_samples = x.shape[0]
        assert n_samples < self._max_samples

        # Add Samples in sequential order:
        add_idx = np.arange(self._data_n, min(self._data_n + n_samples, self._max_samples))

        self._x[add_idx] = x[:add_idx.size]
        self._y[add_idx] = y[:add_idx.size]

        self._data_n += add_idx.size
        assert self._data_n <= self._max_samples

        # Add samples in random order:
        random_add_idx = random.choice(self._data_n, n_samples - add_idx.size, replace=False)

        self._x[random_add_idx] = x[add_idx.size:]
        self._y[random_add_idx] = y[add_idx.size:]

    def get_mini_batch(self):
        raise RuntimeError

    def __next__(self):
        raise RuntimeError

    def __iter__(self):
        raise RuntimeError


