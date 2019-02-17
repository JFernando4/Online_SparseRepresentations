import torch
import numpy as np

from .networks import TwoLayerFullyConnected


def compute_activation_map(network, granularity=100, layer=1, sample_size=10):
    """
    :param network: an instance of the class TwoLayerFullyConnected
    :param granularity: how fine should it be the partition on each direction
    :param layer: layer used to compute the activation map
    :param sample_size: size of the sample
    :return: random sample of activation maps of non-dead neurons
    """
    assert layer in [1, 2]
    assert isinstance(network, TwoLayerFullyConnected)
    assert network.fc1.in_features == 2                                     # The function admits only 2D state spaces

    layer_activation = network.first_layer_neurons if layer == 1 else network.second_layer_neurons
    num_neurons = network.fc1.out_features if layer == 1 else network.fc2.out_features

    partition_size = 2 / (granularity - 1)
    state_partition = np.arange(-1, 1 + partition_size, partition_size, dtype=np.float64)
    activation_maps = np.zeros((num_neurons, granularity, granularity), dtype=np.float64)

    for i in range(granularity):
        for j in range(granularity):
            temp_state = np.array((state_partition[i], state_partition[j]), dtype=np.float64)
            activation_maps[:, i, j] = layer_activation(temp_state).detach().numpy()

    activation_map_sample = sample_activation_maps(activation_maps, num_neurons, granularity, sample_size)

    return activation_map_sample


def sample_activation_maps(activation_maps, num_neurons=32, granularity=100, sample_size=10):
    """
    :param activation_maps: np array of shape (num_neurons, granularity, granularity)
    :param num_neurons: number of neurons in the nn layer corresponding to the activation map
    :param granularity: the granularity of the partition of the state space in each dimension
    :param sample_size
    :return: random sample of activation maps of non-dead neurons
    """
    sampled_activation_maps = np.zeros((sample_size, granularity, granularity), dtype=np.float64)
    indices = np.arange(num_neurons, dtype=np.int64)
    sampled_count = 0
    rejected_count = 0
    while sampled_count != sample_size:
        temp_sampled_idx = np.random.choice(indices, size=1)
        indices = np.delete(indices, np.where(temp_sampled_idx[0] == temp_sampled_idx))
        if np.sum(activation_maps[temp_sampled_idx]) == 0:
            rejected_count += 1
        else:
            sampled_activation_maps[sampled_count] = activation_maps[temp_sampled_idx]
            sampled_count += 1

        if (rejected_count + sampled_count) == num_neurons:
            print("There were too many dead neurons to fill the sample. Returning zeros instead.")
            break
    return sampled_activation_maps
