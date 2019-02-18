import torch
import numpy as np

from .networks import TwoLayerFullyConnected


def compute_activation_map(network, granularity=100, layer=1):
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

    return eliminate_dead_neuron_maps(activation_maps)


def eliminate_dead_neuron_maps(activation_maps):
    """
    :param activation_maps: the activation maps of each neuron as computed by the function compute_activation_map.
                            Shape of activation_maps: (num_neurons, granularity, granularity)
                                num_neurons: number of neurons in the layer
                                granularity: number of partitions of each dimension of the state space
    :return: a numpy array of type np.float64 with dimensions (num_of_alive_neurons, granularity, granularity)
             which correspond to the activation maps of the alive neurons
    """
    assert isinstance(activation_maps, np.ndarray)
    assert len(activation_maps.shape) == 3
    indices = []
    number_of_neurons = activation_maps.shape[0]
    for i in range(number_of_neurons):
        total_activation = np.sum(activation_maps[i])
        if total_activation != 0:
            indices.append(i)
    indices = np.array(indices, dtype=np.int64)
    alive_neuron_maps = activation_maps[indices, :, :]
    return alive_neuron_maps


def sample_activation_maps(activation_maps, sample_size=10):
    alive_neuron_maps = eliminate_dead_neuron_maps(activation_maps)
    if sample_size > alive_neuron_maps.shape[0]:
        print("Not enough alive neurons for a sample size of " + str(sample_size) + ".")
        return alive_neuron_maps
    else:
        sampled_indices = np.random.choice(alive_neuron_maps.shape[0], size=sample_size, replace=False)
        return alive_neuron_maps[sampled_indices, :, :]


def compute_instance_sparsity(activation_maps):
    assert isinstance(activation_maps, np.ndarray)
    sample_size = activation_maps.shape[0]
    positive_activations = np.int64((activation_maps > 0))
    active_neurons = np.sum(positive_activations, axis=0)
    percentage_active_neurons = (active_neurons / sample_size) * 100
    return active_neurons, percentage_active_neurons.flatten()


