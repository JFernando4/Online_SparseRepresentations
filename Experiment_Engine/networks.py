import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TwoLayerFullyConnected(nn.Module):

    def __init__(self, input_dims=1, h1_dims=1, h2_dims=1, output_dims=1, network_type='relul-relu'):
        super(TwoLayerFullyConnected, self).__init__()
        # input_dims = state dimensions
        # h1_dims, h2_dims = number of hidden neurons in hidden layer 1 and hidden layer 2
        # output_layer_dims = number of actions
        self.fc1 = nn.Linear(input_dims, h1_dims, bias=True)
        self.fc2 = nn.Linear(h1_dims, h2_dims, bias=True)
        self.fc3 = nn.Linear(h2_dims, output_dims, bias=True)

        if network_type not in ['relu-relu', 'silu-silu', 'silu-dsilu']:
            raise ValueError("The network type has to be one of these options: 'relu-relu', 'silu-silu', 'silu-dsilu'")
        self.network_type = network_type

    def forward(self, x):
        x = to_variable(x)
        if self.network_type == 'relu-relu':
            # Let theta_i and b_i be the parameters and the bias of layer 1. Let ^T be the transpose.
            x = F.relu(self.fc1(x))     # Operation: z_1 = Relu(theta_1^T x + b_1)
            x = F.relu(self.fc2(x))     # Operation: z_2 = Relu(theta_2^T z_1 + b_2)
            x = self.fc3(x)             # Operation: z_3 = theta_3^T z_2 + b_3
        elif self.network_type == 'silu-silu':
            z1 = self.fc1(x)
            x = silu_gate(z1)
            z2 = self.fc2(x)
            x = silu_gate(z2)
            x = self.fc3(x)
        elif self.network_type == 'silu-dsilu':
            z1 = self.fc1(x)
            x = silu_gate(z1)
            z2 = self.fc2(x)
            x = dsilu_gate(z2)
            x = self.fc3(x)
        return x


def to_variable(x):
    if isinstance(x, torch.autograd.Variable):
        return x
    elif isinstance(x, np.ndarray):
        x = np.float64(x)
    x = torch.from_numpy(x).float()
    return torch.autograd.Variable(x)


def weight_init(m):
    # Initializes the weights of the linear layers according to a N(0, 2/nl), where nl is the size of the layers.
    # Biases are initialized with a value of zero.
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        std_dev = np.sqrt(2 / np.prod(size))
        m.weight.data.normal_(0, std_dev)
        m.bias.data.uniform_(0, 0)


def silu_gate(x):
    return x * torch.sigmoid(x)


def dsilu_gate(x):
    return torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))


if __name__ == "__main__":
    import numpy as np
    import argparse

    ###################
    " Argument Parser "
    ###################
    parser = argparse.ArgumentParser()
    parser.add_argument("-minibatch_size", action='store', default=np.int8(32))
    parser.add_argument("-lr", action='store', default=np.float64(0.001))
    parser.add_argument("-threshold", action='store', default=1e-4, type=float)
    parser.add_argument('-optimizer', action='store', default='adam', type=str, choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('-regularization', action='store', default='none', type=str, choices=['none', 'l1', 'l2'])
    parser.add_argument('-reg_factor', action='store', default=0.0001, type=float)
    parser.add_argument('-test_copy_params', action='store_true', default=False)
    args = parser.parse_args()

    ############################################
    " EXample: initializing the neural network "
    ############################################
    print("Creating Two Layer Fully Connected Network...")
    in_dims, hidden1_dims, hidden2_dims, num_actions = (2, 300, 300, 1)
    print("\tInput dimensions:", in_dims)
    print("\tHidden neurons in layer 1:", hidden1_dims)
    print("\tHidden neurons in layer 2:", hidden2_dims)
    print("\tNumber of actions:", num_actions, "\n")
    network = TwoLayerFullyConnected(input_dims=in_dims, h1_dims=hidden1_dims, h2_dims=hidden2_dims,
                                     output_dims=num_actions)
    network.apply(weight_init)  # initializes the weights (see weight_init function above)

    print("Printing Network...")
    print("\t", network, "\n")
        # Printing an example of an output
    print("Printing output for ten (1,2) inputs...")
    output = network.forward(torch.from_numpy(np.random.uniform(size=(10,2), low=0, high=1)).float())
    print("\t", output, "\n")

    ########################################
    " Example: training the neural network "
    ########################################
    print("Learning the function f(x,y) = x^2 + y^2...")
    print("Parameters:")
        # mini-batch size
    minibatch = args.minibatch_size
    print("\tmini-batch size:", minibatch)
        # learning rate
    learning_rate = args.lr
    print("\tlearning rate:", learning_rate)
        # optimizer
    if args.optimizer == 'adam': optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif args.optimzier == 'sgd': optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    elif args.optimizer == 'rmsprop': optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate)
    print("\toptimizer:", args.optimizer)
        # threshold
    threshold = args.threshold
    print("\tstopping criteria: loss <", threshold, "\n")

    current_loss = np.inf
    training_steps = 0
    while current_loss > threshold:

        inputs = np.random.uniform(size=(minibatch, 2), low=0, high=1)
        x = inputs[:,0]
        y = inputs[:,1]
        fxy = torch.from_numpy(x**2 + y**2).float()

        optimizer.zero_grad()
        prediction = network(torch.from_numpy(inputs).float()).view(1,-1)
        loss = torch.mean((prediction - fxy)**2)

        if args.regularization != 'none':
            if args.regularization == 'l1': reg_function = torch.abs
            elif args.regularization == 'l2': reg_function = lambda z: torch.pow(z, 2)
            reg_factor = args.reg_factor
            reg_loss = 0
            for name, param in network.named_parameters():
                reg_loss += torch.sum(reg_function(param))
            loss += reg_factor * reg_loss
        loss.backward()
        optimizer.step()

        training_steps += 1
        current_loss = loss.detach().numpy()
        if training_steps % 1000 == 0:
            print("\tIteration number:", training_steps)
            print("\tloss:", current_loss)
    print("The network finished training after", training_steps, "training steps.\n")

    print("Testing the network's predictions...")
    test_size = 5
    test_inputs = np.random.uniform(size=(test_size,2), low=0, high=1)
    print("The inputs are:")
    for i in range(test_size):
        print("\t", test_inputs[i])
    print("The predictions are:")
    test_labels = test_inputs[:, 0] ** 2 + test_inputs[:, 1] ** 2
    test_predictions = network(torch.from_numpy(test_inputs).float()).detach().numpy().flatten()
    for i in range(test_size):
        print("\tTrue Value:", np.round(test_labels[i], 3),
              "\tPrediction:", np.round(test_predictions[i], 3))

    #########################################################
    " Test: copying parameters from one network onto another"
    #########################################################
    if args.test_copy_params:
        print("\n\nInitializing another network with the same architecture...")
        network2 = TwoLayerFullyConnected(input_dims=in_dims, h1_dims=hidden1_dims, h2_dims=hidden2_dims,
                                         output_dims=num_actions)
        print(network2)

        print("Testing the new network's predictions...")
        test_inputs = np.random.uniform(size=(test_size, 2), low=0, high=1)
        test_labels = test_inputs[:, 0] ** 2 + test_inputs[:, 1] ** 2
        test_predictions = network2(torch.from_numpy(test_inputs).float()).detach().numpy().flatten()
        for i in range(test_size):
            print("\tTrue Value:", np.round(test_labels[i], 3),
                  "\tPrediction:", np.round(test_predictions[i], 3))

        print("Copying the parameters from the trained network onto the new network...")
        network2.load_state_dict(network.state_dict())

        print("Testing the new network's predictions again...")
        test_labels = test_inputs[:, 0] ** 2 + test_inputs[:, 1] ** 2
        test_predictions = network2(torch.from_numpy(test_inputs).float()).detach().numpy().flatten()
        for i in range(test_size):
            print("\tTrue Value:", np.round(test_labels[i], 3),
                  "\tPrediction:", np.round(test_predictions[i], 3))
