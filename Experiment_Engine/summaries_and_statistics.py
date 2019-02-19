from scipy.stats import t, chi2   # Faster than importing scipy.stats
import numpy as np
import os
import pickle


####################################
# Statistical comparison functions #
####################################
# Confidence interval for sample average
def compute_tdist_confidence_interval(sample_mean, sample_std, alpha_proportion, sample_size):
    if sample_size <= 1:
        return None, None, None

    dof = sample_size - 1
    t_dist = t(df=dof)  # from scipy.stats
    tdist_factor = t_dist.ppf(1 - alpha_proportion / 2)

    sqrt_inverse_sample_size = np.sqrt(1 / sample_size)
    me = sample_std * tdist_factor * sqrt_inverse_sample_size
    upper_bound = sample_mean + me
    lower_bound = sample_mean - me

    return upper_bound, lower_bound, me


# Confidence interval for sample variance
def compute_chi2dist_confidence_interval(sample_variance, proportion, sample_size):
    dof = sample_size - 1
    chi2_dist = chi2(dof)

    upper_proportion = 1 - (proportion / 2)
    lower_proportion = proportion / 2

    upper_chi2_value = chi2_dist.ppf(upper_proportion)
    lower_chi2_value = chi2_dist.ppf(lower_proportion)

    # the lower and upper chi2_values get inverted because we're bounding sigma^2 in the equation:
    # chi2.ppf(proportion/2) < (n-1) s^2 / sigma^2 < chi2.ppf(1 - proportion/2)
    upper_bound_ci = (dof * sample_variance) / lower_chi2_value
    lower_bound_ci = (dof * sample_variance) / upper_chi2_value

    return upper_bound_ci, lower_bound_ci


# Welch's Test for difference in sample averages
def compute_welchs_test(mean1, std1, sample_size1, mean2, std2, sample_size2):
    mean_difference = mean1 - mean2
    sum_of_stds = ((std1 ** 2) / sample_size1) + ((std2 ** 2) / sample_size2)
    tdenominator = np.sqrt(sum_of_stds)
    if tdenominator == 0:
        tvalue = 0; p_value = 0
        if mean1 > mean2:
            tvalue = np.inf
            p_value = 1
        elif mean2 > mean1:
            tvalue = -np.inf
            p_value = 0
        elif mean2 == mean1:
            tvalue = 0
            p_value = 0.5
    else:
        tvalue = mean_difference / tdenominator

        dof_numerator = sum_of_stds ** 2
        dof_sample1 = sample_size1 - 1
        dof_sample2 = sample_size2 - 1
        dof_denominator = ((std1 ** 4) / (dof_sample1 * (sample_size2 ** 2))) + \
                          ((std2 ** 4) / (dof_sample2 * (sample_size2 ** 2)))
        dof = dof_numerator / dof_denominator

        t_dist = t(df=dof)  # from scipy.stats
        p_value = t_dist.pdf(tvalue)
    return tvalue, p_value


###############################################
# Functions for loading and comparing results #
###############################################
class ParameterCombinationSummary:
    """ Loads the results from all the runs of one particular parameter combination """
    def __init__(self, param_comb_path, param_comb_name, parameter_names, summary_names,
                 performance_measure_name='return_per_episode', number_of_episodes=500):
        """
        :param param_comb_path: path to the directory containing all the runs. The directory has the form:
                                ['agent_1', 'agent_2', ..., 'agent_{sample_size}']
        :param param_comb_name: name of the directory, e.g. LearningRate0.01_BufferSize10000_Freq10
        :param parameter_names: name of the parameters. Each method has a specified set of parameters that is included
                                in the directory name (i.e., in param_comb_name).
        :param summary_names: names of the summaries of the run, e.g. return_per_episode and cumulative_loss_per_episode
        :param performance_measure_name: name of the summary used as a performance measure
        :param number_of_episodes: the number of episodes of each run
        """
        assert isinstance(param_comb_name, str)
        # extracting the parameter values
        self.param_comb_name = param_comb_name
        self.param_comb_path = param_comb_path
        self.parameter_values = self.extract_method_parameter_values(parameter_names)
        # extracting summary of each run
        self.runs = []
        for agent_name in os.listdir(self.param_comb_path):
            agent = self.extract_agent_results(agent_name)
            if agent is not None:
                self.runs.append(agent)
        # storing summaries into one single numpy array
        self.summary_names = summary_names
        self.summaries = {}
        self.sample_size = len(self.runs)
        for name in self.summary_names:
            self.summaries[name] = np.zeros((self.sample_size, number_of_episodes), dtype=np.float64)
            for i in range(self.sample_size):
                assert 'summary' in self.runs[i].keys()
                self.summaries[name][i] += self.runs[i]['summary'][name]
        # computing the performance measure with confidence intervals
        self.perf_meas = performance_measure_name
        mean_perf_over_episodes = np.average(self.summaries[self.perf_meas], axis=1)
        self.mean_perf = np.average(mean_perf_over_episodes)
        self.stddev_perf = np.std(mean_perf_over_episodes, ddof=1)
        _, _, self.me = compute_tdist_confidence_interval(self.mean_perf, self.stddev_perf, 0.05, self.sample_size)
        # sorting the agents according tot he highest mean performance measure across episodes
        self.sort_agents()

    def extract_method_parameter_values(self, parameter_names):
        # extract the parameter values from the parameter_comb_name
        # this might be removed since it has no use so far
        split_method_name = self.param_comb_name.split('_')
        method_parameters = {}
        for parameter_name, name_value_string in zip(parameter_names, split_method_name):
            parameter_value = name_value_string.split(parameter_name)[1]
            method_parameters[parameter_name] = parameter_value
        return method_parameters

    def extract_agent_results(self, agent_name):
        # extracts the agents (run) results: summary, config (parameter configuration), and path to the weights of the
        # neural network
        agent_dir_path = os.path.join(self.param_comb_path, agent_name)
        if 'config.p' not in os.listdir(agent_dir_path) or 'summary.p' not in os.listdir(agent_dir_path):
            print('No config file or summary file found for', agent_name,
                  'for method ' + self.param_comb_name + '.')
            return None
        agent = {}
        with open(os.path.join(agent_dir_path, 'config.p'), mode='rb') as config_file:
            # config is stored so that we can retroactively go back and check the parameters of the run if necessary
            agent['config'] = pickle.load(config_file)
        with open(os.path.join(agent_dir_path, 'summary.p'), mode='rb') as summary_file:
            agent['summary'] = pickle.load(summary_file)
        agent_weights_path = os.path.join(agent_dir_path, 'network_weights_500episodes.pt')
        agent['weights_path'] = agent_weights_path
        return agent

    def sort_agents(self):
        # sort agents according to the mean performance measure across all the episodes
        sorted_runs = []
        performances = []
        for i in range(self.sample_size):
            assert 'summary' in self.runs[i].keys()
            assert isinstance(self.runs[i]['summary'], dict)
            temp_performance = np.average(self.runs[i]['summary'][self.perf_meas])
            idx = 1
            while idx <= len(sorted_runs):
                if temp_performance > performances[-idx]:
                    insert_idx = len(sorted_runs) - idx + 1
                    performances.insert(insert_idx, temp_performance)
                    sorted_runs.insert(insert_idx, self.runs[i])
                    break
                else:
                    idx += 1
            if idx > len(sorted_runs):
                performances.insert(0, temp_performance)
                sorted_runs.insert(idx, self.runs[i])
        self.runs = sorted_runs

    def print_summary(self, round_dec=2, sep=''):
        print("Parameter Combination name:", self.param_comb_name)
        print("The performance measure is:", self.perf_meas)
        print("\tSample size:", self.sample_size)
        print("\tSample average:", np.round(self.mean_perf, round_dec))
        print("\tSample standard deviation:", np.round(self.stddev_perf, round_dec))
        print("\tMargin of error:", np.round(self.me, round_dec))
        uci = np.round(self.mean_perf + self.me, round_dec)
        lci = np.round(self.mean_perf - self.me, round_dec)
        print("\tLower and upper 95% confidence interval bounds:", "(" + str(lci) + ", " + str(uci) + ")")
        if sep != "":
            print(sep)


class MethodResults:
    """
    Stores a list of ParameterCombinationResults and finds the best parameter combination according to the 95% C.I.
    """
    def __init__(self, method_name):
        """
        :param method_name: e.g. experience_replay, sigmoid_weighted_units
        """
        self.method_name = method_name
        self.top_param_comb = []
        self.all_param_comb = []

    def append(self, param_comb_results):
        # Appends parameter combination results to the all_param_comb list.
        # Additionally, it compares the new item to the parameter combinations results in top_param_comb and updates
        # the list according to the 95% CI's
        assert isinstance(param_comb_results, ParameterCombinationSummary)
        self.all_param_comb.append(param_comb_results)
        append = True
        popped = 0
        for i in range(len(self.top_param_comb)):
            idx = i-popped
            if len(self.top_param_comb) != 0:
                assert isinstance(self.top_param_comb[idx], ParameterCombinationSummary)
                top_mean = self.top_param_comb[idx].mean_perf
                top_me = self.top_param_comb[idx].me
                top_uci, top_lci = (top_mean + top_me, top_mean - top_me)
                candidate_mean = param_comb_results.mean_perf
                candidate_me = param_comb_results.me
                candidate_uci, candidate_lci = (candidate_mean + candidate_me, candidate_mean - candidate_me)
                if candidate_uci < top_lci:
                    append = False
                    break   # performance is statistically significantly lower
                if candidate_lci > top_uci:
                    self.top_param_comb.pop(idx)  # performance is statistically significantly higher
                    popped += 1
                if (top_lci <= candidate_uci <= top_uci) or (top_lci <= candidate_lci <= top_uci):
                    pass    # performance is not statistically significant
        if append:
            self.top_param_comb.append(param_comb_results)

    def print_top_results(self):
        for results in self.top_param_comb:
            assert isinstance(results, ParameterCombinationSummary)
            results.print_summary(2)

    def print_all_results(self):
        for results in self.all_param_comb:
            assert isinstance(results, ParameterCombinationSummary)
            results.print_summary(sep='\n')
