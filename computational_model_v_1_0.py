import numpy as np
import pandas as pd
import settings
import itertools as it

class Rule:

    def __init__(self, MAX_DIM_VALUE, MIN_DIM_VALUE, relevant_dimensions, thresholds, directions, positive_combinations, negative_combinations):

        """
        :param relevant_dimensions: np.array of relevant dimensions, sorted in an ascending order
        :param thresholds: np.array: threshold on each relevant dimension
        :param directions: np.array (logical): direction on each relevant dimension (which extreme to consider "large").
        'False' means reversed direction
        :param positive_combinations: set of tuples of len(relevant_dimensions).
        For example, a pair {(0,1), (1, 0)} would mean that either the value along the first relevant dimension
        should be big and the value along the second-small or vice versa.
        :param negative_combinations: set of tuples of len(relevant_dimensions) ## Not used currently
        """

        self.MAX_DIM_VALUE = MAX_DIM_VALUE
        self.MIN_DIM_VALUE = MIN_DIM_VALUE

        self.relevant_dimensions = relevant_dimensions
        self.thresholds = thresholds
        self.directions = directions
        self.positive_combinations = positive_combinations
        self.negative_combinations = negative_combinations

    def exact_classify(self, examples):
        """
        Classify every element in examples.

        :param examples - example matrix (one row per example)

        :return: a vector of size (examples.shape[0]) containing classification decisions
        """

        vals = examples[:, self.relevant_dimensions]

        vals[:, ~self.directions] = self.MAX_DIM_VALUE - vals[:, ~self.directions] + self.MIN_DIM_VALUE

        is_large = vals > self.thresholds

        res = np.zeros(examples.shape[0])

        for i in range(is_large.shape[0]):
            res[i] = tuple(is_large[i, :]) in self.positive_combinations

        return res

    def compare_to(self, other):

        """
        A helper function for manual rule examination. Compares the rule to another to see the differences.

        :param other: an instance of Rule class
        :return: a dictionary describing the differences between the rules.
        """

        res = {}
        set_dims_my = set(self.relevant_dimensions)
        set_dims_other = set(other.relevant_dimensions)

        same_dims = set_dims_my.intersection(set_dims_other)

        res["dims_1_in_2"] = set_dims_my.issubset(set_dims_other)
        res["dims_2_in_1"] = set_dims_my.issuperset(set_dims_other)

        res["dims_intersect_size"] = len(same_dims)
        res["same_type"] = len(set_dims_my) == len(set_dims_other)

        threshold_diffs = [np.abs(t_1 - t_2) for (i, t_1), (j, t_2) in it.product(enumerate(self.thresholds), enumerate(other.thresholds)) if self.relevant_dimensions[i] != other.relevant_dimensions[j]]
        res["diff_in_thresholds"] = np.mean(threshold_diffs) if threshold_diffs else 0

        dir_diffs = [d_1 != d_2 for (i, d_1), (j, d_2) in it.product(enumerate(self.directions), enumerate(other.directions)) if self.relevant_dimensions[i] != other.relevant_dimensions[j]]
        res["diff_in_directions"] = np.mean(dir_diffs) if dir_diffs else 0

        return res

    def __str__(self):

        """Generate a readable description of a rule"""

        #return "My relevant dimensions are {}.\nMy positive combinations: {}.\nMy thresholds: ".format(", ".join(list(self.relevant_dimensions)), self.positive_combinations, self.directions, self.thresholds)

        description = "I classify the following as A:\n"

        for comb in self.positive_combinations:

            description += ", ".join(["{} {} on dimension {}".format("below" if comb ^ direct else "above", t, d) for comb, d, t, direct in
                                                                                zip(comb, self.relevant_dimensions, self.thresholds, self.directions)])
            description += "\n"

        return description


class Simulation:

    def __init__(self, MAX_DIM_VALUE, MIN_DIM_VALUE, NUM_DIM):

        """
        :param MAX_DIM_VALUE: maximal possible value of every feature. Positive integer.
        :param MIN_DIM_VALUE: minimal possible value of every feature. Positive integer.
        :param NUM_DIM: number of dimensions (counting both relevant and irrelevant). Positive interer >= 0.
        """

        self.MAX_DIM_VALUE = MAX_DIM_VALUE
        self.MIN_DIM_VALUE = MIN_DIM_VALUE
        self.DIM_RANGE = MAX_DIM_VALUE - MIN_DIM_VALUE
        self.POSSIBLE_THRESHOLDS = np.arange(
            MAX_DIM_VALUE - MIN_DIM_VALUE) + MIN_DIM_VALUE + 0.5

        self.NUM_DIM = NUM_DIM

        ###########################
        ##### Rule generation #####
        ###########################

        # Below we run a simple combinatorial calculation of the number of possible rules.
        # Thresholds part reflects the number of possible threshold choices for relevant dimensions,
        # Relevant dimensions specifies the number of ways to pick dimensions relevant for a rule out of total
        # number of varying features.
        # Directions specify which direction to consider "increasing" for a specific feature.
        # For example, if we have a one-dimensional rule that says "classify fish as A if it has
        # a big value along the belly color feature dimension", we need to separately decide whether to consider
        # black or white to be "the big value" in this dimension.

        # Note: doubly reversed direction would preserve the 2D category we use therefore we only consider
        # 2 possible directions even when the rule has 2 relevant dimensions.

                             # Thresholds                 # Relevant dimensions                       # Directions
        num_2D_rules =    self.DIM_RANGE  ** 2     *    self.NUM_DIM * (self.NUM_DIM - 1) / 2       *    2
        num_1D_rules =    self.DIM_RANGE  ** 1     *        self.NUM_DIM                            *    2



        self.num_rules = num_1D_rules + num_2D_rules

        # Prepare the matrix to store likelihoods

        rules_2D = []

        for t_1, t_2 in it.product(self.POSSIBLE_THRESHOLDS, self.POSSIBLE_THRESHOLDS):

            for rel_dim_1, rel_dim_2 in it.combinations(range(self.NUM_DIM), 2):

                for d_1, d_2 in [[True, True], [True, False]]: # Both false gives an equivalent category

                    tmp = Rule(MAX_DIM_VALUE=self.MAX_DIM_VALUE,
                               MIN_DIM_VALUE=self.MIN_DIM_VALUE,
                               relevant_dimensions=np.array([rel_dim_1, rel_dim_2]),
                               thresholds=np.array([t_1, t_2]),
                               directions=np.array([d_1, d_2]),
                               positive_combinations={(1, 0), (0, 1)},
                               negative_combinations={(1, 1), (0, 0)})

                    rules_2D.append(tmp)

        rules_1D = []

        for t_1 in self.POSSIBLE_THRESHOLDS:
            for rel_dim_1 in range(NUM_DIM):
                for d_1 in [True, False]:
                    tmp = Rule(MAX_DIM_VALUE=self.MAX_DIM_VALUE,
                               MIN_DIM_VALUE=self.MIN_DIM_VALUE,
                               relevant_dimensions=np.array([rel_dim_1]),
                               thresholds=np.array([t_1]),
                               directions=np.array([d_1]),
                               positive_combinations={(1, )},
                               negative_combinations={(0, )})

                    rules_1D.append(tmp)


        assert len(rules_2D) == num_2D_rules and len(rules_1D) == num_1D_rules, "Wrong number of rules!"

        self.rules_1D = rules_1D
        self.rules_2D = rules_2D
        self.all_rules = list(it.chain(rules_1D, rules_2D))

        ##############################
        ##### Stimuli generation #####
        ##############################

        self.all_stimuli = np.array(list(it.product(range(self.MIN_DIM_VALUE, self.MAX_DIM_VALUE + self.MIN_DIM_VALUE), repeat=self.NUM_DIM)))

        # Verbal effort defines the confusion probability between rules. Easiest - relevant dimensions, rule type hardest - threshold
        # Could be naturally derived from the uncertainty reduction? I.e. a rule metric that, on average, allows to classify the largest numbers of examples correctly
        # - the best one.
        # Show that this metric favors similarity based on dimensions

        # Think about rule embedding techniques! Then - a map from sentence embeddings to rule embeddings!!! - write in the "future directions".

        # Could be based on editing distance, etc. Simple classification pattern-based distance, for now. // Study what does this metric favor
        self.rule_example_matrix = np.zeros((self.num_rules, self.all_stimuli.shape[0]))

        for i, r in enumerate(self.all_rules):
            if i % 100 == 0:
                print("Processed {} out of {} rules.".format(i, len(self.all_rules)))
            self.rule_example_matrix[i, :] = r.exact_classify(self.all_stimuli)

        self.rule_correlation_matrix = np.corrcoef(self.rule_example_matrix)

        self.neg_rule_example_matrix = 1 - self.rule_example_matrix

        true_pos = (self.rule_example_matrix).dot(self.rule_example_matrix.T)
        true_neg = (self.neg_rule_example_matrix).dot(self.neg_rule_example_matrix.T)

        self.acc_mat = (true_neg + true_pos) / np.full_like(true_pos, fill_value=self.rule_example_matrix.shape[1])

        ############################################################################################################
        #### Initialize rule priors. Currently, we arbitrarily set 2D rules to be 2 times less likely a-priori. ####
        ############################################################################################################

        rule_priors = np.zeros((len(self.all_rules),))
        rule_priors[0:len(rules_1D)] = 2
        rule_priors[len(rules_1D):] = 1
        rule_priors = rule_priors / np.sum(rule_priors)

        self.rule_priors = rule_priors

        ###################################################
        #### Initialize initial sampling probabilities ####
        ###################################################

        # Note how giving a positive vs negative example is represented.
        # We stack the example matrix twice, horizontally, with the intepretation
        # that the first half of the columns corresponds to giving a stimulus with a positive label,
        # while the second half of the columns - with a negative.
        #
        # I.e. if j < total_number_of_stimuli, initial_sampling_probabilities[i, j] would give the probability that
        # for the correct hypothesis i, we will sample the stimulus j and mark it as a positive example.
        #
        # If j > total_number_of_stimuli, initial_sampling_probabilities[i, j] gives the probability of selecting
        # the same stimulus (j % total_number_of_stimuli), but marking it as a negative example.

        if settings.STRONG_SAMPLING_PRIOR:  ## First choose whether to select a positive or a negative example, then sample conditionally.
            initial_example_probabilities_pos = self.rule_example_matrix / self.rule_example_matrix.sum(1, keepdims=1)
            initial_example_probabilities_neg = (1 - self.rule_example_matrix) / (1 - self.rule_example_matrix).sum(1, keepdims=1)

            initial_example_probabilities_pos = 0.95 * initial_example_probabilities_pos + 0.05 * initial_example_probabilities_neg  # 0.05 mislabeling probability
            initial_example_probabilities_neg = 0.05 * initial_example_probabilities_pos + 0.95 * initial_example_probabilities_neg

            initial_example_probabilities = np.hstack(
                [initial_example_probabilities_pos, initial_example_probabilities_neg])
            self.initial_example_probabilities = initial_example_probabilities / 2  # Strong sampling: flip a coin for positive vs negative example.

        else:  ## Weak sampling (first select an example at random, then mark it as positive or negative, according to the rule)

            initial_example_probabilities_pos = (0.95 * self.rule_example_matrix + 0.05 * (1 - self.rule_example_matrix)) / \
                                                self.rule_example_matrix.shape[1]
            initial_example_probabilities_neg = (0.05 * self.rule_example_matrix + 0.95 * (1 - self.rule_example_matrix)) / \
                                                self.rule_example_matrix.shape[1]

            self.initial_example_probabilities = np.hstack(
                [initial_example_probabilities_pos, initial_example_probabilities_neg])

    def get_rule_softmax_confusion_probs(self, rule_correlation_matrix, verbal_effort=1):
        """
        Constructs correlation-based confusion probabilities using verbal effort-scaled softmax

        :param rule_correlation_matrix Exhaustive correlation matrix between all possible rules
        :return a confusion matrix (element j in row i specifies the probability to receive rule j given that the teacher attempted to explain rule i)

        """

        rule_softmax_confusion_normalized_rows = np.exp(
            (rule_correlation_matrix - 1) * verbal_effort)  # -1 is for numerical stability
        rule_softmax_confusion_normalized_rows = rule_softmax_confusion_normalized_rows / np.sum(
            rule_softmax_confusion_normalized_rows, 1).reshape(-1, 1)

        return rule_softmax_confusion_normalized_rows

    def get_rules_data(self, verbal_effort=1):

        """Write information about rules into a csv file. Could be used to visualise the similarity metric"""


        rule_softmax_confusion_normalized_rows = self.get_rule_softmax_confusion_probs(self.rule_correlation_matrix, verbal_effort=verbal_effort)

        df = pd.DataFrame(columns=["rule_1_num", "rule_2_num", "dims_1_in_2", "dims_2_in_1", "dims_intersect_size",
                                   "diff_in_thresholds", "diff_in_directions", "same_type",
                                   "confusion_prob", "backward_confusion_prob", "correlation"])

        all_results = []

        for i, r_1 in enumerate(self.all_rules):
            for j, r_2 in enumerate(self.all_rules[i:len(self.all_rules)]):
                res = r_1.compare_to(r_2)
                res["rule_1_num"] = i
                res["rule_2_num"] = j
                res["confusion_prob"] = rule_softmax_confusion_normalized_rows[i, j]
                res["backward_confusion_prob"] = rule_softmax_confusion_normalized_rows[j, i] # Currently redundant
                res["correlation"] = self.rule_correlation_matrix[i, j]

                all_results.append(res)

            if i % 100 == 0:
                print("Processed {} rules out of {} for saving".format(i, len(self.all_rules)))


        df = df.append(all_results, ignore_index=True)
        with open("./rules_data.csv", "w") as f:
            df.to_csv(f)


    def calculate_student_posterior_given_data(self, verbal_data_sent, example_index, sampling_beliefs, hypothesis_beliefs, verbal_effort=1):
        """
        Recalculates teacher's beliefs about student's beliefs about hypotheses posterior after new data is received.

        :param verbal_data_sent: None or the index of the hypothesis intended for communication
        :param example_number: None or the index of the communicated example
        :param sampling_beliefs: a matix. Probability distributions over examples given hypotheses. p_t(d | h)
        :param hypothesis_beliefs: current student's probability distribution over hypotheses
        :return updated probability distribution over hypotheses.

        Note that in the case of verbal communication expected value is taken over noise. I.e. an expected updated distribution is returned.

        """

        ## Currently, we use a simplifying assumption that a teacher computes an expectation over the noise channel.
        ## Also we assume that there are is no noise in the in example communication channel.

        assert verbal_data_sent is None or example_index is None, "update beliefs received both verbal and example data"

        if verbal_data_sent is not None:

            softmax_confusion = self.get_rule_softmax_confusion_probs(self.rule_correlation_matrix, verbal_effort=verbal_effort)
            # Calculating expected student beliefs after verbal communication
            new_hypothesis_beliefs = softmax_confusion[verbal_data_sent, :] * hypothesis_beliefs

        else:

            new_hypothesis_beliefs = sampling_beliefs[:, example_index].reshape((-1,)) * hypothesis_beliefs

        new_hypothesis_beliefs = new_hypothesis_beliefs / np.sum(new_hypothesis_beliefs)

        return new_hypothesis_beliefs

    def calculate_p_learner_hypothesis_given_data(self, teacher_p_examples_given_rule, student_rule_beliefs):

        #Recalculates student's distributions over hypotheses given different potential datapoints

        # Potential hypothesis posteriors associated with giving different examples
        student_posterior_rule_beliefs = teacher_p_examples_given_rule * student_rule_beliefs.reshape(-1, 1)
        student_posterior_rule_beliefs = student_posterior_rule_beliefs / (student_posterior_rule_beliefs.sum(0, keepdims=True)  + 1e-9) # the last term is for numerical stability)

        return student_posterior_rule_beliefs

    def calculate_p_teacher_data_given_hypothesis(self, student_posterior_rule_beliefs, intensity=1):
        # New teacher sampling distribution
        new_teacher_p_examples_given_rule = student_posterior_rule_beliefs ** intensity
        new_teacher_p_examples_given_rule = new_teacher_p_examples_given_rule / (new_teacher_p_examples_given_rule.sum(1, keepdims=True) + 1e-9) # the last term is for numerical stability)

        return new_teacher_p_examples_given_rule

    def run_until_convergence(self, initial_sampling_probabilities, rule_priors, intensity, eps=1e-5, max_iter=100):

        sampling_probabilities = initial_sampling_probabilities.copy()

        for i in range(max_iter):

            student_p_learner_hypothesis_given_data = self.calculate_p_learner_hypothesis_given_data(sampling_probabilities, rule_priors)

            sampling_probabilities_new = self.calculate_p_teacher_data_given_hypothesis(student_p_learner_hypothesis_given_data, intensity)

            crit = np.sum(np.abs(sampling_probabilities - sampling_probabilities_new))
            print("Difference after iteration {}: {}".format(i+1, crit))

            if crit < eps:
                print("Converged after {} iterations.".format(i))
                return sampling_probabilities_new
            else:
                sampling_probabilities = sampling_probabilities_new

        print("Failed to converge.")
        return sampling_probabilities

    def calculate_expected_accuracy(self, rule_selection_probabilities, correct_rule_index):
        """

        :param rule_selection_probabilities: a probability distribution over all possible rules
        :param correct_rule_index: index of the correct rule
        :return: expected accuracy of a student under weak sampling
        """
        return self.acc_mat[:, correct_rule_index].dot(rule_selection_probabilities)



if __name__ == "__main__":



    ## Create a simulation object we will work with. Under default settings, it should not take long.
    ## Under some settings it may take some time, since it exhaustively creates all possible rules.

    S = Simulation(MAX_DIM_VALUE=7, MIN_DIM_VALUE=1, NUM_DIM=3) # create a simulation.

    ## If we want better understand how similarity metrics work, we could export the rules as csv.
    S.get_rules_data(verbal_effort=2)


    ## Now, let's explore the model behaviour.

    ## First we should set some constants affecting the simulation
    # (feel free to change these numbers).
    true_hypothesis_ind = 140
    verbal_effort = 1 # Please see the paper for the description of this parameter
    intensity = 2 # controls how peaked is the sampling distribution.
    # Changing it from 1 to infinity interpolates the teacher's behaviour.
    # Value of 1 corresponds to probability matching, value of infinity corresponds to deterministically picking the maximum.
    # Note that we used intensity of 1 for plot generation, since under some settings high intensities lead to numerical instabilities.


    ## Second, we could pick a random rule as a target (feel free to change this number as well)
    ## We could view information about this rule by printing it:
    print("The rule is:")
    print(S.all_rules[true_hypothesis_ind])

    ## Then, we could calculate sampling probabilities (obtain the solution to the recursive pair of equations)
    sampling_probabilities = S.run_until_convergence(S.initial_example_probabilities, S.rule_priors, intensity=2)

    # Now we could take a look at the best stimulus for explaining this hypothesis
    best_stimulus = np.argmax(sampling_probabilities[true_hypothesis_ind, :])

    # Check whether the best stimulus is given as a positive or as a negative example
    if best_stimulus >= S.all_stimuli.shape[0]:
        pos = False
    else:
        pos = True

    best_stimulus_ind = best_stimulus % S.all_stimuli.shape[0]


    # Now we could take a look at the stimulus
    print("For this rule, using the pedagogical model, it is best to give the stimulus {}, marking it as {}".format(
        S.all_stimuli[best_stimulus_ind, :], "positive" if pos else "negative."))

    best_stimulus_strong_sampling = np.argmax(S.initial_example_probabilities[true_hypothesis_ind, :])

    # Whether it the best stimulus is given a s positive or as a negative example
    if best_stimulus_strong_sampling >= S.all_stimuli.shape[0]:
        pos = False
    else:
        pos = True

    best_stimulus_strong_sampling_ind = best_stimulus_strong_sampling % S.all_stimuli.shape[0]

    # Now we could take a look at the stimulus
    print("According to {} sampling, it is best to give the stimulus {}, marking it as {}\n".format(
        "strong" if settings.STRONG_SAMPLING_PRIOR else "weak",
        S.all_stimuli[best_stimulus_strong_sampling_ind, :], "positive" if pos else "negative."))


    # Now, let's look at the posterior if we give the most informative example:

    post_model = S.calculate_student_posterior_given_data(None, best_stimulus, sampling_probabilities, S.rule_priors, verbal_effort=None)
    post_strong_sampling = S.calculate_student_posterior_given_data(None, best_stimulus, S.initial_example_probabilities, S.rule_priors, verbal_effort=None)

    print("Expected accuracy after one example.\nModel: {}\nStrong sampling: {}\n\n".format(
                         S.calculate_expected_accuracy(post_model, true_hypothesis_ind),
                         S.calculate_expected_accuracy(post_strong_sampling, true_hypothesis_ind)))

    # As we see, the difference is striking.
    # Depending on the target rule and other conditions, the difference may be more or less dramatic,
    # but it is never negligible.


    ##### Verbal communication ####
    # We could also take a look at the impact of verbal communication

    initial = S.rule_priors
    post_verbal = S.calculate_student_posterior_given_data(true_hypothesis_ind, None, sampling_probabilities, S.rule_priors, verbal_effort=verbal_effort)

    print("Expected accuracy\nInitial: {}\nWith verbal explanation: {} (verbal_effort: {})".format(
        S.calculate_expected_accuracy(initial, true_hypothesis_ind),
        S.calculate_expected_accuracy(post_verbal, true_hypothesis_ind),
        verbal_effort))


    ##### Recalculating the best stimulus #####
    ## Let's see whether verbal communication changes the most informative example we could choose and its effect:

    sampling_probabilities_after_verbal = S.run_until_convergence(sampling_probabilities, post_verbal, intensity=3)

    best_stimulus_post_verbal = np.argmax(sampling_probabilities_after_verbal[true_hypothesis_ind, :]) # Using previous sampling probabilities actually decreases the result

    # Check whether the best stimulus is given as a positive or as a negative example
    if best_stimulus_post_verbal >= S.all_stimuli.shape[0]:
        pos = False
    else:
        pos = True

    best_stimulus_post_verbal_ind = best_stimulus_post_verbal % S.all_stimuli.shape[0]

    # Now we could take a look at the stimulus
    print("After verbal communication it is best to give the stimulus {}, marking it as {}".format(
        S.all_stimuli[best_stimulus_post_verbal_ind, :], "positive" if pos else "negative."))


    post_verbal_and_stim = S.calculate_student_posterior_given_data(None, best_stimulus_post_verbal,
                                        sampling_probabilities_after_verbal, post_verbal, verbal_effort=None)

    print("Expected accuracy with verbal explanation and the new stimulus: {}".format(
        S.calculate_expected_accuracy(post_verbal_and_stim, true_hypothesis_ind)))


    ## Note that we used a very low verbal effort, which by itself barely gives any improvement in performance.
    ## Using only the verbal explanation gives an expected accuracy barely above chance (0.533)
    ## Nevertheless, combined with one example, It boosts the accuracy from 0.767 to 0.897, illustrating how using
    ## both channels of communication could lead to sharp increases in the overal efficiency.