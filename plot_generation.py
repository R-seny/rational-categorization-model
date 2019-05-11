from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})

from computational_model_v_1_0 import *

def get_example_trajectory(S, num_examples, true_hypothesis_ind, intensity):

    examples = [None]
    accuracies = [0.5]

    sampling_probabilities = S.initial_example_probabilities.copy()
    prior = S.rule_priors.copy()

    for i in range(num_examples):
        sampling_probabilities = S.run_until_convergence(sampling_probabilities, prior, intensity=intensity)

        best_stimulus = np.argmax(sampling_probabilities[true_hypothesis_ind, :])

        examples.append(best_stimulus)
        post_model = S.calculate_student_posterior_given_data(None, best_stimulus, sampling_probabilities,
                                                              prior, verbal_effort=None)

        accuracies.append(S.calculate_expected_accuracy(post_model, true_hypothesis_ind))

        prior = post_model.copy()

    return examples, accuracies

def get_verbal_effort_trajectory(S, steps, true_hypothesis_ind, min_ve, max_ve):


    ves = np.linspace(min_ve, max_ve, steps)
    accuracies = []

    for ve in ves:
        post_model = S.calculate_student_posterior_given_data(true_hypothesis_ind, None, S.initial_example_probabilities,
                                                              S.rule_priors, verbal_effort=ve)

        accuracies.append(S.calculate_expected_accuracy(post_model, true_hypothesis_ind))

    return examples, accuracies


if __name__ == "__main__":

    num_examples = 10
    num_ve = 21

    all_examples = []
    all_accuracies = []
    all_accuracies_v = []

    max_dims = [4, 6, 8] # Note that it is preferable to use only even or only odd numbers here.
    # It happens because even perceptual confusabilities have a specific "mid threshold" value, as was in our experiments.
    # Odd perceptual confusabilities have two "mid threshold" values, therefore the accuracies differ between these cases.

    for max_dim in max_dims:

        S = Simulation(MAX_DIM_VALUE=max_dim, MIN_DIM_VALUE=1, NUM_DIM=4)  # create a simulation.

        ## Find rules with thresholds near the center (similar to what we had in the empirical study)
        med_thresholds = set([S.POSSIBLE_THRESHOLDS[len(S.POSSIBLE_THRESHOLDS) // 2 - 1],
                              S.POSSIBLE_THRESHOLDS[len(S.POSSIBLE_THRESHOLDS) // 2]]) if len(S.POSSIBLE_THRESHOLDS) % 2 == 0 else set([S.POSSIBLE_THRESHOLDS[len(S.POSSIBLE_THRESHOLDS) // 2]])

        threshold_med_rules = set([i for i, r in enumerate(S.all_rules) if med_thresholds.issuperset(set(r.thresholds))])
        threshold_med_1D = threshold_med_rules.intersection(set(range(len(S.rules_1D))))
        threshold_med_2D = threshold_med_rules.intersection(set(range(len(S.rules_1D), len(S.all_rules))))

        threshold_med_1D = sorted(list(threshold_med_1D))
        threshold_med_2D = sorted(list(threshold_med_2D))

        #true_hypothesis_ind = 140
        true_hypothesis_ind = threshold_med_2D[0]

        verbal_effort_min = 0  # Please see the paper for the description of this parameter
        verbal_effort_max = 20  # Please see the paper for the description of this parameter

        intensity = 1.1  # controls how peaked is the sampling distribution. Intensity of 1.0 gives the most numerically stable results.

        # Unfortunately, contrary to results in (Shafto 2014), we find that this parameter significantly affects the model performance.
        #
        # In general, setting the value at least slightly higher than 1.0 introduces important nonlinearities into the equation,
        # and produces the results we report. Most values slightly above 1 should work (e.g. 1.05, 1.1, 1.2). Higher values (e.g. 1.5) may lead to
        # numerical instabilities and convergence issues, while very small deviations from one (e.g. 1.01) may lead to slow convergence.
        # In general, high values lead to distorted results when either the teacher and the student converge on some very peaked distribution (akin to a code),
        # leading to unrealistically high accuracies achieved through communicating single example, or they fail to converge.
        #
        # Lastly, it is important to mention that the value of exactly one leads to qualitatively different model performance when both the number of
        # dimensions and the perceptual confusability matter to a comparable extent for exemplar channel of communication.

        print("### Getting examples trajectory ####")
        print(len(S.all_rules))
        print(S.all_stimuli.shape)
        examples, accuracies = get_example_trajectory(S, num_examples, true_hypothesis_ind, intensity)

        steps, accuracies_v = get_verbal_effort_trajectory(S, num_ve, true_hypothesis_ind, verbal_effort_min, verbal_effort_max)


        all_examples.append(examples)
        all_accuracies.append(accuracies)
        all_accuracies_v.append(accuracies_v)


    plt.figure()
    #plt.title("Number of examples vs accuracy for different perceptual confusabilities")
    plt.xlabel("Number of examples")
    plt.ylabel("Expected accuracy")
    plt.tight_layout()
    for i, max_dim in enumerate(max_dims):
        plt.plot(range(num_examples + 1), all_accuracies[i], label="Confusability {}".format(max_dim))
    plt.legend()


    plt.figure()
    plt.xlabel("Verbal effort")
    plt.ylabel("Expected accuracy")
    plt.tight_layout()
    #plt.title("Verbal effort vs accuracy for different perceptual confusabilities")
    for i, max_dim in enumerate(max_dims):
        plt.plot(np.linspace(verbal_effort_min, verbal_effort_max, num_ve), all_accuracies_v[i], label="Confusability {}".format(max_dim))
    plt.legend()


    ### Studying the effect of the number of dimensions

    fixed_max_dim = 6 # Here we pick one (approximately average) fixed perceptual confusability value

    num_dims = [2, 3, 4, 5]

    all_examples = []
    all_accuracies = []
    all_accuracies_v = []

    for num_dim in num_dims:
        S = Simulation(MAX_DIM_VALUE=fixed_max_dim, MIN_DIM_VALUE=1, NUM_DIM=num_dim)  # create a simulation.

        ## Find rules with thresholds near the center (similar to what we had in the empirical study)
        med_thresholds = set([S.POSSIBLE_THRESHOLDS[len(S.POSSIBLE_THRESHOLDS) // 2 - 1],
                              S.POSSIBLE_THRESHOLDS[len(S.POSSIBLE_THRESHOLDS) // 2]]) if len(
            S.POSSIBLE_THRESHOLDS) % 2 == 0 else set([S.POSSIBLE_THRESHOLDS[len(S.POSSIBLE_THRESHOLDS) // 2]])

        threshold_med_rules = set(
            [i for i, r in enumerate(S.all_rules) if med_thresholds.issuperset(set(r.thresholds))])
        threshold_med_1D = threshold_med_rules.intersection(set(range(len(S.rules_1D))))
        threshold_med_2D = threshold_med_rules.intersection(set(range(len(S.rules_1D), len(S.all_rules))))

        threshold_med_1D = sorted(list(threshold_med_1D))
        threshold_med_2D = sorted(list(threshold_med_2D))

        # true_hypothesis_ind = 140
        true_hypothesis_ind = threshold_med_2D[0]

        verbal_effort_min = 0  # Please see the paper for the description of this parameter
        verbal_effort_max = 20  # Please see the paper for the description of this parameter

        examples, accuracies = get_example_trajectory(S, num_examples, true_hypothesis_ind, intensity)

        steps, accuracies_v = get_verbal_effort_trajectory(S, num_ve, true_hypothesis_ind, verbal_effort_min,
                                                           verbal_effort_max)

        all_examples.append(examples)
        all_accuracies.append(accuracies)
        all_accuracies_v.append(accuracies_v)

    plt.figure()
    #plt.title("Number of examples vs accuracy for different numbers of dimensions")
    plt.xlabel("Number of examples")
    plt.ylabel("Expected accuracy")
    plt.tight_layout()
    for i, nd in enumerate(num_dims):
        plt.plot(range(num_examples + 1), all_accuracies[i], label="{} dimensions".format(nd))
    plt.legend()


    plt.figure()
    plt.xlabel("Verbal effort")
    plt.ylabel("Expected accuracy")
    plt.tight_layout()
    #plt.title("Verbal effort vs accuracy for different numbers of dimensions")
    for i, nd in enumerate(num_dims):
        plt.plot(np.linspace(verbal_effort_min, verbal_effort_max, num_ve), all_accuracies_v[i], label="{} dimensions".format(nd))

    plt.legend()