import numpy as np

def estimate_log_class_priors(data):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column, calculate the logarithm of the empirical class priors,
    that is, the logarithm of the proportions of 0s and 1s:
    log(P(C=0)) and log(P(C=1))

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                 the first column contains the binary response (coded as 0s and 1s).

    :return log_class_priors: a numpy array of length two
    """
    # Initialize variables
    data_size = len(data)
    ham = 0 # Binary response 0
    spam = 0 # Binary response 1
    
    # Classify each data point
    for test in data:
        if test[0] == 0:
            ham += 1
        else:
            spam += 1
    
    # Calculate proportions and empirical class priors
    ham_ecp = np.log(ham / data_size)
    spam_ecp = np.log(spam / data_size)
    log_class_priors = np.array([ham_ecp, spam_ecp])
    
    return log_class_priors
	
def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column and binary features, calculate the empirical
    class-conditional likelihoods, that is,
    log(P(w_i | c_j)) for all features i and both classes (j in {0, 1}).

    Assume a multinomial feature distribution and use Laplace smoothing
    if alpha > 0.

    :param data: a two-dimensional numpy-array with shape = [n_samples, n_features]

    :return theta:
        a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
        logarithm of the probability of feature i appearing in a sample belonging 
        to class j.
    """
    # Ham data
    ham_indexes = (data[:,0] == 0)
    ham_data = data[ham_indexes]
    ham_sums = np.delete((ham_data.sum(0)),0)
    ham_total_words = np.sum(ham_sums)
    ham_features = len(ham_sums)
    
    for i in range(ham_features):
        ham_sums[i] = (ham_sums[i]+alpha) / (ham_total_words + (ham_features * alpha))
    
    ham_sums = np.log(ham_sums)
    
    # Spam data
    spam_indexes = np.nonzero(data[:,0])
    spam_data = data[spam_indexes]
    spam_sums = np.delete((spam_data.sum(0)),0)
    spam_total_words = np.sum(spam_sums)
    spam_features = len(spam_sums)
    
    for i in range(spam_features):
        spam_sums[i] = (spam_sums[i]+alpha) / (spam_total_words + (spam_features * alpha))
    
    spam_sums = np.log(spam_sums)
    
    theta = np.array([ham_sums, spam_sums])
    return theta
	
def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
    """
    Given a new data set with binary features, predict the corresponding
    response for each instance (row) of the new_data set.

    :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
    :param log_class_priors: a numpy array of length 2.
    :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
        theta[j, i] corresponds to the logarithm of the probability of feature i appearing
        in a sample belonging to class j.
    :return class_predictions: a numpy array containing the class predictions for each row
        of new_data.
    """
    class_predictions = []
    
    # Loop through message and predict
    for message in new_data:
        ham_probability = log_class_priors[0]+(np.sum(log_class_conditional_likelihoods[0]*message))
        spam_probability = log_class_priors[1]+(np.sum(log_class_conditional_likelihoods[1]*message))
        class_probability = np.array([ham_probability, spam_probability])
        predicted_class = np.argmax(class_probability)
        class_predictions.append(predicted_class)
        
    # Return numpy array
    class_predictions = np.array(class_predictions)
    return class_predictions

def accuracy(y_predictions, y_true):
    """
    Calculate the accuracy.
    
    :param y_predictions: a one-dimensional numpy array of predicted classes (0s and 1s).
    :param y_true: a one-dimensional numpy array of true classes (0s and 1s).
    
    :return acc: a float between 0 and 1 
    """
    total_messages = len(y_predictions)
    accuracy = 0
    for i in range(total_messages):
        if y_predictions[i] == y_true[i]:
            accuracy += 1
            
    acc = accuracy / total_messages
    print(acc)
    return acc