class Layer():
    def __init__(self, perceptron_number, inputs_per_neuron):
        '''Creates layer with weights initialised using Xavier initialisation'''
        mean, sd = 0, 0.5
        n = np.sqrt(perceptron_number*inputs_per_neuron)
        self.weights = (np.random.normal(mean, sd, (inputs_per_neuron, perceptron_number))) / n
        
class Network():
    def __init__(self, input_layer, hidden_layer, learning_rate = 1e-4):
        '''Adds layers to network and initialises learning rate'''
        self.layer1 = input_layer
        self.layer2 = hidden_layer
        self.learning_rate = learning_rate
        
    def normalise(self, val):
        '''Normalises input to range (0 to 1) using sigmoid function'''
        normalised = (1 / (1 + np.exp(-val)))
        return normalised
    
    def weight_derivative(self, output):
        '''Calculates error weight derivative for weight adjustment'''
        derivative = self.normalise(output)
        derivative = derivative * (1 - derivative)
        return derivative
    
    def learn(self, training_inputs, training_values, iterations):
        '''Train network using given data'''
        for _ in range(iterations):
            # Pass training set into network and collect outputs
            layer_1_output, layer_2_output = self.think(training_inputs)
            
            # Calculate error for layer 2 from true values
            layer2_error = training_values - layer_2_output
            layer2_delta = layer2_error * self.weight_derivative(layer_2_output)

            # Back propagate the error into the weightings for layer 1
            layer1_error = layer2_delta.dot(self.layer2.weights.T)
            layer1_delta = layer1_error * self.weight_derivative(layer_1_output)

            # Calculate weight adjustment for each layer
            layer1_weight_adjustment = training_inputs.T.dot(layer1_delta)
            layer2_weight_adjustment = layer_1_output.T.dot(layer2_delta)

            # Adjust weights by measure of learning rate
            self.layer1.weights += (layer1_weight_adjustment*self.learning_rate)
            self.layer2.weights += (layer2_weight_adjustment*self.learning_rate)
    
    def think(self, inputs):
        '''Generates outputs from input using weights and sigmoid function'''
        layer1_output = self.normalise(np.dot(inputs, self.layer1.weights))
        layer2_output = self.normalise(np.dot(layer1_output, self.layer2.weights))
        return layer1_output, layer2_output
		
def train(training_data, iterations=4000):
    """
    Train a model on the training_data

    :param training_data: a two-dimensional numpy-array with shape = [5000, 39] 
    
    :return fitted_model: any data structure that captures your model
    """
    # Initialise values for network creation
    feature_number = len(training_data[:, 1:][0])
    neuron_number = int(feature_number / 3) # somewhat arbitrary
    
    # Create 2 layer network with these values
    input_layer = Layer(neuron_number, feature_number)
    hidden_layer_1 = Layer(1, neuron_number)
    neural_network = Network(input_layer, hidden_layer_1)

    # Train on given examples with given iterations.
    training_inputs = training_data[:, 1:]
    training_values = np.array([training_data[:, 0]]).T
    neural_network.learn(training_inputs, training_values, iterations)
    
    # Return trained network
    fitted_model = neural_network
    return fitted_model
	
def test(testing_data, fitted_model):
    """
    Classify the rows of testing_data using a fitted_model. 

    :param testing_data: a two-dimensional numpy-array with shape = [n_test_samples, 38]
    :param fitted_model: the output of your train function.

    :return class_predictions: a numpy array containing the class predictions for each row
        of testing_data.
    """
    predictions = []
    for data in testing_data:
        hidden_layer, output = fitted_model.think(data)
        output = (np.around(output)).item()
        predictions.append(output)
    
    # Return numpy array
    class_predictions = np.array(predictions)
    return class_predictions
	
def cross_validate(training_data, k=5):
    '''Performs k-fold cross validation and returns mean accuracy'''
    # Initialize variables
    accuracies = []
    tests = np.array_split(training_data,k)
    
    # Split data and perform validations
    for i in range(k):
        test_data = tests[i]
        remaining_data = np.delete(tests, i, 0)
        train_data = remaining_data[0]
        for j in range(1,len(remaining_data)):
            train_data = np.append(train_data, remaining_data[j], axis = 0)
        
        # Train and test
        fitted_model = train(train_data)
        predictions = test(test_data[:, 1:], fitted_model)
        
        # Check solutions
        solutions = test_data[:, 0]
        acc = 0
        for x in range(len(predictions)):
            if solutions[x] == predictions[x]:
                acc += 1

        acc /= len(predictions)
        accuracies.append(acc)

    accuracy = np.mean(accuracies)
    return accuracy

fitted_model = train(training_data)
