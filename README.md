# Multi-Layer-Perceptron-Learning-in-Tensorflow
We will understand the concept of a multi-layer perceptron and its implementation in Python using the TensorFlow library.

Multi-layer perception is also known as MLP. It is fully connected dense layers, which transform any input dimension to the desired dimension. A multi-layer perception is a neural network that has multiple layers. To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other neurons.
A multi-layer perceptron has one input layer and for each input, there is one neuron(or node), it has one output layer with a single node for each output and it can have any number of hidden layers and each hidden layer can have any number of nodes. A schematic diagram of a Multi-Layer Perceptron (MLP) is depicted below.

![nodeNeural](https://github.com/user-attachments/assets/e31f7821-1958-4043-b61a-ea5f381f08a4)

In the multi-layer perceptron diagram above, we can see that there are three inputs and thus three input nodes and the hidden layer has three nodes. The output layer gives two outputs, therefore there are two output nodes. The nodes in the input layer take input and forward it for further process, in the diagram above the nodes in the input layer forwards their output to each of the three nodes in the hidden layer, and in the same way, the hidden layer processes the information and passes it to the output layer. 

Every node in the multi-layer perception uses a sigmoid activation function. The sigmoid activation function takes real values as input and converts them to numbers between 0 and 1 using the sigmoid formula.

α(x) = 1/( 1 + exp(-x))

Now that we are done with the theory part of multi-layer perception, let’s go ahead and implement some code in python using the TensorFlow library.

Stepwise Implementation 

Step 1: Import the necessary libraries.

Step 2: Download the dataset.
        TensorFlow allows us to read the MNIST dataset and we can load it directly in the program as a train and test dataset.
        
Step 3: Now we will convert the pixels into floating-point values.
        We are converting the pixel values into floating-point values to make the predictions. Changing the numbers into grayscale values will be beneficial as the values 
        become small and the computation becomes easier and faster. As the pixel values range from 0 to 256, apart from 0 the range is 255. So dividing all the values by 255 
        will convert it to range from 0 to 1
        
Step 4: Understand the structure of the dataset
        Thus we get that we have 60,000 records in the training dataset and 10,000 records in the test dataset and Every image in the dataset is of the size 28×28.

Step 5: Visualize the data.

Step 6: Form the Input, hidden, and output layers.

Some important points to note:

->The Sequential model allows us to create models layer-by-layer as we need in a multi-layer perceptron and is limited to single-input, single-output stacks of layers.

->Flatten flattens the input provided without affecting the batch size. For example, If inputs are shaped (batch_size,) without a feature axis, then flattening adds an extra 
  channel dimension and output shape is (batch_size, 1).

->Activation is for using the sigmoid activation function.

->The first two Dense layers are used to make a fully connected model and are the hidden layers.

->The last Dense layer is the output layer which contains 10 neurons that decide which category the image belongs to.

Step 7: Compile the model.
        Compile function is used here that involves the use of loss, optimizers, and metrics. Here loss function used is sparse_categorical_crossentropy, optimizer used is 
        adam.

Step 8: Fit the model.

Some important points to note:

->Epochs tell us the number of times the model will be trained in forwarding and backward passes.

->Batch Size represents the number of samples, If it’s unspecified, batch_size will default to 32.

->Validation Split is a float value between 0 and 1. The model will set apart this fraction of the training data to evaluate the loss and any model metrics at the end of 
  each epoch. (The model will not be trained on this data)

Step 9: Find Accuracy of the model.
        We got the accuracy of our model 92% by using model.evaluate() on the test samples.
        
