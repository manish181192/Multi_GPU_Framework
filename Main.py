import tensorflow as tf
from multiGPU_Framework import get_cpu_variable, get_cpu_variable_shape, multiGPU_Framework


placeholder_list = []
def loss_prediction_function():
    '''
    :param : None
    :return: loss, prediction
    
    Note :
            1 .use get_cpu_variable instead of tf.Variable
            OR get_cpu_variable_shape to specify shape(In case 
            initializer doesnt accept shape as argument)

            2. Add Loss to collection
            Use tf.add_to_collection('losses', loss)

            mnist multi-gpu
    '''
    #Placeholder
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    placeholder_list.append(x)
    placeholder_list.append(y_)
    # Create the model
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    '''
      Calculate Prediction and Loss
    '''
    # Define loss and optimizer

    predictions = tf.nn.softmax(y)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    tf.add_to_collection('losses', loss)
    return loss, predictions

from tensorflow.examples.tutorials.mnist import input_data
data_dir = '/home/mvidyasa/Downloads/Multi_GPU_Framework-master/mnist'

mnist = input_data.read_data_sets(data_dir, one_hot=True)
batch_generator =  mnist.train.next_batch
mgf = multiGPU_Framework(loss_prediction_function= loss_prediction_function)
mgf.train_model(num_epochs=1000,
                placeholders= placeholder_list,
                learning_rate=0.001,
                batch_generator=batch_generator,
                batch_size = 100)
