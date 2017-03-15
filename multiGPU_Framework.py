import tensorflow as tf
class multiGPU_Framework(object):

    loss_prediction_function = None
    batches = []
    count_batch =0

    def __init__(self, loss_prediction_function,num_gpus=1,TOWER_NAME = "MODEL_TOWER"):

        self.loss_prediction_function = loss_prediction_function
        self.num_gpus = num_gpus
        self.TOWER_NAME = TOWER_NAME

    def calculate_loss(self, scope):
        '''

        :param scope: Current Scope
        :return: Total loss
        '''

        _, prediction = self.loss_prediction_function(self.batches[self.count_batch])
        self.count_batch +=1
        losses = tf.get_collection('losses', scope)

        total_loss = tf.add_n(losses, name= 'Total_loss')
        return total_loss


    def train_model(self, batches, num_epochs, learning_rate = 0.0001):
        '''

        :param batches: List of all batches
        :param num_epochs: No of epochs
        :param learning_rate: Learning rate
        :return: _
        '''

        self.convert_to_tensor(batches)
        with tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
            # Calculate the gradients for each model tower.
            tower_grads = []
            for i in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (self.TOWER_NAME, i)) as scope:
                        loss = self.calculate_loss(scope)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = optimizer.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
            init = tf.global_variables_initializer()
            with tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False)) as session:
                session.run(init)
                for epoch in range(num_epochs):
                    print "epoch : " + str(epoch)
                    _, loss_value = session.run([train_op, loss])
                    print "LOSS: " + str(loss_value)

    def convert_to_tensor(self, batches):
        for batch in batches:
            batch_components_tensors = []
            for batch_component in batch:
                batch_components_tensors.append(tf.pack(batch_component))
            self.batches.append(batch_components_tensors)

def get_cpu_variable(name, initializer, dtype = None):
    if dtype is None:
        dtype = tf.float32
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, initializer=initializer, dtype=dtype)
    return var
def get_cpu_variable_shape(name, shape, initializer, dtype = None):
    if dtype is None:
        dtype = tf.float32
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype)
    return var
def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      print "G-VALUE: " + str(g)
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


