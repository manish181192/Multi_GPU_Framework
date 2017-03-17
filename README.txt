# Multi_GPU_Framework
Using Multiple GPU with tensorflow

*Refer Main file for usage of framework.
*Default configurations:
  -Optimizer : AdamOptimizer( can be modified in multiGPU_Framework.py) 

###########################################
Note :
            1 .use get_cpu_variable instead of tf.Variable
            OR get_cpu_variable_shape to specify shape
            (In case initializer doesnt accept shape as argument)

            2. Add Loss to collection
            Use tf.add_to_collection('losses', loss)
  #########################################
