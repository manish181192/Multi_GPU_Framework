from multiGPU_Figer import multiGPU_Figer
from DataExtractor import DataExtractor
from parameters import FLAGS
data_reader = DataExtractor(embedding_size= FLAGS.embedding_size,
                            lstm_hidden_size=FLAGS.lstm_hidden_size,
                            Da= FLAGS.Da,
                            l2_reg_lambda= FLAGS.l2_reg_lambda,
                            batch_size= FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)
mgpu_lstm = multiGPU_Figer(num_gpus=3, TOWER_NAME='FIGER', data_reader = data_reader)
mgpu_lstm.train_figer()