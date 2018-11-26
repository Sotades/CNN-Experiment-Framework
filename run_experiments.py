from experimental_run import experimental_run
from build_model128x128 import build_model_128x128
from build_model64x64 import build_model_64x64

path_to_128 = 'D:/Data/Dataset2_23-11_128'
path_to_64 = 'D:/Data/Dataset2_23-11_64'

epochs = 30
lamda = 0.00005

# Run #1:
# image_size: (64 x 64)
# learning rate: 0.00005
# minibatch size: 25
# optimizer: RMSProp
model = experimental_run(experiment_name='Run #1',
                         epochs=epochs,
                         image_size=64,
                         lamda=lamda,
                         minibatch_size=25,
                         learning_rate=0.00005,
                         optimizer='RMSprop',
                         path_to_64=path_to_64)

# Run #2:
# image_size: (64 x 64)
# learning rate: 0.00005
# minibatch size: 250
# optimizer: Adam
model = experimental_run(experiment_name='Run #2',
                         epochs=epochs,
                         image_size=64,
                         lamda=lamda,
                         minibatch_size=250,
                         learning_rate=0.00005,
                         optimizer='Adam',
                         path_to_64=path_to_64)

# Run #3:
# image_size: (64 x 64)
# learning rate: 0.001
# minibatch size: 25
# optimizer: Adam
model = experimental_run(experiment_name='Run #3',
                         epochs=epochs,
                         image_size=64,
                         lamda=lamda,
                         minibatch_size=25,
                         learning_rate=0.001,
                         optimizer='Adam',
                         path_to_64=path_to_64)

# Run #4:
# image_size: (64 x 64)
# learning rate: 0.001
# minibatch size: 250
# optimizer: RMSprop
model = experimental_run(experiment_name='Run #4',
                         epochs=epochs,
                         image_size=64,
                         lamda=lamda,
                         minibatch_size=250,
                         learning_rate=0.001,
                         optimizer='RMSprop',
                         path_to_64=path_to_64)


# Run #5:
# image_size: (128 x 128)
# learning rate: 0.00005
# minibatch size: 25
# optimizer: Adam
model = experimental_run(experiment_name='Run #5',
                         epochs=epochs,
                         image_size=128,
                         lamda=lamda,
                         minibatch_size=25,
                         learning_rate=0.00005,
                         optimizer='Adam',
                         path_to_128=path_to_128)

# Run #6:
# image_size: (128 x 128)
# learning rate: 0.00005
# minibatch size: 250
# optimizer: RMSprop
model = experimental_run(experiment_name='Run #6',
                         epochs=epochs,
                         image_size=128,
                         lamda=lamda,
                         minibatch_size=250,
                         learning_rate=0.00005,
                         optimizer='RMSprop',
                         path_to_128=path_to_128)

# Run #7:
# image_size: (128 x 128)
# learning rate: 0.001
# minibatch size: 25
# optimizer: RMSprop
model = experimental_run(experiment_name='Run #7',
                         epochs=epochs,
                         image_size=128,
                         lamda=lamda,
                         minibatch_size=25,
                         learning_rate=0.001,
                         optimizer='RMSprop',
                         path_to_128=path_to_128)

# Run #8:
# image_size: (128 x 128)
# learning rate: 0.001
# minibatch size: 250
# optimizer: Adam
model = experimental_run(experiment_name='Run #8',
                         epochs=epochs,
                         image_size=128,
                         lamda=lamda,
                         minibatch_size=250,
                         learning_rate=0.001,
                         optimizer='Adam',
                         path_to_128=path_to_128)






