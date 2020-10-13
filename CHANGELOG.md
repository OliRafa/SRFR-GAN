## 1.9.0 (2020-10-12)

### Fix

- **save_metrics**: correct saving to logger file

### Feat

- **visualization**: add more info about loss function on tensorboard

## 1.8.1 (2020-10-09)

### Fix

- **models**: fix sr output size

## 1.8.0 (2020-10-09)

### Feat

- **models**: change UpSampling2D to Conv2DTranspose on generator
- **hyperparameters**: change lr, momentum, beta_2 and weight_decay

### Fix

- correct train path, from training to services

## 1.7.0 (2020-10-04)

### Feat

- **hyperparameters**: change lr, momentum, beta_2 and weight_decay
- **loss_function**: change arcloss for crossentropy
- **hyperparameters**: change momentum and learning rate
- **model**: removed batch normalization from model
- **tensorboard**: add in depth loss visualization
- **hyperparameters**: change weight decay and learning rate
- **activation_function**: added MISH activation function

### Fix

- correct train path, from training to services
- bug fix, add tests, mocks, and training use case
- **losses**: fix losses to work with distributed training
- **validation**: refact lfw validation to use case architecture
- **loss_function**: correct loss behaviour when distributed for training
- **losses**: fix softmax for generator and discriminator losses
- **train**: fix train call on main.py
- **validation**: fix validation on LFW
- **validation_dataset**: Correct validation dataset input
- **optimizer**: Correct version of TensorFlow Addons on Dockerfile and requirements.txt
- **dataset**: Change dataset full to dataset 50k

### Refactor

- **models**: rollback resnet model to batch normalization implementation
- **scripts_to_tfrecords**: fix vgg script to work outside module
- **models**: add keras path to load weights instead of downloading it each time
- **tensorboard**: remove scalars for each batch
- adapt to coding style
- refactor some training code

## 1.2.0 (2020-05-30)

### Feat

- **optimizer**: Add NovoGrad optimizer
- **dataset**: Add caching to HD instead of RAM

### Refactor

- **hyperparameter_tuning**: Modify Adam hyperparameters
