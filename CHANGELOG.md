## 3.0.0 (2020-10-28)

### Feat

- **datasets**: add casia-webface
- **hyperparameters**: add bayesian optimizer

### Refactor

- **scripts_to_tfrecords**: refac casia script to work with shardding
- **repository**: change path for class_pairs and overlapping_identities

### Fix

- **dataset**: fix remainders when batching the dataset
- **test**: fix test dataset, and accuracy function
- **visualization**: fix tensorboard not catching hyperparameter runs, accuracy not being displayed, and training being stopped

## 2.1.0 (2020-10-18)

### Feat

- **hyperparameters**: lower learning rate

## 2.0.0 (2020-10-17)

### Fix

- **validation**: fix validation not compatible with model's outputs
- **testing**: fix testing loop
- **testing**: fix model test during training

### Feat

- **hyperparameters**: lower beta_1 and face_recognition_weight

## 1.10.0 (2020-10-15)

### Feat

- **hyperparameters**: change super_resolution_weight, perceptual_weight, generator_weight, l1_weight
- **train**: add cosine decay as a learning rate scheduler
- **visualization**: add more info about loss function on tensorboard
- **models**: change UpSampling2D to Conv2DTranspose on generator

### Perf

- **validation**: moved cache from disk to ram and add prefetching
- **data**: add prefetch and change cache buffer from hdd to memory
- **data**: add prefetch and change cache buffer from hdd to memory

### Fix

- fix summary writer path
- **save_metrics**: correct saving to logger file
- **models**: fix sr output size

## 1.7.0 (2020-10-04)

### Feat

- **hyperparameters**: change lr, momentum, beta_2 and weight_decay

### Fix

- correct train path, from training to services

### Refactor

- **models**: rollback resnet model to batch normalization implementation

## 2.2.2 (2020-10-23)

### Fix

- **dataset**: fix remainders when batching the dataset

## 2.2.1 (2020-10-22)

### Fix

- **test**: fix test dataset, and accuracy function
- **visualization**: fix tensorboard not catching hyperparameter runs, accuracy not being displayed, and training being stopped

## 2.2.0 (2020-10-21)

### Feat

- **hyperparameters**: add bayesian optimizer

## 1.12.0 (2020-10-18)

### Feat

- **hyperparameters**: lower learning rate

### Fix

- **validation**: fix validation not compatible with model's outputs

## 1.11.0 (2020-10-17)

### Feat

- **hyperparameters**: lower beta_1 and face_recognition_weight
- **hyperparameters**: change super_resolution_weight, perceptual_weight, generator_weight, l1_weight
- **train**: add cosine decay as a learning rate scheduler

### Fix

- **testing**: fix model test during training
- fix summary writer path

### Perf

- **validation**: moved cache from disk to ram and add prefetching

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
