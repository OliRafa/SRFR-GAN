def test__step_function(instantiate_training, train_01):
    train = instantiate_training
    output = train._step_function(*train_01)
    assert output is True
