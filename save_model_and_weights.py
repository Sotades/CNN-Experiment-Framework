
def save_model_and_weights(experiment_name, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(experiment_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(experiment_name + '.h5')

    return
