import pickle
import torch


def save_trained_model(model, model_name, errors_train, errors_test, n_layer, h_dim, learning_rate, nb_epochs):
    
    torch.save(model, f"../COBAR_Models/{model_name}_hidden_{n_layer}x{h_dim}_lr_{learning_rate}_epochs_{nb_epochs}.pt")
    
    errors = open(f"../COBAR_Models/errors_{model_name}_hidden_{n_layer}x{h_dim}_lr_{learning_rate}_epochs_{nb_epochs}.pkl", "wb")
    pickle.dump(errors_train, errors)
    pickle.dump(errors_test, errors)
    errors.close()