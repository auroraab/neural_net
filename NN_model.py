from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, activity_l1
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.constraints import maxnorm

from sklearn import metrics

def build_model(train_features,
             neurons = 100, 
             layers = 1, 
             verbose = 0,
             lmbda = 1e-5, 
             learn_rate = 0.01,
             b_maxnorm = 0,
             batch_norm = 0, 
             decay = 0, 
             momentum = 0.5, 
             dropout_P=0.5,
             activation = "relu"):
    #Start from empty sequential model where we can stack layers
    model = Sequential()
    #Add one+ fully-connected layers
    for l in range(layers):
        if l ==0:
            model.add(Dense(output_dim=neurons, input_dim=train_features.shape[1],W_regularizer=l1(lmbda),b_constraint=maxnorm(m=b_maxnorm)))
        else:
            model.add(Dense(b_constraint=maxnorm(m=b_norm),output_dim=neurons, input_dim=neurons,W_regularizer=l1(lmbda)))
            
        # Add activation function to each neuron
        model.add(Activation(activation))             #rectified linear activation

        #batch normalization: maintain mean activation ~ 0 and activation standard deviation ~ 1.
        if batch_norm:
            model.add(BatchNormalization())

        #dropout
        model.add(Dropout(dropout_P))

    #Add fully-connected layer with 2 neurons, one for each class of labels
    model.add(Dense(output_dim=2))

    #Add softmax layer to force the 2 outputs to sum up to one so that we have a probability representation over the labels.
    model.add(Activation("softmax"))

    #Compile model with categorical_crossentrotry as loss, and stochastic gradient descent(learning rate=0.001, momentum=0.5 as optimizer)
    model.compile(loss='categorical_crossentropy',  optimizer=SGD(lr= learn_rate, momentum=momentum,decay = decay, nesterov=True), metrics=['accuracy'])
    return model


def train_model(model, train_features, train_labels, class_weights, num_epochs=10, verbose = 1):
    model.fit(train_features, train_labels, verbose= verbose, nb_epoch=num_epochs, batch_size=32, validation_split=0.1, shuffle = True, class_weight=class_weights)
      
def save_model(model, model_number):
    model_name = "model_" + str(model_number) + ".hdf"
    model.save_weights(model_name)


