from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, activity_l1
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.constraints import maxnorm

from sklearn import metrics
import math
import numpy as np

def prune_noise_edges(model, prune_thresholds):
    ''' Sets all edge weights below the prune threshold to 0.
        Modifies the original model. prune_thresholds should 
        be an array with one value per hidden node. '''
    new_weights = model.get_weights()
    for weights in new_weights[0]:
        for hidden_node_idx in xrange(len(weights)):
            if abs(weights[hidden_node_idx]) <= prune_thresholds[hidden_node_idx]:
                weights[hidden_node_idx] = 0.000
              
    model.set_weights(new_weights)
    
def choose_best_inputs(model, num_desired_inputs=2, inputs_per_position=21):
    ''' For each node, sets edge weights not coming from the top 
        ranked num_desired_input positions to zero. Modifies the original
        model. '''
    new_weights = model.get_weights()
    input_edges = new_weights[0]
    
    summed_edge_weights = sum_over_positions(input_edges, inputs_per_position)
    top_ranked_dict = {}
    #for hidden_node_idx in xrange(len(input_edges[0])):
    for hidden_node_idx in xrange(len(new_weights[0][1])):
        top_ranked_dict[hidden_node_idx] = rank_inputs(summed_edge_weights, hidden_node_idx, num_desired_inputs)
        
    print len(new_weights[0][2])
    print len(top_ranked_dict.keys())
    
    for i in xrange(len(new_weights[0])):
        for j in xrange(len(new_weights[0][i])):
            if (i%20) not in top_ranked_dict[j]:
                new_weights[0][i][j] = 0.0
                
    model.set_weights(new_weights)
       
def rank_inputs(summed_weights, hidden_node_idx, num_desired_inputs):
    ''' Ranks positions in descending order based on which has the greatest 
        summed weight. Returns the top num_desired_inputs positions. '''
    pos_weights = []
    for i in xrange(len(summed_weights)):
        pos_weights.append((summed_weights[i][hidden_node_idx], i))
    
    ranked_positions = sorted(pos_weights, key=lambda x: x[0], reverse=True)
    return [x[1] for x in ranked_positions[:num_desired_inputs]]
    
    
def sum_over_positions(input_edges, inputs_per_position):
    ''' Sums edge weights for each inputs_per_position. Returns a 2D array
        of inputs. '''
    edge_weight_sums = []
    
    totaled_weights = []
    pos_count = 1
    
    for i in xrange(len(input_edges)):
        if len(totaled_weights) == 0:
            totaled_weights = input_edges[i]
        else:
            totaled_weights = [np.abs(x) + np.abs(y) for x, y in zip(input_edges[i], totaled_weights)]
        
        if i%(inputs_per_position-1) == 0:
            edge_weight_sums.append(totaled_weights)
            totaled_weights = []
     
    return edge_weight_sums
    
            