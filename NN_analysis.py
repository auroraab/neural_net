import numpy as np
import NN_model
import NN_preprocessing

aas = list('ARNDCQEGHILKMFPSTWYV-')

def evaluate_model_goodness(model, val_features, val_labels):
    samp_weight = np.array(len(val_labels)*[0.06921372])
    samp_weight[val_labels[:,0] == [0]] = 0.93

    loss, acc = model.evaluate(val_features, val_labels, batch_size=32, verbose =0, sample_weight=samp_weight)
    return (loss, acc)

def predict_using_model(model, train_features):
    return model.predict(train_features, verbose = 1)

def aa_significance(model, batch_norm = 0):
    #layers
    ls = model.get_weights()
    l1_mat = np.array(ls[0])
    n_inputs, n_neur = l1_mat.shape
    n_pos = n_inputs / 21
    tot_w1_in = np.abs(l1_mat).sum(axis = 0)
    input_means = np.abs(l1_mat).sum(axis = 1)
    input_std = np.std(l1_mat, axis=1)
    aa_matrices = {}
    pos_sig = np.zeros((n_pos))

    for p1 in range(n_pos):
        sig = np.zeros((21,1))
        for aa1 in range(21):
            i1= p1*21+aa1
            x = l1_mat[i1]        #vector of outgoing weights from input i
            x = np.divide(x, tot_w1_in)    #normalizing by total weights going into each hidden node
            x = np.multiply(x,(ls[2 +4*batch_norm]*np.array([-1,1])).sum(axis = 1))    #multiply by values of each hidden node (value = abs(out1-out0))
            sig [aa1] = x.sum()
#         pos_sig[p1] = np.abs(sig).sum()/21
        pos_sig[p1] = (sig**2).sum()**0.5/21
        aa_matrices[p1] = np.array(sig)
    return aa_matrices, pos_sig

def aa_correlation(model, batch_norm = 0):
    #layers
    ls = model.get_weights()
    l1_mat = np.array(ls[0])
    n_inputs, n_neur = l1_mat.shape
    n_pos = n_inputs / 21
    tot_w1_in = (np.abs(l1_mat).sum(axis = 0))
    input_std = np.std(l1_mat, axis = 1)
    input_std += np.percentile(input_std,5)
    aa_matrices = {}
    pos_corr = np.zeros((n_pos, n_pos))
    for p1 in range(n_pos):
        for p2 in range(n_pos):
            #generate matrix of all aa x aa compatibilities between positions p1/p2
            corr = np.zeros((21,21))
            for aa1 in range(21):
                for aa2 in range(21):
                    i1, i2 = p1*21+aa1, p2*21+aa2
                    x = np.multiply(l1_mat[i1],l1_mat[i2])/(input_std[i1]*input_std[i2])      #calculate raw covariance
                    x = np.divide(x, tot_w1_in**2)      #divide by total input to each node (squared)
                    x = np.multiply(x,(ls[2+4*batch_norm]*np.array([-1,1])).sum(axis = 1))     #multiply by values of each hidden node (value = abs(out1-out0))
                    corr [aa1, aa2] = (x).sum()
            pos_corr[p1,p2] = (corr**2).sum()**0.5/21**2
            try:
                aa_matrices[p1][p2] = np.array(corr)
            except KeyError:
                aa_matrices[p1] = {}
                aa_matrices[p1][p2] = np.array(corr)
    return aa_matrices, pos_corr

def aa_correlation3(model):    
    ls = model.get_weights()
    l1_mat = np.array(ls[0])
    n_inputs, n_neur = l1_mat.shape
    n_pos = n_inputs / 21     
    tot_w1_in = np.abs(l1_mat).sum(axis = 0)**3
    input_means = np.abs(l1_mat).sum(axis = 1)
    input_std = np.std(l1_mat, axis=1)
    input_std += np.percentile(input_std,5)
    
    aa_matrices = {}
    pos_corr = np.zeros((n_pos, n_pos, n_pos))

    for p1 in range(n_pos):
        print p1
        for p2 in range(n_pos):
            for p3 in range(n_pos):
                corr = np.zeros((21,21,21))
                for aa1 in range(21):
                    for aa2 in range(21):
                        for aa3 in range(21):
                            i1, i2, i3 = p1*21+aa1, p2*21+aa2, p3*21+aa3
                            x = np.multiply(np.multiply(l1_mat[i1],l1_mat[i2]),l1_mat[i3])/(input_std[i1]*input_std[i2]*input_std[i3])
                            x = np.divide(x, tot_w1_in)
                            x = np.multiply(x,(ls[2]*np.array([-1,1])).sum(axis = 1))
                            corr [aa1,aa2,aa3] = x.sum()
                pos_corr[p1,p2,p3] = (corr**2).sum()**0.5/21**3

                try:
                    aa_matrices[p1][p2][p3] = np.array(corr)
                except KeyError:
                    try:
                        aa_matrices[p1][p2] = {}
                        aa_matrices[p1][p2][p3] = np.array(corr)
                    except KeyError:
                        aa_matrices[p1] = {}
                        aa_matrices[p1][p2] = {}
                        aa_matrices[p1][p2][p3] = np.array(corr)
                        

    return  pos_corr,aa_matrices


def ZNmat(mat):
    ZNmat = np.zeros(np.shape(mat))

    for i in range(np.shape(mat)[0]):
        for j in range(np.shape(mat)[1]):
            col = mat[:,j]
            col = np.concatenate((col[0:i],col[i+1:]))
            row = mat[i,:]
            row = np.concatenate((row[0:j],row[j+1:]))
            X_var = np.std(row)**2
            X_mean = np.mean(row)
            Y_var = np.std(col)**2
            Y_mean = np.mean(col)

            XY_mean = (X_mean*Y_var + Y_mean*X_var)/(X_var + Y_var)
            XY_var = (X_var*Y_var)/(X_var + Y_var)
            ZNmat[i,j] = (mat[i,j]-XY_mean)/(np.sqrt(XY_var))
    return ZNmat

def calc_overlap (hist1,hist2):
    y1,x1 = hist1[0],hist1[1]
    y2,x2 = hist2[0],hist2[1]
    rangee = [min(min(x1), min(x2))]
    step = max([x1[1]-x1[0],x2[1]-x2[0]])
    area1, area2, overlap = 0,0,0
    for i in np.arange(min([x1.min(),x2.min()]),max([x1.max(),x2.max()]), step):
        rect1, rect2 = 0,0
        try:
            rect1 += y1[np.where(np.abs(x1-i)<step/2)].sum()
        except IndexError:
            pass
        try:
            rect2 += y2[np.where(np.abs(x2-i)<step/2)].sum()
        except IndexError:
            pass
        area1 += rect1
        area2 += rect2
        if rect1>rect2:
            overlap += rect2
        else:
            overlap += rect1
    if np.abs(1-area1/area2)>0.01:
        print "HISTOGRAM AREA RATIO != 1 (area =", area1/area2, ")"
    return 2*overlap/(area1+area2)



