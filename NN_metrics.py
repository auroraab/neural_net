import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.stats as st
from sklearn import metrics
from NN_preprocessing import extract_sequences, aa2bin, trunc_seq
from NN_analysis import evaluate_model_goodness, aa_significance,aa_correlation
from NN_visualization import heatmap , bar_graph

def load_gremlin_weights(rr_ind, hk_ind):
    grem_model = pickle.load(open("TCS_grem_1379.pck", "r"))
    grem_e = grem_model['w'].keys()
    grem_e = [(i[0]-1, i[1]) for i in grem_e]
    
    #all edges in our model that GREMLIN scored
    edges = []
    for e in grem_e:
        if e[0] in rr_ind+hk_ind and e[1] in rr_ind+hk_ind:
            edges += [e]
    #inter-protein edges in our model that GREMLIN scored
    inter_edges = []
    for e in edges: 
        if e[0] in rr_ind and e[1] in hk_ind: 
            inter_edges += [e]
    return edges, inter_edges,  grem_model['w']

def gremlin_correlation(rr_ind, hk_ind, gremlin_edges, grem_weights, aa_matrices, plot = False):
    """ calculate correlation between AA-AA compatibility in NN model vs Gremlin model 
        indices of residues in model (rr_ind , hk_ind)
        gremlin edges = tuple of tuples (all edges in both models, interprotein edges in both models)
        grem_weights = dict of 21x21 aa compatibility matrices from gremlin
        aa_matrices =  dict of 21x21 aa compatibility matrices from NN covariation """
    edges, inter_edges = gremlin_edges
    rrhk = rr_ind + hk_ind
    corrs = np.zeros((len(rrhk),len(rrhk)))
    pvals = np.zeros((len(rrhk),len(rrhk)))
    inter_edge_corrs = []
    if plot:
        plt.figure()
    for e in edges:
        r,h = e
        grem = -np.array(grem_weights[(r+1,e[1])]).reshape((-1))
        nn = np.array(aa_matrices[rrhk.index(r)][rrhk.index(h)]).reshape((-1))
        corr, pval =  st.pearsonr(grem, nn)
        corrs[rrhk.index(r),rrhk.index(h)] = corr
        pvals[rrhk.index(r),rrhk.index(h)] = np.log10(pval)
        if e in inter_edges:
            inter_edge_corrs += [corr]
            if plot:
                plt.plot(grem, nn, ".", alpha = 0.3)
    if plot:
        m_corrs, m_pvals = (corrs.max()-corrs.min())/2,(pvals.min()-pvals.max())/2
        for i in range(len(rrhk)):
            for j in range(len(rrhk)):
                if j <= i:
                    corrs[i,j] = m_corrs
                    pvals[i,j] = m_pvals
        plt.xlabel("- Gremlin score")
        plt.ylabel("NN correlation score")
        plt.title("Weight Values")
        # heatmap(corrs, labels = [rrhk,rrhk], title = "Pearson correlation: GREMLIN vs NN corr score")
        # heatmap(pvals, labels = [rrhk,rrhk], title = "Pearson log10(P-value): GREMLIN vs NN corr score")
        heatmap(corrs[:len(rr_ind),len(rr_ind):],labels = [hk_ind,rr_ind], title = "Pearson correlation: GREMLIN vs NN corr score")
        # heatmap(pvals[:len(rr_ind),len(rr_ind):],labels = [hk_ind,rr_ind], title = "Pearson log10(P-value): GREMLIN vs NN corr score")
    return inter_edge_corrs

def rank_pairs(corr_mat, rrhk, perc_cutoff):
    """returns list of position-pairs with in the top [perc_cutoff] percentile"""
    nn_corr = {}
    for pair in np.array(np.where(np.triu(corr_mat) >  np.percentile(corr_mat,90))).T:
        if pair[1] != pair[0]:
            nn_corr [(rrhk[pair[0]],rrhk[pair[1]])] =  corr_mat[pair[0], pair[1]]
    return nn_corr

def rank_positions(nn_corr, print_this = 1):
    """collect, rank and print all positions in top pairs"""
    res_totals = {}
    for pair in sorted(nn_corr, key= nn_corr.__getitem__, reverse = True):
        if pair[0]<130 and pair[1]>120:
            for res in pair:
                try:
                    res_totals[res] += nn_corr[pair]**1
                except KeyError:
                    res_totals[res] = nn_corr[pair]**1

    top_rr, top_hk = [], []
    print "top HK spec residues"
    for res in sorted(res_totals, key= res_totals.__getitem__, reverse = True):
        if res > 110:
            print res, res_totals[res]  
            top_hk += [res]
    print
    print "top RR spec residues"
    for res in sorted(res_totals, key= res_totals.__getitem__, reverse = True):
        if res <=110:
            print res, res_totals[res]    
            top_rr += [res]

    print "rr_ind =",top_rr
    print "hk_ind =",top_hk

def make_all_features_pairwise(seqs,rr_ind, hk_ind, edges):
    l = len(makeFeature_pairwise(seqs[0],rr_ind, hk_ind, edges = edges))
    n = len(seqs)
    features = np.zeros((n,l))
    for i in range(n):
        features[i] = makeFeature_pairwise(seqs[i],rr_ind, hk_ind, edges)
    return features

def makeFeature_pairwise(s,rr_ind, hk_ind, edges):
    feat = []
    rr, hk = trunc_seq(s, rr_ind, hk_ind)
    for edge in edges:
        aa1 = s[edge[0]]
        aa2 = s[edge[1]]
        edge_feat = np.outer(aa2bin(aa1),aa2bin(aa2)).reshape(-1)
        feat = np.concatenate((feat, edge_feat))
    return feat
    
def make_pairwise_features(nn_corr, edges, rr_ind, hk_ind):
    """ generate features that describe with AA-AA pairs found at top position-pairs
            nn_corr: dict of top position pairs and their correlation scores
            edges: position pairs incorporated into features
            rrhk: rr_ind + hk_ind"""
    rrhk = rr_ind + hk_ind
    val_y, val_seqs= extract_sequences("validation_data.txt")
    valX = make_all_features_pairwise(val_seqs,rr_ind, hk_ind, edges = edges)
    return valX, val_y

def pairwise_score_matrix(edges, aa_matrices, rrhk):
    """ generate score matrix for specified position-position edges """
    n_e = len(edges)
    W_nn = np.zeros((n_e,21**2))
    for i in range(n_e):
        e = edges[i]
        r,h = e
        W_nn[i] = np.array(aa_matrices[rrhk.index(r)][rrhk.index(h)]).reshape((-1))
    W_nn = W_nn.reshape((-1))
    return W_nn

def pairwise_auc(valX, val_y, W_nn, plot = 0):
    """ calculate ROC AUC on validation set using only position pair weights """
    scores = np.sum((valX*W_nn), axis=1)
    scores = scores-scores.min()
    scores = scores/scores.max()
    if not plot:
        return metrics.roc_auc_score(val_y,scores)
    else:
        fpr, tpr, thresholds = metrics.roc_curve(val_y,scores )
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        auc = metrics.roc_auc_score(val_y,scores)
        plt.title("AUC = "+str(round(auc,4)))
        return auc

#combine all 
def analytics (model, valX, val_y, rr_ind, hk_ind, verbose = 0, batch_norm = 0):
    #accuracy, loss and AUC
    rrhk = rr_ind + hk_ind
    y_hat = model.predict(valX, verbose = 0)
    loss ,acc = evaluate_model_goodness(model, valX, val_y)
    auc = metrics.roc_auc_score(val_y, y_hat)

    if verbose:
        print "Loss: " + str(loss)
        print "Accuracy: " + str(acc)
        print "AUC score: " + str(auc)
        
        ls = model.get_weights()
        layer = np.array(ls[-2])
        plt.figure()
        plt.hist(layer[:,0], bins = 100, alpha = 0.5)
        plt.hist(layer[:,1], bins = 100, alpha = 0.5)
        plt.figure()
        plt.plot(layer[:,0], layer[:,1],".", alpha = 0.5)
        plt.xlabel("HN-PAIR weight")
        plt.ylabel("HN-NOT weight")

    #positional significance and covariation 
    aa_sig, pos_sig = aa_significance(model, batch_norm=batch_norm)
    if verbose:
        bar_graph(pos_sig,rrhk,'position significance',title = 'Specificity position significance')

    (aa_matrices, pos_corr) = aa_correlation(model, batch_norm=batch_norm)
    if verbose:
        heatmap(pos_corr, labels=[rrhk, rrhk])

    # remove diagonal
    diag_ind =  np.diag_indices(len(pos_corr), ndim=2)
    pos_corr_di = np.array(pos_corr)
    pos_corr_di[diag_ind] = np.min(pos_corr, axis=1)
    if verbose:
        heatmap(pos_corr_di, title = "diagonals removed", labels = [rrhk, rrhk])

    edges, inter_edges, grem_weights = load_gremlin_weights(rr_ind, hk_ind)
    inter_edge_corrs = gremlin_correlation(rr_ind, hk_ind,(edges, inter_edges), grem_weights, aa_matrices, plot = verbose)
    nn_corr = rank_pairs(pos_corr_di, rrhk,90)

    n_edges = 40
    edges = sorted(nn_corr, key= nn_corr.__getitem__, reverse = True)[:n_edges]
    pair_valX, pair_val_y = make_pairwise_features(nn_corr, edges, rr_ind, hk_ind)
    W_nn = pairwise_score_matrix(edges, aa_matrices, rrhk)
    pair_auc = pairwise_auc(pair_valX, pair_val_y, W_nn, plot = verbose)
    if verbose: 
        print n_edges, "edges -- auc = ",pair_auc
    return loss, acc, auc, pair_auc, inter_edge_corrs

def check_stored_models(params2write):
    model_list = []
    file_index = open("nn_models/file_index.tsv","r")
    i = 0
    already_stored = False
    model_number = None
    for line in file_index:
        params = tuple(line.rstrip().split("\t"))
        model_list += [params]
        if params2write == "\t".join(params[5:]):
            already_stored, model_number = 1, i
        i += 1
    file_index.close()
    return model_list,already_stored, model_number

def params_string(p,rrhk):
    params2write = "%e\t%e\t%e\t%d\t%d\t%f\t%e\t%e\t%d\t%d\t%s\t%s\t%s" % (p['lr'],
                p['decay'],
                p['lambda'] ,
                p['neurons'] ,
                p['epochs'],
                p['dropout'] ,
                p['momentum'] ,
                p['b_maxnorm'] ,
                p['batch_norm'] ,
                p['%ID'] ,
                p['activation_func'] ,
                p['comments'],
                str(rrhk))
    return params2write

def write_file_index(model_number,acc,auc,pair_auc,inter_edge_corrs, params2write):
    file_index = open("nn_models/file_index.tsv","a")
    analytics2write = "%f\t%f\t%f\t%f"% (acc,
                    auc,
                    pair_auc,
                    max(inter_edge_corrs))

    file_index.write( "%d\t%s\t%s\n"
                     % (model_number,analytics2write,params2write))
    file_index.close()
