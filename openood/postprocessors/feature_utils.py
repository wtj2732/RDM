import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def tensor2list(x):
    return x.data.cpu().tolist()


def get_torch_feature_stat(feature, only_mean=False):
    feature = feature.view([feature.size(0), feature.size(1), -1])
    feature_mean = torch.mean(feature, dim=-1)
    feature_var = torch.var(feature, dim=-1)
    if feature.size(-2) * feature.size(-1) == 1 or only_mean:
        return feature_mean
    return torch.cat((feature_mean, feature_var), 1)


def process_feature_type(feature_temp, feature_type):
    if feature_type == 'flat':
        return feature_temp.view([feature_temp.size(0), -1])
    if feature_type == 'stat':
        return get_torch_feature_stat(feature_temp)
    if feature_type == 'mean':
        return get_torch_feature_stat(feature_temp, only_mean=True)
    raise ValueError(f'Unknown feature type: {feature_type}')


def reduce_feature_dim(feature_list_full, label_list_full, feature_process):
    if feature_process == 'none':
        return np.eye(feature_list_full.shape[1])
    feature_process, kept_dim = feature_process.split('_')
    kept_dim = int(kept_dim)
    if feature_process == 'pca':
        pca = PCA(n_components=kept_dim)
        pca.fit(feature_list_full)
        return pca.components_.T
    if feature_process == 'lda':
        lda = LinearDiscriminantAnalysis(solver='eigen')
        lda.fit(feature_list_full, label_list_full)
        return lda.scalings_[:, :kept_dim]
    if feature_process == 'capca':
        # Minimal fallback: original OpenOOD uses a custom InverseLDA; this command path uses pca_10.
        lda = LinearDiscriminantAnalysis(solver='eigen')
        lda.fit(feature_list_full, label_list_full)
        return lda.scalings_[:, :kept_dim]
    raise Exception(f'Unknown feature reduction type: {feature_process}')
