from __future__ import print_function

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import math

from .base_postprocessor import BasePostprocessor
from .feature_utils import (process_feature_type,
                            reduce_feature_dim, tensor2list)


class GMMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.postprocessor_args = config.postprocessor.postprocessor_args
        self.feature_type_list = self.postprocessor_args.feature_type_list
        self.reduce_dim_list = self.postprocessor_args.reduce_dim_list
        self.num_clusters_list = self.postprocessor_args.num_clusters_list
        self.alpha_list = self.postprocessor_args.alpha_list

        self.num_layer = len(self.feature_type_list)
        self.feature_mean, self.feature_prec = None, None
        self.component_weight_list, self.transform_matrix_list = None, None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.feature_mean, self.feature_prec, self.component_weight_list, \
            self.transform_matrix_list = get_GMM_stat(net,
                                                      id_loader_dict['train'],
                                                      self.num_clusters_list,
                                                      self.feature_type_list,
                                                      self.reduce_dim_list)

    def postprocess(self, net: nn.Module, data: Any):
        for layer_index in range(self.num_layer):
            pred, score, fea = compute_GMM_score(net,
                                            data,
                                            self.feature_mean,
                                            self.feature_prec,
                                            self.component_weight_list,
                                            self.transform_matrix_list,
                                            layer_index,
                                            self.feature_type_list,
                                            return_pred=True)
            if layer_index == 0:
                score_list = score.view([-1, 1])
            else:
                score_list = torch.cat((score_list, score.view([-1, 1])), 1)
        alpha = torch.cuda.FloatTensor(self.alpha_list)
        # import pdb; pdb.set_trace();
        # conf = torch.matmul(score_list, alpha)
        conf = torch.matmul(torch.log(score_list + 1e-45), alpha)
        return pred, conf, fea


@torch.no_grad()
def get_GMM_stat(model, train_loader, num_clusters_list, feature_type_list,
                 reduce_dim_list):
    """ Compute GMM.
    Args:
        model (nn.Module): pretrained model to extract features
        train_loader (DataLoader): use all training data to perform GMM
        num_clusters_list (list): number of clusters for each layer
        feature_type_list (list): feature type for each layer
        reduce_dim_list (list): dim-reduce method for each layer

    return: feature_mean: list of class mean
            feature_prec: list of precisions
            component_weight_list: list of component
            transform_matrix_list: list of transform_matrix
    """
    feature_mean_list, feature_prec_list = [], []
    component_weight_list, transform_matrix_list = [], []
    num_layer = len(num_clusters_list)
    feature_all = [None for x in range(num_layer)]
    label_list = []
    # collect features
    for batch in tqdm(train_loader, desc='Compute GMM Stats [Collecting]'):
        data = batch['data'].cuda()
        label = batch['label']
        _, feature_list = model(data, return_feature_list=True)
        label_list.extend(tensor2list(label))
        for layer_idx in range(num_layer):
            feature_type = feature_type_list[layer_idx]
            feature_processed = process_feature_type(feature_list[layer_idx],
                                                     feature_type)
            if isinstance(feature_all[layer_idx], type(None)):
                feature_all[layer_idx] = tensor2list(feature_processed)
            else:
                feature_all[layer_idx].extend(tensor2list(feature_processed))
    label_list = np.array(label_list)
    # reduce feature dim and perform gmm estimation
    for layer_idx in tqdm(range(num_layer),
                          desc='Compute GMM Stats [Estimating]'):
        feature_sub = np.array(feature_all[layer_idx])
        transform_matrix = reduce_feature_dim(feature_sub, label_list,
                                              reduce_dim_list[layer_idx])
        feature_sub = np.dot(feature_sub, transform_matrix)
        # GMM estimation
        gm = GaussianMixture(
            n_components=num_clusters_list[layer_idx],
            random_state=0,
            covariance_type='tied',
        ).fit(feature_sub)
        feature_mean = gm.means_
        # feature_prec = gm.covariances_
        feature_prec = gm.precisions_
        component_weight = gm.weights_
        # c = gm.covariances_

        feature_mean_list.append(torch.Tensor(feature_mean).cuda())
        feature_prec_list.append(torch.Tensor(feature_prec).cuda())
        component_weight_list.append(torch.Tensor(component_weight).cuda())
        transform_matrix_list.append(torch.Tensor(transform_matrix).cuda())

    # return feature_mean_list, feature_prec_list, component_weight_list, transform_matrix_list

    """ 重新计算GMM参数, 缩小范围 """
    re_feature_mean_list, re_feature_prec_list = [], []
    re_component_weight_list, re_transform_matrix_list = [], []
    re_num_layer = len(num_clusters_list)
    re_feature_all = [None for x in range(num_layer)]
    re_label_list = []

    for batch in tqdm(train_loader, desc='Compute GMM Stats [Collecting]'):
        data_2 = batch['data'].cuda()
        label_2 = batch['label'].cuda()
        pred, score, fea = compute_GMM_score(model,
                                        data_2,
                                        feature_mean_list,
                                        feature_prec_list,
                                        component_weight_list,
                                        transform_matrix_list,
                                        -1,
                                        feature_type_list,
                                        return_pred=True)
        score = torch.log(score + 1e-45)

        # a, idx1 = torch.sort(score, 0, descending=True)  # descending为False，升序，为True，降序
        # ratio = round(data_2.size(0) * 0.005)
        # idx = idx1[:ratio].view(-1)
        # dd = data_2[idx]
        # ll = label_2[idx]


        for i in range(10):  # 每个类
            index = torch.nonzero(torch.eq(label_2, i)).view((-1))  # 获取每个类的index
            _index = score.view(-1)[index]                          # 根据索引获取相应得分
            sort_score = torch.argsort(_index, 0, descending=True)          # 对得分进行排序，获取降序得分的索引
            rcp = round(len(sort_score) * 0.3)                              # 中心点比例

            da = data_2[sort_score[:rcp]]

            temp = torch.FloatTensor(da.size()).uniform_(-0.00001, 0.00001).cuda()
            temp2 = torch.FloatTensor(da.size()).uniform_(-0.00001, 0.00001).cuda()
            temp3 = torch.FloatTensor(da.size()).uniform_(-0.00001, 0.00001).cuda()
            temp4 = torch.FloatTensor(da.size()).uniform_(-0.00001, 0.00001).cuda() 
            temp5 = torch.FloatTensor(da.size()).uniform_(-0.00001, 0.00001).cuda()

            # data_2 = torch.cat((data_2, (da + temp)), 0)
            ddd = torch.cat(((da + temp), (da + temp2), (da + temp3), (da + temp4)), 0)

        # temp = torch.FloatTensor(dd.size()).uniform_(-0.001, 0.001).cuda()
        # temp_2 = torch.FloatTensor(dd.size()).uniform_(-0.001, 0.001).cuda()
        # temp_3 = torch.FloatTensor(dd.size()).uniform_(-0.001, 0.001).cuda()
        # temp_4 = torch.FloatTensor(dd.size()).uniform_(-0.001, 0.001).cuda()

        # # dd_2 = dd + temp
        # data_2 = torch.cat((data_2, (data_2[idx] + temp), (data_2[idx] + temp_2), (data_2[idx] + temp_3), (data_2[idx] + temp_4)), 0)
        data_2 = torch.cat((data_2, ddd), 0)
        _, feature_list = model(data_2, return_feature_list=True)
        re_label_list.extend(tensor2list(label_2))
        for layer_idx in range(re_num_layer):
            feature_type = feature_type_list[layer_idx]
            feature_processed = process_feature_type(feature_list[layer_idx],
                                                     feature_type)
            if isinstance(re_feature_all[layer_idx], type(None)):
                re_feature_all[layer_idx] = tensor2list(feature_processed)
            else:
                re_feature_all[layer_idx].extend(tensor2list(feature_processed))
    re_label_list = np.array(re_label_list)
    # reduce feature dim and perform gmm estimation
    for layer_idx in tqdm(range(num_layer),
                          desc='Compute GMM Stats [Estimating]'):
        feature_sub = np.array(re_feature_all[layer_idx])

        transform_matrix = reduce_feature_dim(feature_sub, re_label_list,
                                              reduce_dim_list[layer_idx])
        feature_sub = np.dot(feature_sub, transform_matrix)
        # GMM estimation
        gm = GaussianMixture(
            n_components=num_clusters_list[layer_idx],
            random_state=0,
            covariance_type='tied',
        ).fit(feature_sub)
        feature_mean = gm.means_
        feature_prec = gm.precisions_
        component_weight = gm.weights_

        re_feature_mean_list.append(torch.Tensor(feature_mean).cuda())
        re_feature_prec_list.append(torch.Tensor(feature_prec).cuda())
        re_component_weight_list.append(torch.Tensor(component_weight).cuda())
        re_transform_matrix_list.append(torch.Tensor(transform_matrix).cuda())

    return re_feature_mean_list, re_feature_prec_list, re_component_weight_list, re_transform_matrix_list


def compute_GMM_score(model,
                      data,
                      feature_mean,
                      feature_prec,
                      component_weight,
                      transform_matrix,
                      layer_idx,
                      feature_type_list,
                      return_pred=False):
    """ Compute GMM.
    Args:
        model (nn.Module): pretrained model to extract features
        data (DataLoader): input one training batch
        feature_mean (list): a list of torch.cuda.Tensor()
        feature_prec (list): a list of torch.cuda.Tensor()
        component_weight (list): a list of torch.cuda.Tensor()
        transform_matrix (list): a list of torch.cuda.Tensor()
        layer_idx (int): index of layer in interest
        feature_type_list (list): a list of strings to indicate feature type
        return_pred (bool): return prediction and confidence, or only conf.

    return:
        pred (torch.cuda.Tensor):
        prob (torch.cuda.Tensor):
    """
    # extract features
    pred_list, feature_list = model(data, return_feature_list=True)

    fea = feature_list[-1].view(-1, feature_list[-1].size(1))

    pred = torch.argmax(pred_list, dim=1)
    feature_list = process_feature_type(feature_list[layer_idx],
                                        feature_type_list[layer_idx])
    feature_list = torch.mm(feature_list, transform_matrix[layer_idx])
    # compute prob
    for cluster_idx in range(len(feature_mean[layer_idx])):
        zero_f = feature_list - feature_mean[layer_idx][cluster_idx]
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, feature_prec[layer_idx]), zero_f.t()).diag()

        # term_gau = -0.5 * torch.mm(torch.mm(zero_f, torch.inverse(feature_prec[layer_idx][cluster_idx])), zero_f.t()).diag()
        
        prob_gau = torch.exp(term_gau)
        if cluster_idx == 0:
            prob_matrix = prob_gau.view([-1, 1])
        else:
            prob_matrix = torch.cat((prob_matrix, prob_gau.view(-1, 1)), 1)

    prob = torch.mm(prob_matrix, component_weight[layer_idx].view(-1, 1))

    if return_pred:
        return pred, prob, fea
    else:
        return prob


def compute_single_GMM_score(model,
                             data,
                             feature_mean,
                             feature_prec,
                             component_weight,
                             transform_matrix,
                             layer_idx,
                             feature_type_list,
                             return_pred=False):
    # extract features
    pred_list, feature_list = model(data, return_feature_list=True)
    pred = torch.argmax(pred_list, dim=1)
    feature_list = process_feature_type(feature_list[layer_idx],
                                        feature_type_list)
    feature_list = torch.mm(feature_list, transform_matrix)
    # compute prob
    for cluster_idx in range(len(feature_mean)):
        zero_f = feature_list - feature_mean[cluster_idx]
        term_gau = -0.5 * torch.mm(torch.mm(zero_f, feature_prec), zero_f.t()).diag()

        # term_gau = -0.5 * torch.mm(torch.mm(zero_f, torch.inverse(feature_prec[cluster_idx])), zero_f.t()).diag()
        
        prob_gau = torch.exp(term_gau)

        # prob_gau = prob_gau / torch.sqrt(torch.inverse(feature_prec).det() * (2 * math.pi)**10)

        if cluster_idx == 0:
            prob_matrix = prob_gau.view([-1, 1])
        else:
            prob_matrix = torch.cat((prob_matrix, prob_gau.view(-1, 1)), 1)
    prob = torch.mm(prob_matrix, component_weight.view(-1, 1))
    if return_pred:
        return pred, prob
    else:
        return prob
