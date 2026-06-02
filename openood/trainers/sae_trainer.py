# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.mixture import GaussianMixture
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# from openood.losses import soft_cross_entropy
# from openood.postprocessors.gmm_postprocessor import compute_single_GMM_score
# from openood.postprocessors.mds_ensemble_postprocessor import (
#     process_feature_type, reduce_feature_dim, tensor2list)
# from openood.utils import Config
#
# from .lr_scheduler import cosine_annealing
# from .mixup_trainer import mixing, prepare_mixup
#
#
# class SAETrainer:
#     def __init__(self, net: nn.Module, train_loader: DataLoader,
#                  config: Config) -> None:
#
#         self.net = net
#         self.train_loader = train_loader
#         self.config = config
#         self.trainer_args = self.config.trainer.trainer_args
#
#         self.optimizer = torch.optim.SGD(
#             net.parameters(),
#             config.optimizer.lr,
#             momentum=config.optimizer.momentum,
#             weight_decay=config.optimizer.weight_decay,
#             nesterov=True,
#         )
#
#         self.scheduler = torch.optim.lr_scheduler.LambdaLR(
#             self.optimizer,
#             lr_lambda=lambda step: cosine_annealing(
#                 step,
#                 config.optimizer.num_epochs * len(train_loader),
#                 1,
#                 1e-6 / config.optimizer.lr,
#             ),
#         )
#
#     @torch.no_grad()
#     def setup(self):
#         feature_all = None
#         label_all = []
#         # collect features
#         for batch in tqdm(self.train_loader,
#                           desc='Compute GMM Stats [Collecting]'):
#             data = batch['data'].cuda()
#             label = batch['label']
#             _, feature_list = self.net(data, return_feature_list=True)
#             label_all.extend(tensor2list(label))
#             feature_processed = process_feature_type(
#                 feature_list[-1], self.trainer_args.feature_type)
#             if isinstance(feature_all, type(None)):
#                 feature_all = tensor2list(feature_processed)
#             else:
#                 feature_all.extend(tensor2list(feature_processed))
#         label_all = np.array(label_all)
#
#         # reduce feature dim and perform gmm estimation
#         feature_all = np.array(feature_all)
#         transform_matrix = reduce_feature_dim(feature_all, label_all,
#                                               self.trainer_args.reduce_dim)
#         feature_all = np.dot(feature_all, transform_matrix)
#         # GMM estimation
#         gm = GaussianMixture(n_components=self.trainer_args.num_clusters,
#                              random_state=0,
#                             #  covariance_type='full'
#                              covariance_type='tied'
#                              ).fit(feature_all)
#         feature_mean = gm.means_
#         # feature_prec = gm.covariances_
#         feature_prec = gm.precisions_
#         component_weight = gm.weights_
#
#         self.feature_mean = torch.Tensor(feature_mean).cuda()
#         self.feature_prec = torch.Tensor(feature_prec).cuda()
#         self.component_weight = torch.Tensor(component_weight).cuda()
#         self.transform_matrix = torch.Tensor(transform_matrix).cuda()
#
#     def train_epoch(self, epoch_idx):
#         self.net.train()
#
#         loss_avg = 0.0
#         train_dataiter = iter(self.train_loader)
#
#         for train_step in tqdm(range(1,
#                                      len(train_dataiter) + 1),
#                                desc='Epoch {:03d}: '.format(epoch_idx),
#                                position=0,
#                                leave=True):
#             batch = next(train_dataiter)
#             data = batch['data'].cuda()
#             target = batch['label'].cuda()
#
#             # mixup operation
#             index, lam = prepare_mixup(batch, self.trainer_args.alpha)
#             data_mix = mixing(batch['data'].cuda(), index, lam)
#             soft_label_mix = mixing(batch['soft_label'].cuda(), index, lam)
#
#             # classfication loss
#             logits_cls, feature_list = self.net(data, return_feature_list=True)
#             loss_clsstd = F.cross_entropy(logits_cls, target)  # standard cls
#             logits_mix = self.net(data_mix)
#             loss_clsmix = soft_cross_entropy(logits_mix, soft_label_mix)
#
#             # source awareness enhancement
#             prob_id = compute_single_GMM_score(self.net, data,
#                                                self.feature_mean,
#                                                self.feature_prec,
#                                                self.component_weight,
#                                                self.transform_matrix, -1,
#                                                self.trainer_args.feature_type)
#             prob_ood = compute_single_GMM_score(self.net, data_mix,
#                                                 self.feature_mean,
#                                                 self.feature_prec,
#                                                 self.component_weight,
#                                                 self.transform_matrix, -1,
#                                                 self.trainer_args.feature_type)
#
#             loss_sae_id = torch.mean(prob_id)
#             loss_sae_ood = 1 - torch.mean(prob_ood)
#
#             # """ 拉进与类中心的距离 """
#             # pdist = nn.PairwiseDistance(p=2)
#             # feature = feature_list[-1].view(feature_list[-1].size(0), feature_list[-1].size(1))  # b, 512
#
#             # # all_cen_fea_list = [None for x in range(100)]
#             # cen_loss_list = []
#             # for i in range(10):  # 每个类
#             #     index = torch.nonzero(torch.eq(target, i)).view((-1))  # 获取每个类的index
#             #     score = prob_id.view(-1)[index]  # 根据索引获取相应得分
#             #     sort_score = torch.argsort(score, 0, descending=True)  # 对得分进行排序，获取降序得分的索引
#             #     rcp = round(len(sort_score) * 0.1)  # 中心点比例
#             #     rop = round(len(sort_score) * 0.7)  # 离群点比例
#
#             #     # all_cen_fea = feature[sort_score[: rcp]]  # 获取所有中心点向量
#             #     # if isinstance(all_cen_fea_list[i], type(None)):
#             #     #     all_cen_fea_list[i] = all_cen_fea
#             #     # for s in range(len())
#
#             #     # 计算方式一:
#             #     # all_cen_fea = feature[sort_score[ : rcp]]
#             #     # cen_list = []
#             #     # for j in range(1, rop+1):
#             #     #     ou_fea = feature[sort_score[-j]]
#             #     #     dist = pdist(ou_fea, all_cen_fea)
#             #     #     min_dist = torch.min(dist)
#             #     #     cen_list.append(min_dist)
#
#             #     # 计算方式二:
#             #     cenl_list = []
#             #     for j in range(1, rop + 1):  # 对于每个相对离群点 Relative outlier
#             #         ou_fea = feature[sort_score[-j]]  # 相对离群点特征向量
#
#             #         dist_list = []
#             #         for k in range(rcp):  # 对于每个相对中心点 Relative center point
#             #             cen_fea = feature[sort_score[k]]  # 相对中心点特征向量
#             #             dist = pdist(ou_fea, cen_fea)  # 距离计算
#             #             dist_list.append(dist)
#             #         if len(dist_list) != 0:
#             #             cenl_list.append(min(dist_list))
#
#             #     if len(cenl_list) != 0:
#             #         cen_loss_list.append(torch.stack(cenl_list, 0).mean())
#             # cen_loss = torch.stack(cen_loss_list, 0).mean()
#             # """ 拉进与类中心的距离 """
#
#             # loss
#             loss = self.trainer_args.loss_weight[0] * loss_clsstd \
#                    + self.trainer_args.loss_weight[1] * loss_clsmix \
#                 #    + self.trainer_args.loss_weight[2] * loss_sae_id \
#                 #    + self.trainer_args.loss_weight[3] * loss_sae_ood \
#                 #    + self.trainer_args.loss_weight[4] * cen_loss
#
#
#             # backward
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             self.scheduler.step()
#
#             # exponential moving average, show smooth values
#             with torch.no_grad():
#                 loss_avg = loss_avg * 0.8 + float(loss) * 0.2
#
#         metrics = {}
#         metrics['epoch_idx'] = epoch_idx
#         metrics['loss'] = loss_avg
#
#         return self.net, metrics


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from openood.losses import soft_cross_entropy
from openood.postprocessors.gmm_postprocessor import compute_single_GMM_score
from openood.postprocessors.feature_utils import (
    process_feature_type, reduce_feature_dim, tensor2list)
from openood.utils import Config

from .lr_scheduler import cosine_annealing
from .mixup_trainer import mixing, prepare_mixup


class SAETrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.trainer_args = self.config.trainer.trainer_args

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

    @torch.no_grad()
    def setup(self):
        feature_all = None
        label_all = []
        # collect features
        for batch in tqdm(self.train_loader,
                          desc='Compute GMM Stats [Collecting]'):
            data = batch['data'].cuda()
            label = batch['label']
            _, feature_list = self.net(data, return_feature_list=True)
            label_all.extend(tensor2list(label))
            feature_processed = process_feature_type(
                feature_list[-1], self.trainer_args.feature_type)
            if isinstance(feature_all, type(None)):
                feature_all = tensor2list(feature_processed)
            else:
                feature_all.extend(tensor2list(feature_processed))
        label_all = np.array(label_all)

        # reduce feature dim and perform gmm estimation
        feature_all = np.array(feature_all)
        transform_matrix = reduce_feature_dim(feature_all, label_all,
                                              self.trainer_args.reduce_dim)
        feature_all = np.dot(feature_all, transform_matrix)
        # GMM estimation
        gm = GaussianMixture(n_components=self.trainer_args.num_clusters,
                             random_state=0,
                             covariance_type='tied').fit(feature_all)
        feature_mean = gm.means_
        feature_prec = gm.precisions_
        component_weight = gm.weights_

        self.feature_mean = torch.Tensor(feature_mean).cuda()
        self.feature_prec = torch.Tensor(feature_prec).cuda()
        self.component_weight = torch.Tensor(component_weight).cuda()
        self.transform_matrix = torch.Tensor(transform_matrix).cuda()

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # mixup operation
            index, lam = prepare_mixup(batch, self.trainer_args.alpha)
            data_mix = mixing(batch['data'].cuda(), index, lam)
            soft_label_mix = mixing(batch['soft_label'].cuda(), index, lam)

            # classfication loss
            logits_cls, feature_list = self.net(data, return_feature_list=True)
            loss_clsstd = F.cross_entropy(logits_cls, target)  # standard cls
            logits_mix = self.net(data_mix)
            loss_clsmix = soft_cross_entropy(logits_mix, soft_label_mix)

            # # source awareness enhancement
            prob_id = compute_single_GMM_score(self.net, data,
                                               self.feature_mean,
                                               self.feature_prec,
                                               self.component_weight,
                                               self.transform_matrix, -1,
                                               self.trainer_args.feature_type)
            # prob_ood = compute_single_GMM_score(self.net, data_mix,
            #                                     self.feature_mean,
            #                                     self.feature_prec,
            #                                     self.component_weight,
            #                                     self.transform_matrix, -1,
            #                                     self.trainer_args.feature_type)
            # loss_sae_id = torch.mean(prob_id)
            # loss_sae_ood = torch.mean(prob_ood)

            """ 拉进与类中心的距离 """
            pdist = nn.PairwiseDistance(p=2)
            feature = feature_list[-1].view(feature_list[-1].size(0), feature_list[-1].size(1))       # b, 512

            # # feature = feature.view([feature.size(0), feature.size(1), -1])
            # feature_mean = torch.mean(feature, dim=-1)
            # feature_var = torch.var(feature, dim=-1)
            # feature = torch.cat((feature_mean, feature_var), 1)

            # all_cen_fea_list = [None for x in range(10)]
            cen_loss_list = []
            for i in range(10):  # 每个类
                index = torch.nonzero(torch.eq(target, i)).view((-1))  # 获取每个类的index
                score = prob_id.view(-1)[index]                        # 根据索引获取相应得分
                sort_score = torch.argsort(score, 0, descending=True)          # 对得分进行排序，获取降序得分的索引
                rcp = round(len(sort_score) * 0.1)                             # 中心点比例
                rop = round(len(sort_score) * 1)                             # 离群点比例

                # all_cen_fea = feature[sort_score[ : rcp]]          # 获取所有中心点向量
                # if isinstance(all_cen_fea_list[i], type(None)):
                #     all_cen_fea_list[i] = all_cen_fea
                # for s in range(len())

                # 计算方式一:
                # all_cen_fea = feature[sort_score[ : rcp]]
                # cen_list = []
                # for j in range(1, rop+1):
                #     ou_fea = feature[sort_score[-j]]
                #     dist = pdist(ou_fea, all_cen_fea)
                #     min_dist = torch.min(dist)
                #     cen_list.append(min_dist)

                # 计算方式二:
                cenl_list = []
                for j in range(1, rop+1):               # 对于每个相对离群点 Relative outlier
                    ou_fea = feature[sort_score[-j]]          # 相对离群点特征向量

                    dist_list = []
                    for k in range(rcp):          # 对于每个相对中心点 Relative center point
                        cen_fea = feature[sort_score[k]]       # 相对中心点特征向量
                        dist = pdist(ou_fea, cen_fea)     # 距离计算
                        dist_list.append(dist)
                    if len(dist_list) != 0:
                        cenl_list.append(min(dist_list))

                if len(cenl_list) != 0:
                    cen_loss_list.append(torch.stack(cenl_list, 0).mean())
            cen_loss = torch.stack(cen_loss_list, 0).mean()
            """ 拉进与类中心的距离 """

            # loss
            loss = self.trainer_args.loss_weight[0] * loss_clsstd \
                + self.trainer_args.loss_weight[1] * loss_clsmix \
                # + self.trainer_args.loss_weight[2] * cen_loss
            # + self.trainer_args.loss_weight[2] * loss_sae_id \
            # + self.trainer_args.loss_weight[3] * loss_sae_ood \

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                # loss_avg = loss_avg * 0.8 + float(loss) * 0.2
                loss_avg = float(loss)

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = loss_avg

        return self.net, metrics

    # def single_dist(self, x, y):
    #     x2 = torch.sum(x ** 2)
    #     y2 = torch.sum(y ** 2)
    #     dist = x2 + y2 - 2 * torch.matmul(x, y)
    #     dist = torch.sqrt(F.relu(dist))
    #     return dist

        # x = x.repeat(y.size(0), 1)
        # x2 = torch.sum(x ** 2, 1)
        # y2 = torch.sum(y ** 2, 1)
        # dist = x2.unsqueeze(1) + y2.unsqueeze(1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
        # dist = torch.sqrt(F.relu(dist))
        # return dist


