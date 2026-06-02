import csv
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from .ood_evaluator import OODEvaluator


class FSOODEvaluator(OODEvaluator):
    """
    Full-Spectrum OOD evaluator with additional robustness analysis on
    covariate-shifted in-distribution (CSID) data.

    New metric:
        FRR@95ID: false rejection rate on CSID data when the threshold is
        chosen on clean ID data so that 95% of clean ID samples are accepted.

    Assumption:
        Higher confidence score means more ID-like.
    """

    def _get_csid_tpr(self):
        if hasattr(self.config, 'evaluator') and hasattr(self.config.evaluator, 'csid_tpr'):
            return float(self.config.evaluator.csid_tpr)
        return 0.95

    def _get_csid_threshold_source(self):
        if hasattr(self.config, 'evaluator') and hasattr(self.config.evaluator, 'csid_threshold_source'):
            return str(self.config.evaluator.csid_threshold_source)
        return 'test'

    def _get_threshold_from_id(self, id_conf: np.ndarray, tpr: float = 0.95):
        """
        Choose threshold beta so that TPR on clean ID is tpr.
        Since higher score means more ID-like, beta is the (1 - tpr)-quantile.
        """
        id_conf = np.asarray(id_conf)
        beta = np.quantile(id_conf, 1.0 - tpr)
        return float(beta)

    def _compute_frr(self, conf: np.ndarray, beta: float):
        """
        False rejection rate on ID data:
            FRR = P(score < beta | x is ID)
        """
        conf = np.asarray(conf)
        return float(np.mean(conf < beta))

    def _compute_acc(self, pred: np.ndarray, gt: np.ndarray):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        return float(np.mean(pred == gt))

    def _save_csid_results(self, dataset_name, acc, frr):
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '-',
            'AUROC': '-',
            'AUPR_IN': '-',
            'AUPR_OUT': '-',
            'ACC': '{:.2f}'.format(100 * acc),
            'FRR@95ID': '{:.2f}'.format(100 * frr),
        }

        fieldnames = list(write_content.keys())

        print(
            'CSID[{}] accuracy: {:.2f}%, FRR@95ID: {:.2f}%'.format(
                dataset_name, 100 * acc, 100 * frr),
            flush=True)

        csv_path = os.path.join(self.config.output_dir, 'csid.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def eval_csid_metrics(self,
                          csid_cache: Dict[str, Dict[str, np.ndarray]],
                          beta: float):
        """
        Compute accuracy and FRR on each CSID dataset using a threshold
        determined from clean ID data.
        """
        acc_list, frr_list = [], []

        for dataset_name, item in csid_cache.items():
            acc = self._compute_acc(item['pred'], item['gt'])
            frr = self._compute_frr(item['conf'], beta)

            acc_list.append(acc)
            frr_list.append(frr)

            if self.config.recorder.save_csv:
                self._save_csid_results(dataset_name, acc, frr)

        if len(acc_list) > 0:
            mean_acc = float(np.mean(acc_list))
            mean_frr = float(np.mean(frr_list))

            if self.config.recorder.save_csv:
                self._save_csid_results('csid_mean', mean_acc, mean_frr)

            print('CSID mean accuracy: {:.2f}%'.format(100 * mean_acc), flush=True)
            print('CSID mean FRR@95ID: {:.2f}%'.format(100 * mean_frr), flush=True)

        print(u'\u2500' * 70, flush=True)

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loader: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor):
        # ensure the networks in eval mode
        net.eval()

        assert 'test' in id_data_loader, \
            'id_data_loaders should have the key: test!'
        assert 'csid' in ood_data_loaders, \
            'ood_data_loaders should have the key: csid!'

        dataset_name = self.config.dataset.name

        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt, fea = postprocessor.inference(
            net, id_data_loader['test'])

        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, fea, dataset_name)

        # # ------------------------------------------------------------------
        # # 1) determine threshold beta on CLEAN ID only
        # # ------------------------------------------------------------------
        # threshold_source = self._get_csid_threshold_source()
        # tpr = self._get_csid_tpr()
        #
        # if threshold_source in id_data_loader:
        #     print(
        #         f'Computing clean-ID threshold from split="{threshold_source}" '
        #         f'with target TPR={tpr:.2f}...',
        #         flush=True)
        #     _, id_conf_for_thr, _, fea = postprocessor.inference(
        #         net, id_data_loader[threshold_source])
        # else:
        #     print(
        #         f'Split "{threshold_source}" not found in id_data_loader, '
        #         f'fallback to "test".',
        #         flush=True)
        #     id_conf_for_thr = id_conf
        #
        # beta = self._get_threshold_from_id(id_conf_for_thr, tpr=tpr)
        # print(f'Clean-ID threshold beta = {beta:.6f}', flush=True)
        # print(u'\u2500' * 70, flush=True)
        #
        # ------------------------------------------------------------------
        # 2) run inference on CSID and cache predictions / scores / labels
        # ------------------------------------------------------------------
        csid_cache = {}

        for csid_name, csid_dl in ood_data_loaders['csid'].items():
            print(f'Performing inference on {csid_name} dataset...', flush=True)
            csid_pred, csid_conf, csid_gt, fea = postprocessor.inference(net, csid_dl)

            if self.config.recorder.save_scores:
                self._save_scores(csid_pred, csid_conf, csid_gt, fea, csid_name)

            csid_cache[csid_name] = {
                'pred': np.asarray(csid_pred),
                'conf': np.asarray(csid_conf),
                'gt': np.asarray(csid_gt),
            }

            # for standard full-spectrum near/far OOD evaluation:
            # covariate-shifted ID should be included into the ID pool
            id_pred = np.concatenate([id_pred, csid_pred])
            id_conf = np.concatenate([id_conf, csid_conf])
            id_gt = np.concatenate([id_gt, csid_gt])
        #
        # # ------------------------------------------------------------------
        # # 3) report CSID robustness metrics
        # # ------------------------------------------------------------------
        # self.eval_csid_metrics(csid_cache, beta)

        # ------------------------------------------------------------------
        # 4) standard near-OOD / far-OOD evaluation under full-spectrum
        # ------------------------------------------------------------------
        print(u'\u2500' * 70, flush=True)
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split='nearood')

        print(u'\u2500' * 70, flush=True)
        self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split='farood')


# import csv
# import os
# from typing import Dict, List
#
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
#
# from openood.postprocessors import BasePostprocessor
#
# from .ood_evaluator import OODEvaluator
#
#
# class FSOODEvaluator(OODEvaluator):
#     def eval_csid_acc(self, net: nn.Module,
#                       csid_loaders: Dict[str, Dict[str, DataLoader]]):
#         # ensure the networks in eval mode
#         net.eval()
#         for dataset_name, csid_dl in csid_loaders.items():
#             print(f'Computing accuracy on {dataset_name} dataset...')
#             correct = 0
#             with torch.no_grad():
#                 for batch in csid_dl:
#                     data = batch['data'].cuda()
#                     target = batch['label'].cuda()
#                     # forward
#                     output, fea = net(data, return_feature_list=True)
#                     # accuracy
#                     pred = output.data.max(1)[1]
#                     correct += pred.eq(target.data).sum().item()
#             acc = correct / len(csid_dl.dataset)
#             if self.config.recorder.save_csv:
#                 self._save_acc_results(acc, dataset_name)
#                 self._save_fea(fea[-1].cpu().numpy(), dataset_name)
#         print(u'\u2500' * 70, flush=True)
#
#     def _save_acc_results(self, acc, dataset_name):
#         write_content = {
#             'dataset': dataset_name,
#             'FPR@95': '-',
#             'AUROC': '-',
#             'AUPR_IN': '-',
#             'AUPR_OUT': '-',
#             'ACC': '{:.2f}'.format(100 * acc),
#         }
#         fieldnames = list(write_content.keys())
#         # print csid metric results
#         print('CSID[{}] accuracy: {:.2f}%'.format(dataset_name, 100 * acc),
#               flush=True)
#         csv_path = os.path.join(self.config.output_dir, 'csid.csv')
#         if not os.path.exists(csv_path):
#             with open(csv_path, 'w', newline='') as csvfile:
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerow(write_content)
#         else:
#             with open(csv_path, 'a', newline='') as csvfile:
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 writer.writerow(write_content)
#
#     def eval_ood(self, net: nn.Module, id_data_loader: List[DataLoader],
#                  ood_data_loaders: List[DataLoader],
#                  postprocessor: BasePostprocessor):
#         # ensure the networks in eval mode
#         net.eval()
#         # load training in-distribution data
#         assert 'test' in id_data_loader, \
#             'id_data_loaders should have the key: test!'
#         dataset_name = self.config.dataset.name
#         print(f'Performing inference on {dataset_name} dataset...', flush=True)
#         id_pred, id_conf, id_gt, fea = postprocessor.inference(
#             net, id_data_loader['test'])
#
#
#         # """ 保存feature"""
#         # for batch in id_data_loader['test']:
#         #     data = batch['data'].cuda()
#         #     _, fea = net(data, return_feature_list=True)
#         # if self.config.recorder.save_scores:
#         #     self._save_fea(fea[-1].view(-1, 512).cpu().detach().numpy(), dataset_name)
#         # """ 保存feature"""
#
#
#         if self.config.recorder.save_scores:
#             self._save_scores(id_pred, id_conf, id_gt, fea, dataset_name)
#
#         # load csid data and compute confidence
#         for dataset_name, csid_dl in ood_data_loaders['csid'].items():
#             print(f'Performing inference on {dataset_name} dataset...',
#                   flush=True)
#             csid_pred, csid_conf, csid_gt, fea = postprocessor.inference(
#                 net, csid_dl)
#             if self.config.recorder.save_scores:
#                 self._save_scores(csid_pred, csid_conf, csid_gt, fea, dataset_name)
#
#             # """ 保存feature"""
#             # for batch in csid_dl:
#             #     data = batch['data'].cuda()
#             #     _, fea = net(data, return_feature_list=True)
#             # if self.config.recorder.save_scores:
#             #     self._save_fea(fea[-1].view(-1, 512).cpu().detach().numpy(), dataset_name)
#             # """ 保存feature"""
#
#             id_pred = np.concatenate([id_pred, csid_pred])
#             id_conf = np.concatenate([id_conf, csid_conf])
#             id_gt = np.concatenate([id_gt, csid_gt])
#
#
#         # compute accuracy on csid
#         print(u'\u2500' * 70, flush=True)
#         self.eval_csid_acc(net, ood_data_loaders['csid'])
#
#         # load nearood data and compute ood metrics
#         print(u'\u2500' * 70, flush=True)
#         self._eval_ood(net, [id_pred, id_conf, id_gt],
#                        ood_data_loaders,
#                        postprocessor,
#                        ood_split='nearood')
#
#         # load farood data and compute ood metrics
#         print(u'\u2500' * 70, flush=True)
#         self._eval_ood(net, [id_pred, id_conf, id_gt],
#                        ood_data_loaders,
#                        postprocessor,
#                        ood_split='farood')
