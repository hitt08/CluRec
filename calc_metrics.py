import os
import argparse
import sys
import logging

import pickle
from recommenders.models.deeprec.deeprec_utils import ndcg_score,roc_auc_score,mrr_score
from recsys_metrics import precision, recall
import torch
import numpy as np
from tqdm import tqdm

def eval_metrics(labels, preds, k_list=[5, 10, 30, 50]):
    ndcg, p = {}, {}
    mrr, auc = [], []
    for each_labels, each_preds in tqdm(zip(labels, preds),total=len(preds)):
        for k in k_list:
            if k not in ndcg:
                ndcg[k] = []
            ndcg[k].append(ndcg_score(each_labels, each_preds, k))

            if k not in p:
                p[k] = []
            p[k].append(precision(torch.tensor(each_preds), torch.tensor(each_labels), k))

        auc.append(roc_auc_score(each_labels, each_preds))
        mrr.append(mrr_score(each_labels, each_preds))

    return ndcg, p, mrr, auc

def get_mean_eval(metrics_dict,out_dict,prefix=""):
    for k,v in metrics_dict.items():
        if type(v)==dict:
            for i, j in v.items():
                out_dict[f"{prefix}{k}@{i}"] = np.mean(j)
        else:
            out_dict[f"{prefix}{k}"] = np.mean(v)

    return out_dict



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='mind_type', default="small", type=str, help='MIND type: demo, small, large')
    parser.add_argument('-f', action='store', dest='file_path', type=str)
    args = parser.parse_args()

    log = logging.getLogger()
    log.warning(args)



    data_path = f"/nfs/mind/{args.mind_type}"

    with open(os.path.join(data_path,args.file_path),"rb") as f:
        model_pred=pickle.load(f)

    raw_metrics={}

    if "group_preds" in model_pred:
        log.warning("INFO: Eval Preds")
        ndcg, p, mrr, auc = eval_metrics(model_pred["group_labels"], model_pred["group_preds"])
        raw_metrics["nogrp"] = dict(ndcg=ndcg, p=p, mrr=mrr, auc=auc)

    if "group_gpreds" in model_pred:
        log.warning("INFO: Eval Group Preds")
        ndcg, p, mrr, auc = eval_metrics(model_pred["group_labels"], model_pred["group_gpreds"])
        raw_metrics["grp"] = dict(ndcg=ndcg, p=p, mrr=mrr, auc=auc)

    if "group_sgpreds" in model_pred:
        log.warning("INFO: Eval SubGroup Preds")
        ndcg, p, mrr, auc = eval_metrics(model_pred["group_labels"], model_pred["group_sgpreds"])
        raw_metrics["subgrp"] = dict(ndcg=ndcg, p=p, mrr=mrr, auc=auc)


    ext = os.path.splitext(args.file_path)[-1]
    out_file = os.path.join(data_path, args.file_path.replace(ext, "_raw_metrics.p"))
    log.warning(f"INFO: Writing raw results to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(raw_metrics, f)



    log.warning("INFO: Calculating Mean")
    modal_metrics = {}
    for k,v in raw_metrics.items():
        prefix = "" if k == "nogrp" else k
        modal_metrics = get_mean_eval(v, modal_metrics,prefix=prefix)

    out_file = os.path.join(data_path, args.file_path.replace(ext, "_metrics.p"))
    log.warning(f"INFO: Writing results to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(modal_metrics, f)

    print(modal_metrics)