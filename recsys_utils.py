from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import homogeneity_score as homogeneity
import sys

from python_utils.data import count_file_lines,write_json_dump
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import layers

def get_vert_labels(iterator,news_filename,silent=False):

    news_indexes = []
    vert_vecs = []
    with tqdm(total=count_file_lines(news_filename),disable=silent) as tqdm_util:
        for batch_data_input in tqdm(iterator.load_news_from_file(news_filename)):
            vert_input = batch_data_input["candidate_vert_batch"]
            news_index = batch_data_input["news_index_batch"]
            news_indexes.extend(np.reshape(news_index, -1))
            vert_vecs.extend(np.reshape(vert_input, -1))
            tqdm_util.update(batch_data_input["count"])
            
    return dict(zip(news_indexes, vert_vecs))

def get_subvert_labels(iterator,news_filename,silent=False):

    news_indexes = []
    vert_vecs = []
    with tqdm(total=count_file_lines(news_filename),disable=silent) as tqdm_util:
        for batch_data_input in tqdm(iterator.load_news_from_file(news_filename)):
            vert_input = batch_data_input["candidate_subvert_batch"]
            news_index = batch_data_input["news_index_batch"]
            news_indexes.extend(np.reshape(news_index, -1))
            vert_vecs.extend(np.reshape(vert_input, -1))
            tqdm_util.update(batch_data_input["count"])
            
    return dict(zip(news_indexes, vert_vecs))

def eval_cluster(cluster_labels,cluster_preds):
    pred_labels=[]
    true_labels=[]
    for i in cluster_labels: #dict keys
        true_labels.append(cluster_labels[i])
        pred_labels.append(cluster_preds[i].argmax())

    return homogeneity(true_labels,pred_labels),nmi(true_labels,pred_labels),len(set(pred_labels))

def write_eval(url,dict_data,epoch=None,mode="w"):
    if epoch is not None:
        dict_data=dict([("epoch",epoch)]+list(dict_data.items()))
    write_json_dump(url,[dict_data],mode=mode)


def count_training_iters(behavior_file,npratio=0,col_spliter="\t"):
    tot = 0
    with tf.io.gfile.GFile(behavior_file, "r") as rd:
        for line in rd:
            impr = line.strip("\n").split(col_spliter)[-1].split()

            if npratio>0:
                tot += sum([i.split("-")[1] == '1' for i in impr])
            else:
                tot += len(impr)

    return tot



def model_eval(model, valid_news_file, valid_behaviors_file,test_news_file=None, test_behaviors_file=None,epoch=-1,write_mode="w"):


    eval_res=model.run_eval(valid_news_file, valid_behaviors_file,is_test=False)

    if model.group_eval:
        cluster_flag=False
        if len(eval_res)==3:
            eval_res,eval_gres,eval_clusters = eval_res
            cluster_flag=True
        else:
            eval_res, eval_gres = eval_res
        
    log_info={}
    for k,v in eval_res.items():
        log_info[f"val_{k}"]=v

    


    if test_news_file is not None:
        test_res=model.run_eval(test_news_file, test_behaviors_file,is_test=True)

        if model.group_eval:
            if cluster_flag:
                test_res,test_gres,test_clusters=test_res
            else:
                test_res,test_gres=test_res

        for k,v in test_res.items():
            log_info[f"test_{k}"]=v

    write_eval(model.hparams.eval_path,log_info,epoch=epoch,mode=write_mode)
    print(log_info)

    if model.group_eval:
        log_info = {}
        for k, v in eval_gres.items():
            log_info[f"val_{k}"] = v
        
        if cluster_flag:
            log_info["val_clusters"] = eval_clusters

        if cluster_flag:
            dec_pred = model.run_dec(valid_news_file, model.val_iterator, silent=True)
            vert_labels = get_vert_labels(model.val_iterator, valid_news_file,silent=True)
            vh, _, _ = eval_cluster(vert_labels, dec_pred)
            log_info["val_h"] = vh.round(4)
            
        if test_news_file is not None:
            for k, v in test_gres.items():
                log_info[f"test_{k}"] = v

            if cluster_flag:
                log_info["test_clusters"] = test_clusters

        write_eval(model.hparams.group_eval_path, log_info, epoch=epoch,mode=write_mode)
        print("Group Eval: ", log_info)



class ModelLayer(layers.Layer):
    def __init__(self,model):
        super(ModelLayer, self).__init__()
        self.model = model
        
    def build(self, input_shape):
        super(ModelLayer, self).build(input_shape)
        
    def call(self, inputs):
        return self.model(inputs)
    
    def compute_output_shape(self, input_shape):
        return self.model.compute_output_shape(input_shape)