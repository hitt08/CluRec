# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1 import keras

from scipy.special import expit
from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.newsrec.newsrec_utils import random
from collections.abc import Iterable
import sys

from python_utils.data import count_file_lines,write
from recsys_utils import eval_cluster, write_eval, count_training_iters, get_vert_labels, get_subvert_labels
from group_layers import group_score
import pickle
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (HParams): A HParams object, holds the entire set of hyperparameters.
        train_iterator (object): An iterator to load the data in training steps.
        test_iterator (object): An iterator to load the data in testing steps.
        graph (object): An optional graph.
        seed (int): Random seed.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
        load_body=False,
        bert_model = None
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A HParams object, holds the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)
        # tf.keras.utils.set_random_seed(seed)
        # tf.random.set_seed(seed)
        # tf.keras.mixed_precision.set_global_policy('mixed_float16')

        rnd_gen=np.random.RandomState(seed)
        random.seed(seed)

        self.train_iterator = iterator_creator(
            hparams,
            hparams.npratio,
            col_spliter="\t",
            rnd_gen=rnd_gen,
            load_body=load_body,
            bert_model=bert_model
        )
        self.val_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
            rnd_gen=rnd_gen,
            load_body=load_body,
            bert_model=bert_model
        )

        self.test_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
            rnd_gen=rnd_gen,
            load_body=load_body,
            bert_model=bert_model
        )

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring

        self.graph=tf.compat.v1.get_default_graph()
        # set GPU use with on demand growth
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(graph=self.graph,
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )

        # set this TensorFlow session as the default session for Keras
        tf.compat.v1.keras.backend.set_session(sess)
        self.sess=sess

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        self.model, self.scorer = self._build_graph()

        self.loss = self._get_loss()
        self.loss_weights = self._get_loss_weights()
        self.train_optimizer = self._get_opt()

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer,loss_weights=self.loss_weights)

        self.group_eval = True if hasattr(hparams, "group_eval") and hparams.group_eval else False
        #and self.support_quick_scoring
            

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained glove embeddings file path.

        Returns:
            numpy.ndarray: A constant numpy array.
        """

        return np.load(file_path)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """Subclass will implement this"""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            object: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_loss_weights(self):
        """ Default loss weight """
        return None

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(lr=lr)

        return train_opt

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.

        Args:
            logit (object): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            object: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value, predicted scores, and ground-truth labels.
        """
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]

        pred_rslt = self.scorer.predict_on_batch(eval_input)

        return pred_rslt, eval_label, imp_index

    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file=None,
        test_behaviors_file=None,
        start_epoch=1,
        eval_every_epoch=True
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_news_file (str): test set.

        Returns:
            object: An instance of self.
        """
        self.best_loss=1e6
        for epoch in range(start_epoch, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            with tqdm(total=count_training_iters(train_behaviors_file,self.hparams.npratio)) as tqdm_util:
                for batch_data_input in self.train_iterator.load_data_from_file(train_news_file, train_behaviors_file):

                    step_result = self.train(batch_data_input)
                    step_data_loss = step_result

                    epoch_loss += step_data_loss
                    step += 1
                    if step % self.hparams.show_step == 0:
                        tqdm_util.set_description(
                            "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                                step, epoch_loss / step, step_data_loss
                            )
                        )
                    tqdm_util.update(batch_data_input["count"])
            if eval_every_epoch or epoch==self.hparams.epochs:
                self.log_epoch_info(epoch, train_start, epoch_loss / step,valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file)

        return self

    def log_epoch_info(self,epoch,train_start,train_loss,valid_news_file, valid_behaviors_file,test_news_file=None,test_behaviors_file=None,save_each=True,vert_labels=None):
        
        train_time = time.time() - train_start
        eval_start = time.time()

        if isinstance(train_loss,Iterable):
            if type(train_loss)==dict:
                loss_dict=train_loss
            else:
                loss_dict=dict(recsys_loss=train_loss[0],dec_loss=train_loss[1])
        else:
            loss_dict={"train_loss":train_loss}

            
        train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in loss_dict.items()#[("logloss loss", train_loss)]
                ]
            )

        if self.hparams.epochs==epoch: #last epoch
            eval_res = self.run_eval(valid_news_file, valid_behaviors_file,is_test=False,save_preds=True)
        else:
            eval_res = self.run_eval(valid_news_file, valid_behaviors_file,is_test=False)
        

        if self.group_eval:
            cluster_flag=False
            if len(eval_res)==3:
                eval_res,eval_gres,eval_clusters = eval_res
                cluster_flag=True
            else:
                eval_res, eval_gres = eval_res
        
        eval_info = ", ".join(
            [
                str(item[0]) + ":" + str(item[1])
                for item in sorted(eval_res.items(), key=lambda x: x[0])
            ]
        )
        if self.group_eval:
            eval_ginfo = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_gres.items(), key=lambda x: x[0])
                ]
            )
        if test_news_file is not None:
            test_res = self.run_eval(test_news_file, test_behaviors_file,is_test=True)
            if self.group_eval:
                if cluster_flag:
                    test_res,test_gres,test_clusters = test_res
                else:
                    test_res, test_gres = test_res

            test_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(test_res.items(), key=lambda x: x[0])
                ]
            )
            if self.group_eval:
                test_ginfo = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_gres.items(), key=lambda x: x[0])
                    ]
                )
        eval_end = time.time()
        eval_time = eval_end - eval_start

        print_info= "at epoch {0:d}".format(epoch)
        print_info+="\ntrain info: " + train_info
        print_info+="\neval info: " + eval_info
        if self.group_eval:
            print_info+="\ngroup_eval info: " + eval_ginfo

        if test_news_file is not None:
            print_info+="\ntest info: " + test_info
            if self.group_eval:
                print_info+="\ngroup_test info: " + test_ginfo
        
        print(print_info)
        print(
            "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                epoch, train_time, eval_time
            )
        )
        
        #Update Best Epoch
        tmp_loss=list(loss_dict.values())[0]
        if tmp_loss<self.best_loss:
            print(f"loss improved from {self.best_loss} to {tmp_loss}. Saving checkpoint.")
            self.best_loss=tmp_loss
            
            if save_each:
                self.model.save_weights(self.hparams.model_weights_path.replace(".h5",f"_{epoch}.h5"))
            else:
                self.model.save_weights(self.hparams.model_weights_path)
            
            if hasattr(self,"dec_model"):
                if save_each:
                    self.dec_model.save_weights(self.hparams.dec_weights_path.replace(".h5",f"_{epoch}.h5"))
                else:
                    self.dec_model.save_weights(self.hparams.dec_weights_path)
                

            write(self.hparams.last_epoch_path,str(epoch))
        
            self.model.save_weights(self.hparams.model_weights_path.replace(".h5",f"_{epoch}.h5"))
            
            if hasattr(self,"dec_model"):
                self.dec_model.save_weights(self.hparams.dec_weights_path.replace(".h5",f"_{epoch}.h5"))

        log_info=dict(train_time=train_time,eval_time=eval_time)
        for k,v in loss_dict.items():
            log_info[k]=v
        for k,v in eval_res.items():
            log_info[f"val_{k}"]=v
        if test_news_file is not None:
            for k,v in test_res.items():
                log_info[f"test_{k}"]=v
        write_eval(self.hparams.eval_path,log_info,epoch=epoch,mode="a")

        if self.group_eval:
            log_info=dict(train_time=train_time,eval_time=eval_time)
            for k,v in loss_dict.items():
                log_info[k]=v
            for k,v in eval_gres.items():
                log_info[f"val_{k}"]=v
            
            if cluster_flag:
                log_info["val_clusters"] = eval_clusters

                if vert_labels is not None:
                    dec_pred=self.run_dec(valid_news_file, self.val_iterator, silent=True)
                    vh, _, _ = eval_cluster(vert_labels, dec_pred)
                    log_info["val_h"] = vh.round(4)

            if test_news_file is not None:
                for k,v in test_gres.items():
                    log_info[f"test_{k}"]=v

                if cluster_flag:
                    log_info["test_clusters"] = test_clusters

            write_eval(self.hparams.group_eval_path,log_info,epoch=epoch,mode="a")


    def group_labels(self, labels, preds, group_keys,verts=None):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            list, list, list:
            - Keys after group.
            - Labels after group.
            - Preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        if self.group_eval:
            group_verts = {k: [] for k in all_keys}
            for label, p, v, k in zip(labels, preds, verts, group_keys):
                group_labels[k].append(label)
                group_preds[k].append(p)
                group_verts[k].append(v)
        else:
            for label, p, k in zip(labels, preds, group_keys):
                group_labels[k].append(label)
                group_preds[k].append(p)

        all_labels = []
        all_preds = []
        all_verts = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
            if self.group_eval:
                all_verts.append(group_verts[k])

        return all_keys, all_labels, all_preds, all_verts

    def run_eval(self, news_filename, behaviors_file,is_test=False,save_preds=False):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        iterator=self.test_iterator if is_test else self.val_iterator
        if self.support_quick_scoring:
            tmp_eval = self.run_fast_eval(
                    news_filename, behaviors_file,iterator,return_verts=save_preds
            )
            if self.group_eval:
                if save_preds:
                    _, group_labels, group_preds, group_gpreds,group_verts = tmp_eval
                else:
                    _, group_labels, group_preds, group_gpreds = tmp_eval
            else:
                _, group_labels, group_preds = tmp_eval
        else:
            tmp_eval = self.run_slow_eval(
                    news_filename, behaviors_file,iterator,return_verts=save_preds
                )
            if save_preds:
                _, group_labels, group_preds, group_gpreds,group_verts = tmp_eval
            else:
                _, group_labels, group_preds, group_gpreds = tmp_eval
        
        if save_preds:
            res={}
            res["group_labels"]=group_labels
            res["group_preds"]=group_preds
            res["group_gpreds"]=group_gpreds
            
            res["group_wise_preds"]=dict([(i,[]) for i in range(self.hparams.eval_clusters)])
            res["group_wise_gpreds"]=dict([(i,[]) for i in range(self.hparams.eval_clusters)])
            res["group_wise_labels"]=dict([(i,[]) for i in range(self.hparams.eval_clusters)])
            
            for vinps,pred,gpred,label in zip(group_verts,group_preds,group_gpreds,group_labels):
                for v,p,g,l in zip(vinps,pred,gpred,label):
                    res["group_wise_preds"][v].append(p)
                    res["group_wise_gpreds"][v].append(g)
                    res["group_wise_labels"][v].append(l)
            with open(self.hparams.model_weights_path.replace(".h5",f"_preds.p"),"wb") as f:
                pickle.dump(res,f)

        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        if self.group_eval:
            gres = cal_metric(group_labels, group_gpreds, self.hparams.metrics)
            return res,gres
        else:
            return res

    def user(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        user_vec = self.userencoder.predict_on_batch(user_input)
        user_index = batch_user_input["impr_index_batch"]

        return user_index, user_vec

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        news_vec = self.newsencoder.predict_on_batch(news_input)
        news_index = batch_news_input["news_index_batch"]

        return news_index, news_vec

    def run_user(self, news_filename, behaviors_file,iterator,silent=False):
        if not hasattr(self, "userencoder"):
            raise ValueError("model must have attribute userencoder")

        user_indexes = []
        user_vecs = []
        with tqdm(total=count_file_lines(behaviors_file),disable=silent) as tqdm_util:
            tqdm_util.set_description("Reading User")
            for batch_data_input in iterator.load_user_from_file(news_filename, behaviors_file):
                user_index, user_vec = self.user(batch_data_input)
                user_indexes.extend(np.reshape(user_index, -1))
                user_vecs.extend(user_vec)
                tqdm_util.update(batch_data_input["count"])

        return dict(zip(user_indexes, user_vecs))

    def run_news(self, news_filename,iterator,silent=False):
        if not hasattr(self, "newsencoder"):
            raise ValueError("model must have attribute newsencoder")

        news_indexes = []
        news_vecs = []
        with tqdm(total=count_file_lines(news_filename),disable=silent) as tqdm_util:
            tqdm_util.set_description("Reading News")
            for batch_data_input in iterator.load_news_from_file(news_filename):
                news_index, news_vec = self.news(batch_data_input)
                news_indexes.extend(np.reshape(news_index, -1))
                news_vecs.extend(news_vec)
                tqdm_util.update(batch_data_input["count"])

        return dict(zip(news_indexes, news_vecs))

    def run_slow_eval(self, news_filename, behaviors_file, iterator,return_verts=False):
        print("slow eval")
        preds = []
        labels = []
        imp_indexes = []
        verts=[]
        with tqdm(total=count_training_iters(behaviors_file)) as tqdm_util:
            for batch_data_input in iterator.load_data_from_file(news_filename, behaviors_file):
                step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                imp_indexes.extend(np.reshape(step_imp_index, -1))
                verts.extend(np.reshape(batch_data_input["candidate_vert_batch"], -1))

                tqdm_util.update(batch_data_input["count"])

        group_impr_indexes, group_labels, group_preds, group_verts = self.group_labels(
            labels, preds, imp_indexes, verts
        )
        
        group_gpreds=[]
        if self.group_eval:
            for pred,vinps in zip(group_preds, group_verts):
                # print(pred,vinps)
                gpred=group_score(np.asarray(pred),np.asarray(vinps,int),self.hparams.eval_clusters)
                group_gpreds.append(gpred)

        if return_verts:
            return group_impr_indexes, group_labels, group_preds, group_gpreds, group_verts
        else:
            return group_impr_indexes, group_labels, group_preds, group_gpreds

    def run_fast_eval(self, news_filename, behaviors_file,iterator,return_verts=False,silent=True):
        news_vecs = self.run_news(news_filename,iterator,silent=silent)
        user_vecs = self.run_user(news_filename, behaviors_file,iterator,silent=silent)

        if self.group_eval:
            vert_inps=get_vert_labels(iterator,news_filename,silent=True)

        self.news_vecs = news_vecs
        self.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []
        group_gpreds = []
        group_verts = []

        for (
            impr_index,
            news_index,
            user_index,
            label,
        ) in tqdm(iterator.load_impression_from_file(behaviors_file),total=count_file_lines(behaviors_file)):
            pred = np.dot(
                np.stack([news_vecs[i] for i in news_index], axis=0),
                user_vecs[impr_index],
            )
            
            if self.group_eval:
                vinps=np.stack([vert_inps[i] for i in news_index], axis=0).reshape(-1)
                gpred=group_score(expit(pred),vinps,self.hparams.eval_clusters)
                group_gpreds.append(gpred)
                group_verts.append(vinps)

            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            group_preds.append(pred)

        if self.group_eval:
            if return_verts:
                return group_impr_indexes, group_labels, group_preds, group_gpreds, group_verts
            else:
                return group_impr_indexes, group_labels, group_preds, group_gpreds
        else:
            return group_impr_indexes, group_labels, group_preds

    def run_fast_vert_svert_eval(self, news_filename, behaviors_file, iterator, return_verts=False, silent=True):
        news_vecs = self.run_news(news_filename, iterator, silent=silent)
        user_vecs = self.run_user(news_filename, behaviors_file, iterator, silent=silent)

        if self.group_eval:
            vert_inps = get_vert_labels(iterator, news_filename, silent=True)
            subvert_inps = get_subvert_labels(iterator, news_filename, silent=True)
            vnum = max(vert_inps.values()) + 1
            svnum = max(subvert_inps.values()) + 1
            print(f"Vert:{vnum}, SVert: {svnum}")

        self.news_vecs = news_vecs
        self.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []
        group_gpreds = []
        group_sgpreds = []
        group_verts = []
        group_subverts = []

        for (
                impr_index,
                news_index,
                user_index,
                label,
        ) in tqdm(iterator.load_impression_from_file(behaviors_file), total=count_file_lines(behaviors_file)):
            pred = np.dot(
                np.stack([news_vecs[i] for i in news_index], axis=0),
                user_vecs[impr_index],
            )

            if self.group_eval:
                vinps = np.stack([vert_inps[i] for i in news_index], axis=0).reshape(-1)
                svinps = np.stack([subvert_inps[i] for i in news_index], axis=0).reshape(-1)
                gpred = group_score(expit(pred), vinps, vnum)
                sgpred = group_score(expit(pred), svinps, svnum)
                group_gpreds.append(gpred)
                group_sgpreds.append(sgpred)
                group_verts.append(vinps)
                group_subverts.append(svinps)

            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            group_preds.append(pred)

        if self.group_eval:
            if return_verts:
                return group_impr_indexes, group_labels, group_preds, group_gpreds, group_sgpreds, group_verts, group_subverts, vnum, svnum
            else:
                return group_impr_indexes, group_labels, group_preds, group_gpreds, group_sgpreds
        else:
            return group_impr_indexes, group_labels, group_preds
