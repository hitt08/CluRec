# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
import numpy as np
import math
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1 import keras
from scipy.special import expit
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans
import pickle
from recommenders.models.deeprec.deeprec_utils import cal_metric
import sys

from python_utils.data import count_file_lines,write
from recsys_utils import write_eval, count_training_iters, get_vert_labels, get_subvert_labels, eval_cluster
from base_model import BaseModel
from group_layers import group_score

__all__ = ["DECBaseModel"]


def get_callbacks_ae(hparams):
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: hparams.dec_learning_rate * math.pow(0.5,
                                                                    math.floor((1 + epoch) / hparams.dec_decay_step)))
    checkpoint = callbacks.ModelCheckpoint(hparams.ae_pretrained_weights_path,
                                           monitor='loss',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1,
                                           mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='loss', mode='min', patience=50)
    return [lr_decay, checkpoint, early_stopping]


class ClusterLayer(layers.Layer):

    def __init__(self, output_dim, input_dim=None, weights=None, alpha=1.0, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.alpha = alpha

        # k-means cluster centre locations
        self.initial_weights = weights
        self.input_spec = [layers.InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(ClusterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [layers.InputSpec(dtype=K.floatx(),
                                            shape=(None, input_dim))]

        self.W = K.variable(self.initial_weights)
        self._trainable_weights = [self.W]

    def call(self, x, mask=None):
        q = 1.0 / (1.0 + K.sqrt(K.sum(K.square(K.expand_dims(x, 1) - self.W), axis=2)) ** 2 / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'input_dim': self.input_dim}
        base_config = super(ClusterLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DECBaseModel(BaseModel):
    """Basic class of DEC-based models

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
        dec_raw=False,
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A HParams object, holds the entire set of hyperparameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.cluster_centroid = None
        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
        )
        self.dec_raw=dec_raw  #Use Untrained DEC
             

    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file=None,
        test_behaviors_file=None,
        start_epoch=1
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

            ##Fine-Tune DEC
            # self.dec_model.get_layer("vert_core_encoder").trainable=False #Freeze DEC Encoder to fine-tune only the cluster layer
            # self.dec_model.compile(loss='kullback_leibler_divergence',optimizer=Adam(learning_rate=self.hparams.dec_learning_rate))
            self.train_iterator.batch_size = self.hparams.dec_batch_size
            self.train_dec(train_news_file,single_epoch=True)
            # self.dec_model.get_layer("vert_core_encoder").trainable=True #UnFreeze DEC Encoder for next recommendation epoch ####?#################################[TODO]to fine-tune encoder+cluster
            # self.dec_model.compile(loss='kullback_leibler_divergence',optimizer=Adam(learning_rate=self.hparams.dec_learning_rate))
            # self.model.compile(loss=self.loss, optimizer=self.train_optimizer)
            # self.train_dec(train_news_file,single_epoch=True)
            # self.train_dec(train_news_file)
            self.train_iterator.batch_size = self.hparams.batch_size

            self.log_epoch_info(epoch, train_start, epoch_loss / step,valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file,save_each=True)

        return self

    def _build_vertencoder(self, embedding_layer):
        hparams = self.hparams

        self.encoders_dims = [hparams.word_emb_dim, 500, 500, 2000, 50]
        # self.input_layer = Input(shape=(self.input_dim,), name='input')
        self.dropout_fraction = 0.2

        self.layer_wise_autoencoders = []
        self.encoders = []
        self.decoders = []
        for i in range(1, len(self.encoders_dims)):
            encoder_activation = 'linear' if i == (len(self.encoders_dims) - 1) else 'selu'
            encoder = layers.Dense(self.encoders_dims[i], activation=encoder_activation,
                                   input_shape=(self.encoders_dims[i - 1],),
                                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=self.seed),
                                   bias_initializer='zeros', name='encoder_dense_%d' % i)
            self.encoders.append(encoder)

            decoder_index = len(self.encoders_dims) - i
            decoder_activation = 'linear' if i == 1 else 'selu'
            decoder = layers.Dense(self.encoders_dims[i - 1], activation=decoder_activation,
                                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=self.seed),
                                   bias_initializer='zeros',
                                   name='decoder_dense_%d' % decoder_index)
            self.decoders.append(decoder)

            autoencoder = keras.Sequential([
                layers.Dropout(self.dropout_fraction, input_shape=(self.encoders_dims[i - 1],),
                               name='encoder_dropout_%d' % i),
                encoder,
                layers.Dropout(self.dropout_fraction, name='decoder_dropout_%d' % decoder_index),
                decoder
            ])
            autoencoder.compile(loss='mse', optimizer=Adam(learning_rate=hparams.dec_learning_rate))
            self.layer_wise_autoencoders.append(autoencoder)

        # build the end-to-end autoencoder for fine-tuning
        # Note that at this point dropout is discarded
        self.encoder = keras.Sequential(self.encoders, name="vert_core_encoder")
        self.encoder.compile(loss='mse', optimizer=Adam(learning_rate=hparams.dec_learning_rate))
        self.decoders.reverse()
        self.autoencoder = keras.Sequential(self.encoders + self.decoders)
        self.autoencoder.compile(loss='mse', optimizer=Adam(learning_rate=hparams.dec_learning_rate))

        sequences_input_vert = keras.Input(shape=(hparams.title_size,), dtype="int32")

        embedded_sequences_vert = embedding_layer(sequences_input_vert)

        encoder_input = layers.Lambda(lambda x: K.mean(x, axis=1))(embedded_sequences_vert)

        y = self.encoder(encoder_input)
        pred_vert = layers.Dense(
            hparams.head_num * hparams.head_dim,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        pred_vert = layers.Reshape((1, hparams.head_num * hparams.head_dim))(pred_vert)        

        self.pre_encoder = keras.Model(sequences_input_vert, encoder_input, name="pre_encoder")
        model = keras.Model(sequences_input_vert, pred_vert, name="vert_encoder")

        init_centroid = np.zeros((self.hparams.n_clusters, self.encoder.layers[-1].output_shape[1]))
        self.dec_model = keras.Sequential([self.pre_encoder, self.encoder, ClusterLayer(self.hparams.n_clusters, weights=init_centroid, name='dec')])
        self.dec_model.compile(loss='kullback_leibler_divergence', optimizer=Adam(learning_rate=self.hparams.dec_learning_rate))

        return model

    def get_batched_data(self, data):

        batch_data = []
        cnt = 0

        for index in range(len(data)):
            batch_data.append(data[index])

            cnt += 1
            if cnt >= self.hparams.batch_size:
                yield np.stack(batch_data)
                batch_data = []
                cnt = 0

        if cnt > 0:
            yield np.stack(batch_data)

    def initialize_dec(self, train_news_file):
        if True:#not self.dec_raw: #pretrained DEC
            if self.hparams.ae_pretrained_weights_path is None or (not os.path.isfile(self.hparams.ae_pretrained_weights_path)):

                # print('layer-wise pre-train')
                train_data = []
                with tqdm(total=count_file_lines(train_news_file)) as tqdm_util:
                    for batch_data_input in self.train_iterator.load_news_from_file(train_news_file,start_idx=1):
                        train_input = self._get_news_feature_from_iter(batch_data_input)
                        train_input = self.pre_encoder.predict_on_batch(train_input)
                        train_data.extend(train_input)
                        tqdm_util.update(batch_data_input["count"])
                train_data = np.stack(train_data)
                current_input = train_data

                iters_per_epoch = int(len(train_data) / self.hparams.dec_batch_size)
                layerwise_epochs = max(int(self.hparams.ae_layerwise_pretrain_iters / iters_per_epoch), 1)
                finetune_epochs = max(int(self.hparams.ae_finetune_iters / iters_per_epoch), 1)

                [lr_schedule, checkpoint, early_stopping] = get_callbacks_ae(self.hparams)

                # greedy-layer wise training
                for i, autoencoder in enumerate(tqdm(self.layer_wise_autoencoders)):
                    if i > 0:
                        weights = self.encoders[i - 1].get_weights()
                        dense_layer = layers.Dense(self.encoders_dims[i], input_shape=(current_input.shape[1],),
                                                activation='selu', weights=weights,
                                                name='encoder_dense_copy_%d' % i)
                        encoder_model = keras.Sequential([dense_layer])
                        encoder_model.compile(loss='mse', optimizer=Adam(learning_rate=self.hparams.dec_learning_rate))
                        tmp_input = []
                        for batch_data_input in self.get_batched_data(current_input):
                            tmp_input.extend(encoder_model.predict_on_batch(batch_data_input))
                        current_input = np.stack(tmp_input)

                    autoencoder.fit(current_input, current_input, batch_size=self.hparams.dec_batch_size,
                                    epochs=layerwise_epochs, callbacks=[lr_schedule])
                    self.autoencoder.layers[i].set_weights(autoencoder.layers[1].get_weights())
                    self.autoencoder.layers[len(self.autoencoder.layers) - i - 1].set_weights(
                        autoencoder.layers[-1].get_weights())

                print('Fine-tuning auto-encoder')

                # ae_train_model=self._ae_pretrainer(self.autoencoder)
                # update encoder and decoder weights:
                self.autoencoder.fit(train_data, train_data,
                                    batch_size=self.hparams.dec_batch_size,
                                    epochs=finetune_epochs,
                                    callbacks=[lr_schedule, checkpoint, early_stopping])
            else:
                print('Loading pre-trained weights for auto-encoder.')
                self.autoencoder.load_weights(self.hparams.ae_pretrained_weights_path)

            # update encoder, decoder

            for i in range(len(self.encoder.layers)):
                self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())

        if not self.dec_raw and (
                    (self.hparams.dec_init_weights_path is not None and os.path.isfile(self.hparams.dec_init_weights_path)) or
                    (self.hparams.dec_pretrained_weights_path is not None and os.path.isfile(self.hparams.dec_pretrained_weights_path))):

            if os.path.isfile(self.hparams.dec_pretrained_weights_path):
                self.dec_model.load_weights(self.hparams.dec_pretrained_weights_path)
            else:
                self.dec_model.load_weights(self.hparams.dec_init_weights_path)

            self.cluster_centroid = self.dec_model.layers[-1].get_weights()[0]
            print('Restored DEC Model weight')
        else:  # initialize cluster centres using k-means
            print('Initializing cluster centres with k-means.')
            if self.cluster_centroid is None:
                kmeans = KMeans(n_clusters=self.hparams.n_clusters, n_init=20, random_state=self.seed)
                vert_vecs = self.run_base_encoder(train_news_file, iterator=self.train_iterator)
                del vert_vecs[0]
                x_data = np.stack([v for v in vert_vecs.values()], axis=0)
                self.y_pred = kmeans.fit_predict(x_data)
                self.cluster_centroid = kmeans.cluster_centers_

            # initial centroid using K-mean
            self.dec_model.layers[-1].set_weights([self.cluster_centroid])
            self.dec_model.save_weights(self.hparams.dec_init_weights_path)

        return

    def p_mat(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def cluster(self, news_file_name, test="train", tol=0.001, iter_max=1e5,update_interval=None, **kwargs):

        if test == "test":
            y_pred = self.run_dec(news_file_name, iterator=self.val_iterator)
            y_pred = np.stack([v.argmax(1) for v in y_pred], axis=0)
            return y_pred
        if update_interval is None:
            update_interval = count_file_lines(news_file_name) // self.hparams.dec_batch_size
        print('Update interval', update_interval)

        train = True
        iteration = 0
        current_acc = 0
        self.accuracy = 0
        # epochs=iter_max//self.hparams.dec_batch_size
        self.train_iterator.batch_size = self.hparams.dec_batch_size
        news_verts = get_vert_labels(self.train_iterator, news_file_name)
        news_subverts = get_subvert_labels(self.train_iterator, news_file_name)
        del news_verts[0]
        del news_subverts[0]
        with tqdm(total=iter_max) as tqdm_util:
            while train:
                train,iteration=self.train_dec(news_file_name,tol=tol,iter_max=iter_max, iteration=iteration,update_interval=update_interval,tqdm_util=tqdm_util,save_checkpoint=True,news_verts=news_verts,news_subverts=news_subverts)
            self.dec_model.save_weights(self.hparams.dec_pretrained_weights_path)

        self.train_iterator.batch_size = self.hparams.batch_size
        return self.y_pred

    def update_target_dec_dist(self,news_file,iterator):
        y_pred = self.run_dec(news_file, iterator, silent=True)
        del y_pred[0] #skip 0th row
        self.q = np.stack([v for k,v in sorted(y_pred.items(),key=lambda x:x[0])], axis=0)
        # self.p = np.vstack([y_pred[0][None,:], self.p_mat(self.q)])
        self.p = self.p_mat(self.q)
        return y_pred

    def train_dec(self,news_file_name,tol=0.001,iter_max=None, iteration=0,update_interval=None,tqdm_util=None,save_checkpoint=False, single_epoch=False,news_verts=None,news_subverts=None):
        
        if tqdm_util is None:
            tqdm_util=tqdm(total=count_file_lines(news_file_name))
            tqdm_close_flag=True
        else:
            tqdm_close_flag=False

        if update_interval is None:
            update_interval = count_file_lines(news_file_name) // self.hparams.dec_batch_size
        save_interval = 1000

        if not hasattr(self,"y_pred"):
            y_pred = self.run_dec(news_file_name, iterator=self.train_iterator, silent=True)
            self.y_pred = np.stack([v.argmax() for v in y_pred.values()], axis=0)

        step = 0
        epoch_loss = 0
        train=True

        for batch_data_input in self.train_iterator.load_news_from_file(news_file_name,start_idx=1):
            # cut off iteration
            if iter_max is not None and iter_max < iteration:
                print('Reached maximum iteration limit. Stopping training.')
                train = False
                break

            if iteration == 0 or iteration % update_interval == 0:
                # update (or initialize) probability distributions and propagate weight changes
                # from DEC model to encoder.
                # y_pred = self.run_dec(news_file_name, self.train_iterator, silent=True)
                # self.q=np.stack([v for k,v in sorted(y_pred.items(),key=lambda x:x[0])], axis=0)
                # self.p = self.p_mat(self.q)
                y_pred=self.update_target_dec_dist(news_file_name,self.train_iterator)
                if news_verts is not None and news_subverts is not None:
                    vh, vn, pred_clusters = eval_cluster(news_verts, y_pred)
                    svh, svn, _ = eval_cluster(news_subverts, y_pred)

                y_pred = self.q.argmax(1)
                if iteration>0:
                    delta_label = (np.sum((y_pred != self.y_pred)).astype(np.float32) / y_pred.shape[0])

                    # pred_clusters = np.unique(np.stack(y_pred)).shape[0]
                    print(f"{np.round(delta_label * 100, 5)}% change in label assignment. Pred Clusters:{pred_clusters}")

                if news_verts is not None and news_subverts is not None:
                    print(f"Vert: H={vh}, NMI={vn}\t SubVert: H={svh}, NMI={svn}")

                if iteration>0 and delta_label < tol:  # and pred_clusters>(2/3)*self.hparams.n_clusters:
                    print('Reached tolerance threshold.')
                    train = False
                    break
                else:
                    self.y_pred = y_pred

                # weight changes if current
                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.dec_model.layers[1].layers[i].get_weights())
                self.cluster_centroid = self.dec_model.layers[-1].get_weights()[0]

                # save checkpoint
                if save_checkpoint and iteration % save_interval:
                    self.dec_model.save_weights(self.hparams.dec_pretrained_weights_path)

            # train on batch
            # print('Iteration %d, ' % iteration)

            train_input = self._get_news_feature_from_iter(batch_data_input)
            # batch_q = self.dec_model.predict_on_batch(train_input)
            # batch_p = self.p_mat(batch_q)
            # self.update_target_dec_dist(news_file_name,self.train_iterator)
            batch_p=self.p[batch_data_input["news_index_batch"]-1]   #p doesn't have 0th news and news_index starts from the 1st news
            step_result = self.dec_model.train_on_batch(train_input, batch_p)
            step_data_loss = step_result

            epoch_loss += step_data_loss
            step += 1
            if step % self.hparams.show_step == 0:
                tqdm_util.set_description(
                    "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                        step, epoch_loss / step, step_data_loss
                    )
                )

            iteration += 1
            if tqdm_close_flag:
                tqdm_util.update(batch_data_input["count"])
            else:
                tqdm_util.update(1)
        
        if train and single_epoch: #Save last batch
            y_pred = self.update_target_dec_dist(news_file_name, self.train_iterator)
            vh, vn, pred_clusters = eval_cluster(news_verts, y_pred)
            svh, svn, _ = eval_cluster(news_subverts, y_pred)

            y_pred = self.q.argmax(1)
            delta_label = (np.sum((y_pred != self.y_pred)).astype(np.float32) / y_pred.shape[0])

            pred_clusters = np.unique(np.stack(y_pred)).shape[0]
            print(f"{np.round(delta_label * 100, 5)}% change in label assignment. Pred Clusters:{pred_clusters}")
            print(f"Vert: H={vh}, NMI={vn}\t SubVert: H={svh}, NMI={svn}")

            if delta_label >= tol:  # and pred_clusters>(2/3)*self.hparams.n_clusters:
                self.y_pred = y_pred

                # weight changes if current
                for i in range(len(self.encoder.layers)):
                    self.encoder.layers[i].set_weights(self.dec_model.layers[1].layers[i].get_weights())
                self.cluster_centroid = self.dec_model.layers[-1].get_weights()[0]

        if tqdm_close_flag:
            tqdm_util.close()

        return train,iteration

    def run_eval(self, news_filename, behaviors_file, is_test=False,save_preds=False):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """
        iterator = self.test_iterator if is_test else self.val_iterator
        if self.support_quick_scoring:
            tmp_eval = self.run_fast_eval(
                    news_filename, behaviors_file, iterator,return_verts=save_preds
                )
            if self.group_eval:
                if save_preds:
                    _, group_labels, group_preds, group_gpreds, clusters,group_verts = tmp_eval
                else:
                    _, group_labels, group_preds, group_gpreds, clusters = tmp_eval
            else:
                _, group_labels, group_preds = tmp_eval
        else:
            tmp_eval = self.run_slow_eval(
                news_filename, behaviors_file,iterator,return_verts=save_preds
            )
            if save_preds:
                _, group_labels, group_preds,group_gpreds, clusters,group_verts = tmp_eval
            else:
                _, group_labels, group_preds, group_gpreds, clusters = tmp_eval

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
            return res, gres, clusters
        else:
            return res

    def run_slow_eval(self, news_filename, behaviors_file, iterator,return_verts=False):
        print("slow eval")
        preds = []
        labels = []
        imp_indexes = []
        verts=[]
        with tqdm(total=count_training_iters(behaviors_file)) as tqdm_util:
            for batch_data_input in iterator.load_data_from_file(news_filename, behaviors_file):
                step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
                if len(step_pred)==3:
                    step_pred,step_vert,_ = step_pred
                else:
                    step_pred, step_vert = step_pred
                step_vert=np.argmax(step_vert,axis=-1)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                imp_indexes.extend(np.reshape(step_imp_index, -1))
                verts.extend(step_vert)

                tqdm_util.update(batch_data_input["count"])

        group_impr_indexes, group_labels, group_preds, group_verts = self.group_labels(
            labels, preds, imp_indexes, verts
        )

        group_gpreds=[]
        pred_clusters=[]
        if self.group_eval:
            for pred,vinps in zip(group_preds, group_verts):
                # print(pred,vinps)
                gpred=group_score(np.asarray(pred),np.asarray(vinps,int),self.hparams.eval_clusters)
                group_gpreds.append(gpred)
                pred_clusters.extend(vinps)
            pred_clusters=len(set(pred_clusters))
            print(f"Number of Clusters: {pred_clusters}")

        if return_verts:
            return group_impr_indexes, group_labels, group_preds, group_gpreds,pred_clusters, group_verts
        else:
            return group_impr_indexes, group_labels, group_preds, group_gpreds,pred_clusters

    def dec(self, batch_news_input):
        vert_input = self._get_news_feature_from_iter(batch_news_input)

        vert_vec = self.dec_model.predict_on_batch(vert_input)
        news_index = batch_news_input["news_index_batch"]

        return news_index, vert_vec

    def run_dec(self, news_filename, iterator, silent=False):
        news_indexes = [0]
        vert_vecs = [np.zeros(self.hparams.n_clusters)]
        # iterator = self.test_iterator if is_test else self.train_iterator

        # orig_batch=self.train_iterator.batch_size
        # self.train_iterator.batch_size = 1024
        with tqdm(total=count_file_lines(news_filename), disable=silent) as tqdm_util:
            for batch_data_input in iterator.load_news_from_file(news_filename,start_idx=1):
                news_index, vert_vec = self.dec(batch_data_input)
                news_indexes.extend(np.reshape(news_index, -1))
                vert_vecs.extend(vert_vec)
                tqdm_util.update(batch_data_input["count"])

        # self.train_iterator.batch_size = orig_batch
        return dict(zip(news_indexes, vert_vecs))

    def base_encoder(self, batch_news_input):
        vert_input = self._get_news_feature_from_iter(batch_news_input)

        y = self.pre_encoder.predict_on_batch(vert_input)
        vert_vec = self.encoder.predict_on_batch(y)
        news_index = batch_news_input["news_index_batch"]

        return news_index, vert_vec

    def run_base_encoder(self, news_filename, iterator):
        if not hasattr(self, "encoder"):
            raise ValueError("model must have attribute encoder")

        news_indexes = [0]
        vert_vecs = [np.zeros(self.hparams.n_clusters)]
        # iterator = self.test_iterator if is_test else self.train_iterator
        with tqdm(total=count_file_lines(news_filename)) as tqdm_util:
            for batch_data_input in iterator.load_news_from_file(news_filename,start_idx=1):
                news_index, vert_vec = self.base_encoder(batch_data_input)
                news_indexes.extend(np.reshape(news_index, -1))
                vert_vecs.extend(vert_vec)
                tqdm_util.update(batch_data_input["count"])

        return dict(zip(news_indexes, vert_vecs))

    def run_fast_eval(self, news_filename, behaviors_file, iterator,return_verts=False):
        news_vecs = self.run_news(news_filename,iterator, silent=True)
        user_vecs = self.run_user(news_filename, behaviors_file,iterator, silent=True)

        if self.group_eval:
            vert_inps = self.run_dec(news_filename, iterator, silent=True)
            vert_inps = dict([(k, v.argmax()) for k, v in vert_inps.items()])

            pred_clusters = np.unique(np.stack(list(vert_inps.values()))).shape[0]
            print(f"Number of Clusters: {pred_clusters}")

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
        ) in tqdm(iterator.load_impression_from_file(behaviors_file), total=count_file_lines(behaviors_file)):
            pred = np.dot(
                np.stack([news_vecs[i] for i in news_index], axis=0),
                user_vecs[impr_index],
            )

            if self.group_eval:
                vinps = np.stack([vert_inps[i] for i in news_index], axis=0).reshape(-1)
                gpred = group_score(expit(pred), vinps, self.hparams.eval_clusters)
                group_gpreds.append(gpred)
                group_verts.append(vinps)

            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            group_preds.append(pred)

        if self.group_eval:
            if return_verts:
                return group_impr_indexes, group_labels, group_preds, group_gpreds, pred_clusters, group_verts
            else:
                return group_impr_indexes, group_labels, group_preds, group_gpreds, pred_clusters
        else:
            return group_impr_indexes, group_labels, group_preds