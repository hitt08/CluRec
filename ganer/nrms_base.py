import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
import tensorflow as tf
import argparse
from sklearn.cluster import KMeans
import time
import math
import os
import numpy as np
from tqdm import tqdm
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention

import sys

from python_utils.data import count_file_lines
from scipy.special import expit
from recsys_utils import get_vert_labels,count_training_iters,ModelLayer
from group_layers import group_score,WeightedAverage
from base_decmodel import DECBaseModel,ClusterLayer



class NRMSModelDECCLImprsVertWAKLGL(DECBaseModel):

    def __init__(
            self,
            hparams,
            iterator_creator,
            seed=None,
            dec_raw=False,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)

        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
            dec_raw=dec_raw
        )

    
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
        return {"recsys":data_loss,"dec_candidate":"kullback_leibler_divergence","dec_clicked":"kullback_leibler_divergence"}

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator
        Args:
            batch data: input batch data from iterator
        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
            # batch_data["candidate_vert_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator
        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        return batch_data["clicked_title_batch"]

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator
        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return batch_data["candidate_title_batch"]

    def _get_vert_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator
        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        return batch_data["candidate_title_batch"]

    def _build_graph(self):
        """Build NRMS model and scorer.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)

        news_index=train_batch_data["candidate_news_index_batch"]
        title_shape=train_input[-1].shape 
        candidate_vert_p=np.stack([self.target_dec_dist[j] for i in news_index for j in i], axis=0).reshape((title_shape[0],title_shape[1],-1))

        news_index=train_batch_data["clicked_news_index_batch"]
        title_shape=train_input[0].shape 
        clicked_vert_p=np.stack([self.target_dec_dist[j] for i in news_index for j in i], axis=0).reshape((title_shape[0],title_shape[1],-1))

        # clicked_batch=train_input[0]
        # title_shape=clicked_batch.shape 
        # clicked_vert_inp=clicked_batch.reshape((title_shape[0]*title_shape[1],title_shape[-1]))
        # clicked_vert_vec=self.dec_model.predict_on_batch(clicked_vert_inp)
        # clicked_vert_p = self.p_mat(clicked_vert_vec).reshape((title_shape[0],title_shape[1],-1))

        # candidate_batch=train_input[-1]
        # title_shape=candidate_batch.shape 
        # candidate_vert_inp=candidate_batch.reshape((title_shape[0]*title_shape[1],title_shape[-1]))
        # candidate_vert_vec=self.dec_model.predict_on_batch(candidate_vert_inp)
        # candidate_vert_p = self.p_mat(candidate_vert_vec).reshape((title_shape[0],title_shape[1],-1))

        rslt = self.model.train_on_batch(train_input, {"recsys":train_label,"dec_candidate":candidate_vert_p,"dec_clicked":clicked_vert_p})
        return rslt

    def initialize_dec(self, train_news_file=None):
        tf.compat.v1.keras.backend.set_session(self.sess)
        print('Loading pre-trained weights for auto-encoder.')
        self.autoencoder.load_weights(self.hparams.ae_pretrained_weights_path)
        
        for i in range(len(self.encoder.layers)):
            self.encoder.layers[i].set_weights(self.autoencoder.layers[i].get_weights())
            
        temp_dec_model = keras.Sequential([self.pre_encoder, self.encoder, ClusterLayer(self.hparams.n_clusters,weights=np.zeros((self.hparams.n_clusters, self.encoder.layers[-1].output_shape[1])))])
        if os.path.isfile(self.hparams.dec_pretrained_weights_path):
            temp_dec_model.load_weights(self.hparams.dec_pretrained_weights_path)
        else:
            temp_dec_model.load_weights(self.hparams.dec_init_weights_path)
            
        dec_weights=temp_dec_model.get_weights()
        dec_weights[-1]=np.vstack([np.zeros_like(dec_weights[-1][0]),dec_weights[-1]])
        self.dec_model.set_weights(dec_weights)
        

        self.cluster_centroid = self.dec_model.layers[-1].get_weights()[0]
        print('Restored DEC Model weight')

        return
    
    def update_target_dec_dist(self,news_file,iterator):
        y_pred = self.run_dec(news_file, iterator, silent=True)
        p_idx,q=[],[]
        for k,v in y_pred.items():
            p_idx.append(k)
            q.append(v)
        p_idx = np.asarray(p_idx)
        q = np.stack(q, axis=0)
        q = self.p_mat(q)
        self.target_dec_dist = dict(zip(p_idx,q))
        self.target_dec_dist[0] = np.zeros_like(self.target_dec_dist[0])  # default  nan values to 0
        self.target_dec_dist[0][0] = 1 

    # def update_target_dec_dist(self,news_file,iterator):
    #     y_pred = self.run_dec(news_file, iterator, silent=True)
    #     del y_pred[0] #skip 0th row
    #     self.q = np.stack([v for k,v in sorted(y_pred.items(),key=lambda x:x[0])], axis=0)
    #     # self.p = np.vstack([y_pred[0][None,:], self.p_mat(self.q)])
    #     self.p = self.p_mat(self.q)
    #     return y_pred


    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file,
        valid_behaviors_file,
        test_news_file=None,
        test_behaviors_file=None,
        start_epoch=1,
        dec_update_interval=None
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
        vert_labels = get_vert_labels(self.val_iterator, valid_news_file,silent=True)
        if dec_update_interval is None:
            dec_update_interval = count_file_lines(train_news_file) // self.hparams.dec_batch_size
        step_iteration=0
        self.best_loss=1e6
        for epoch in range(start_epoch, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_total_loss = 0
            epoch_recsys_loss = 0
            epoch_dec_pred_loss = 0
            epoch_dec_click_loss = 0
            train_start = time.time()

            with tqdm(total=count_training_iters(train_behaviors_file,self.hparams.npratio)) as tqdm_util:
                for batch_data_input in self.train_iterator.load_data_from_file(train_news_file, train_behaviors_file):
                    if step_iteration == 0 or step_iteration % dec_update_interval == 0:
                        self.update_target_dec_dist(train_news_file,self.train_iterator)

                    step_result = self.train(batch_data_input)
                    step_data_loss = step_result

                    epoch_total_loss += step_data_loss[0]
                    epoch_recsys_loss += step_data_loss[1]
                    epoch_dec_pred_loss += step_data_loss[2]
                    epoch_dec_click_loss += step_data_loss[3]
                    step += 1
                    if step % self.hparams.show_step == 0:
                        tqdm_util.set_description(
                            "step {0:d} , recsys_loss: {1:.4f} ({2:.4f}), dec_pred_loss: {3:.4f} ({4:.4f}), dec_click_loss: {5:.4f} ({6:.4f})".format(
                                step, epoch_recsys_loss / step, step_data_loss[0], epoch_dec_pred_loss / step, step_data_loss[1], epoch_dec_click_loss / step, step_data_loss[2]
                            )
                        )
                    tqdm_util.update(batch_data_input["count"])
                    step_iteration+=1
            if epoch==self.hparams.epochs:
                self.log_epoch_info(epoch, train_start, dict(recsys_loss=epoch_recsys_loss / step,dec_pred_loss=epoch_dec_pred_loss / step,dec_click_loss=epoch_dec_click_loss / step),valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file,save_each=True,vert_labels=vert_labels)

        return self

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.
        Args:
            titleencoder (object): the news encoder of NRMS.
        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, user_present, name="user_encoder")
        return model

    def _build_uservertencoder(self, vertencoder):
        """The main function to create user vert encoder of NRMS.
        Args:
            vertencoder (object): the vert encoder of NRMS.
        Return:
            object: the user vert encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        click_vert_presents = layers.TimeDistributed(vertencoder)(his_input_title)

        user_present = layers.Lambda(lambda x: K.mean(x, axis=-2))(click_vert_presents)

        # y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
        #     [click_title_presents] * 3
        # )
        # user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title, [user_present,click_vert_presents], name="uservert_encoder")
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.
        Args:
            embedding_layer (object): a word embedding layer.
        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        # pred_title = layers.Reshape((1, hparams.head_num * hparams.head_dim))(pred_title)


        # pred_vert = self.vertencoder(sequences_input_title)
        # concate_repr = layers.Concatenate(axis=-2)([pred_title, pred_vert])
        # combined_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)

        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")

        return model

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

           

        self.pre_encoder = keras.Model(sequences_input_vert, encoder_input, name="pre_encoder")

    
        init_centroid = np.zeros((self.hparams.n_clusters+1, self.encoder.layers[-1].output_shape[1]))
        self.dec_model = keras.Sequential([self.pre_encoder, self.encoder, ClusterLayer(self.hparams.n_clusters+1, weights=init_centroid, name='dec')])
        self.dec_model.compile(loss='kullback_leibler_divergence', optimizer=Adam(learning_rate=self.hparams.dec_learning_rate))

        # pred_vert = self.dec_model(sequences_input_vert)
        # pred_vert = layers.Dense(
        #     hparams.head_num * hparams.head_dim,
        #     activation=hparams.dense_activation,
        #     bias_initializer=keras.initializers.Zeros(),
        #     kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        # )(y)
        # pred_vert = layers.Reshape((1, hparams.head_num * hparams.head_dim))(pred_vert)     
        # model = keras.Model(sequences_input_vert, pred_vert, name="vert_encoder")

        return self.dec_model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        vert_embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        vertencoder = self._build_vertencoder(vert_embedding_layer)
        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.uservertencoder = self._build_uservertencoder(vertencoder)
        self.newsencoder = titleencoder
        self.vertencoder = vertencoder

        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        uservert_present,uservert_present_individual = self.uservertencoder(his_input_title)
        newsvert_present = layers.TimeDistributed(self.vertencoder)(pred_input_title)
        newsvert_present_one = self.vertencoder(pred_title_one_reshape)

        uservert_present_individual = layers.Lambda(lambda k: k, name='dec_clicked')(uservert_present_individual)
        newsvert_present = layers.Lambda(lambda k: k, name='dec_candidate')(newsvert_present)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        vertpreds = layers.Dot(axes=-1)([newsvert_present, uservert_present])
        preds=WeightedAverage(dims=2,name="recsys")([preds,vertpreds])
        preds = layers.Activation(activation="softmax")(preds)
        

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        vertpred_one = layers.Dot(axes=-1)([newsvert_present_one, uservert_present])
        pred_one=WeightedAverage(dims=2)([pred_one,vertpred_one])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)
        

        model = keras.Model([his_input_title, pred_input_title], [preds,newsvert_present,uservert_present_individual])
        scorer = keras.Model([his_input_title, pred_input_title_one], [pred_one,newsvert_present_one,uservert_present_individual])

        return model, scorer

    def run_dec(self, news_filename, iterator, silent=False):
        news_indexes = [0]
        vert_vecs = [np.zeros(self.hparams.n_clusters+1)]
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

    def newsvert(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        vert_vec = self.vertencoder.predict_on_batch(news_input)
        news_vec = self.newsencoder.predict_on_batch(news_input)
        news_index = batch_news_input["news_index_batch"]

        return news_index, news_vec, vert_vec
    
    def uservert(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        uservert_vec,_ = self.uservertencoder.predict_on_batch(user_input)
        user_vec = self.userencoder.predict_on_batch(user_input)
        user_index = batch_user_input["impr_index_batch"]

        return user_index, user_vec, uservert_vec

    def run_uservert(self, news_filename, behaviors_file,iterator,silent=False):
        if not hasattr(self, "uservertencoder"):
            raise ValueError("model must have attribute uservertencoder")

        user_indexes = []
        user_vecs = []
        uservert_vecs = []
        with tqdm(total=count_file_lines(behaviors_file),disable=silent) as tqdm_util:
            tqdm_util.set_description("Reading User")
            for batch_data_input in iterator.load_user_from_file(news_filename, behaviors_file):
                # user_index, user_vec = self.user(batch_data_input)
                user_index,user_vec, uservert_vec = self.uservert(batch_data_input)
                # assert np.all(user_index==uservert_index)
                user_indexes.extend(np.reshape(user_index, -1))
                user_vecs.extend(user_vec)
                uservert_vecs.extend(uservert_vec)
                tqdm_util.update(batch_data_input["count"])

        return dict(zip(user_indexes, user_vecs)),dict(zip(user_indexes, uservert_vecs))
    
    def run_newsvert(self, news_filename,iterator,silent=False):
        if not hasattr(self, "vertencoder"):
            raise ValueError("model must have attribute vertencoder")

        news_indexes = []
        news_vecs = []
        vert_vecs = []
        with tqdm(total=count_file_lines(news_filename),disable=silent) as tqdm_util:
            for batch_data_input in iterator.load_news_from_file(news_filename):
                # news_index, news_vec = self.news(batch_data_input)
                news_index,news_vec, vert_vec = self.newsvert(batch_data_input)
                # assert np.all(news_index==newsvert_index)
                news_indexes.extend(np.reshape(news_index, -1))
                news_vecs.extend(news_vec)
                vert_vecs.extend(vert_vec)
                tqdm_util.update(batch_data_input["count"])

        return dict(zip(news_indexes, news_vecs)),dict(zip(news_indexes, vert_vecs))

    def run_fast_eval(self, news_filename, behaviors_file, iterator, return_verts=False):
        print("here")
        news_vecs,vert_vecs = self.run_newsvert(news_filename,iterator, silent=True)
        user_vecs,uservert_vecs = self.run_uservert(news_filename, behaviors_file,iterator, silent=True)

        if self.group_eval:
            # vert_inps=self.run_dec(news_filename,is_test=True,silent=True)
            vert_inps = dict([(k,v.argmax()) for k,v in vert_vecs.items()])
            pred_clusters = np.unique(np.stack(list(vert_inps.values()))).shape[0]
            print(f"Number of Clusters: {pred_clusters}")

        self.news_vecs = news_vecs
        self.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []
        group_gpreds=[]
        group_verts = []
        W=self.model.get_weights()[-1]
        WS=np.sum(W)
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
            predvert = np.dot(
                np.stack([vert_vecs[i] for i in news_index], axis=0),
                uservert_vecs[impr_index],
            )
            
            pred=(pred*W[0]+predvert*W[1])/WS

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
                return group_impr_indexes, group_labels, group_preds, group_gpreds, pred_clusters, group_verts
            else:
                return group_impr_indexes, group_labels, group_preds, group_gpreds, pred_clusters
        else:
            return group_impr_indexes, group_labels, group_preds