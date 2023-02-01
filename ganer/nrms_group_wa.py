import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse
import sys

from group_layers import WeightedAverage
from recsys_utils import get_vert_labels, model_eval, eval_cluster, write_eval
from iterator import MINDCatIterator as catiterator
from init_params import nrms_init_params
import numpy as np
from ganer.nrms_base import NRMSModelDECCLImprsVertWAKLGL


class GroupScore(layers.Layer):
    def __init__(self, nclusters, name=None):  # seed):
        self.nclusters = nclusters
        super(GroupScore, self).__init__(name=name)

    def build(self, input_shape):
        self.W = self.add_weight("hist_weight",shape=(2,),dtype=tf.float32,trainable=True,initializer=keras.initializers.ones())

    def call(self, inputs):
        nclusters = self.nclusters
        pr = inputs[0]
        inp_means = inputs[1]
        vr = tf.stop_gradient(inputs[2])

        vr = tf.squeeze(vr, -1)

        vmeans, vmaxs, vmins = [], [], []
        for g in range(nclusters):
            gidx = tf.equal(vr, g)
            vmins.append(K.min(tf.where(gidx, pr, np.inf), axis=-1))
            vmaxs.append(K.max(tf.where(gidx, pr, -np.inf), axis=-1))

            tmp_mean=tf.reduce_sum(tf.where(gidx, pr, 0), axis=-1) / tf.cast(tf.reduce_any(gidx, axis=-1), dtype=pr.dtype)
            vmeans.append(tf.where(tf.math.is_nan(tmp_mean),0.0,tmp_mean))

        vmeans = tf.transpose(tf.stack(vmeans))
        vmins = tf.transpose(tf.stack(vmins))
        vmaxs = tf.transpose(tf.stack(vmaxs))

        vmeans = (inp_means * self.W[0] + vmeans * self.W[1]) / (self.W[0] + self.W[1])

        # To avoid nans for groups with single item
        no_scale_idx = tf.equal(vmins, vmaxs)
        vmins = tf.gather(tf.where(no_scale_idx, tf.cast(0.0, dtype=vmins.dtype), vmins), vr,batch_dims=1)
        vmaxs = tf.gather(tf.where(no_scale_idx, tf.cast(1.0, dtype=vmaxs.dtype), vmaxs), vr,batch_dims=1)

        # # Replace nan with min/2 for Convulation
        # nan_idx = tf.math.is_nan(vmeans)
        # vmeans = tf.where(nan_idx, K.min(tf.gather(vmeans, tf.where(tf.logical_not(nan_idx)))) / 2, vmeans)

        sort_ids = tf.argsort(vmeans)
        orig_ids = tf.argsort(sort_ids)

        vmeans_sorted = tf.gather(vmeans, sort_ids,batch_dims=1)

        conv_data = tf.concat([vmeans_sorted[:, :1], vmeans_sorted, vmeans_sorted[:, -1:] * 2], axis=1)
        conv_k = tf.ones(2, dtype=conv_data.dtype)

        # mm lower limit
        # f0 = tf.squeeze(tf.nn.conv1d(tf.reshape(conv_data, [1, -1, 1]), tf.reshape(conv_k / 1.99, [-1, 1, 1]), 1, padding="VALID")[0][:-1])
        f0 = tf.squeeze(tf.nn.conv1d(tf.reshape(conv_data, [tf.shape(conv_data)[0], -1, 1]), tf.reshape(conv_k / 1.999, [-1, 1, 1]), 1,padding="VALID")[:, :-1])

        # mm upper limit
        # f1 = tf.squeeze(tf.nn.conv1d(tf.reshape(conv_data, [1, -1, 1]), tf.reshape(conv_k / 2.01, [-1, 1, 1]), 1, padding="VALID")[0][1:])
        f1 = tf.squeeze(tf.nn.conv1d(tf.reshape(conv_data, [tf.shape(conv_data)[0], -1, 1]), tf.reshape(conv_k / 2.001, [-1, 1, 1]),1, padding="VALID")[:, 1:])

        f0 = tf.gather(tf.where(no_scale_idx, tf.cast(0.0, dtype=f0.dtype), tf.gather(f0, orig_ids,batch_dims=1)), vr,batch_dims=1)
        f1 = tf.gather(tf.where(no_scale_idx, tf.cast(1.0, dtype=f1.dtype), tf.gather(f1, orig_ids,batch_dims=1)), vr,batch_dims=1)


        ##Min Max Scaling
        # tmp = ((tf.stop_gradient(pr) - tf.stop_gradient(vmins)) / tf.stop_gradient(vmaxs - vmins)) * tf.stop_gradient(f1 - f0) + tf.stop_gradient(f0)
        tmp = ((tf.stop_gradient(pr) - tf.stop_gradient(vmins)) / tf.stop_gradient(vmaxs - vmins)) * (f1 - f0) + (f0)

        scale = pr / tmp
        scale = tf.where(tf.logical_or(tf.math.is_nan(scale), tf.math.is_inf(scale)), 0.0, scale)
        #
        res = pr * scale

        return tf.ensure_shape(res, [None, pr.shape[1]])


class NRMSGroupModelDECCLImprsVertWAKLGL(NRMSModelDECCLImprsVertWAKLGL):

    def __init__(
            self,
            hparams,
            iterator_creator,
            seed=None,
            dec_raw=False,
    ):

        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
            dec_raw=dec_raw
        )

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator
        Args:
            batch data: input batch data from iterator
        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        
        # title_shape=batch_data["candidate_title_batch"].shape
        # candidate_vert_inp=batch_data["candidate_title_batch"].reshape((title_shape[0]*title_shape[1],title_shape[-1]))
        # candidate_vert_vec=self.dec_model.predict_on_batch(candidate_vert_inp)
        # candidate_vert_batch = candidate_vert_vec.argmax(1).reshape((title_shape[0],title_shape[1],1))

        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
            # candidate_vert_batch,
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32", name="uhist"
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
            name="candidate"
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
        newsvert_present_one = layers.Lambda(lambda k: k, name='dec_candidate')(newsvert_present_one)

        pred_input_vert = layers.Lambda(lambda x: K.argmax(tf.stop_gradient(x), axis=-1))(newsvert_present)
        pred_input_vert = layers.Reshape((-1,1))(pred_input_vert)
        pred_input_vert_one = layers.Lambda(lambda x: K.argmax(tf.stop_gradient(x), axis=-1))(newsvert_present_one)
        pred_input_vert_one = layers.Reshape((-1,1))(pred_input_vert_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        vertpreds = layers.Dot(axes=-1)([newsvert_present, uservert_present])
        preds=WeightedAverage(dims=2)([preds,vertpreds])
        preds = layers.Activation(activation="softmax")(preds)
        preds = GroupScore(hparams.eval_clusters,name="recsys")([preds,uservert_present,pred_input_vert])
        
        

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        vertpred_one = layers.Dot(axes=-1)([newsvert_present_one, uservert_present])
        pred_one=WeightedAverage(dims=2)([pred_one,vertpred_one])
        pred_one = layers.Activation(activation="sigmoid",name="rec_pred")(pred_one)
        # pred_one = GroupScore(hparams.eval_clusters)([pred_one,pred_input_vert_one])
        

        model = keras.Model([his_input_title, pred_input_title], [preds,newsvert_present,uservert_present_individual])
        scorer = keras.Model([his_input_title, pred_input_title_one], [pred_one,newsvert_present_one,uservert_present_individual])
              

        return model, scorer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='mind_type', default="demo", type=str, help='MIND type: demo, small, large')
    parser.add_argument('-e', action='store', dest='epochs', default=10, type=int, help='epochs')
    parser.add_argument('-b', action='store', dest='batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-c', action='store', dest='n_clusters', type=int, default=0, help='number of clusters')
    parser.add_argument('-u', action='store', dest='update_interval', type=int, default=None, help='DEC update Interval')
    parser.add_argument('--upct', action='store', dest='update_interval_pct', type=float, default=None, help='Percentage of input for DEC update Interval')
    parser.add_argument('--dec_batch', action='store', dest='dec_batch', type=int, help='DEC batch size')
    parser.add_argument('--dec_lr', action='store', dest='dec_lr', type=float, default=0.001, help='DEC learning rate')
    parser.add_argument('--dec_ae_iters', dest='dec_ae_iters', action='store', type=int, default=10000,
                        help='DEC AE pretrain iters')
    parser.add_argument('--dec_ae_fineiters', dest='dec_ae_fineiters', action='store', type=int, default=20000,
                        help='DEC AE finetrain iters')
    parser.add_argument('--dec_pretrain', dest='dec_pretrain', action='store_true', help='pretrain DEC')
    parser.add_argument('--dec_untrained', dest='dec_raw', action='store_true', help='use untrained DEC')
    parser.set_defaults(dec_pretrain=False)
    parser.set_defaults(dec_raw=False)
    args = parser.parse_args()
    print(args)

    seed = 42
    hparams,file_paths=nrms_init_params(args,script_file=__file__,seed=seed,init_dec=True)

    train_news_file     = file_paths["train_news_file"]
    train_behaviors_file= file_paths["train_behaviors_file"] 
    valid_news_file     = file_paths["valid_news_file"]
    valid_behaviors_file= file_paths["valid_behaviors_file"]
    test_news_file      = file_paths["test_news_file"]
    test_behaviors_file = file_paths["test_behaviors_file"] 

    iterator = catiterator

    model = NRMSGroupModelDECCLImprsVertWAKLGL(hparams, iterator, seed=seed,dec_raw=args.dec_raw)
    model.support_quick_scoring=False
    model.val_iterator.batch_size = 512
    model.test_iterator.batch_size = 512

    model.initialize_dec(train_news_file)

    if args.dec_pretrain:
        model.cluster(train_news_file)


        # Train Eval
        news_pred_verts = model.run_dec(train_news_file, model.train_iterator)
        news_verts = get_vert_labels(model.train_iterator, train_news_file)
        h, n, c = eval_cluster(news_verts, news_pred_verts)
        res = f"Train: H={h}, NMI={n}, Clusters:{c}"
        print(res)
        write_eval(hparams.dec_eval_path, res)

        # Test Eval
        news_pred_verts = model.run_dec(valid_news_file, model.test_iterator)
        news_verts = get_vert_labels(model.test_iterator, valid_news_file)
        h, n, c = eval_cluster(news_verts, news_pred_verts)
        res = (f"Test: H={h}, NMI={n}, Clusters:{c}")
        print(res)
        write_eval(hparams.dec_eval_path, res, mode="a")


    # print(model.hparams.eval_clusters)
    model_eval(model, valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file, epoch=-1)

    if not args.dec_pretrain:
        model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file,test_news_file,test_behaviors_file)