import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse
import sys

from group_layers import  WeightedAverage
from recsys_utils import get_vert_labels, model_eval, eval_cluster, write_eval
from iterator import MINDCatIterator as catiterator
from init_params import nrms_init_params
from submit_predition_tf import write_prediction_file

from ganer.npa_base import NPAModelDECCLImprsVertWAKLGL
from ganer.nrms_group_wa import GroupScore

class NPAGroupModelDECCLImprsVertWAKLGL(NPAModelDECCLImprsVertWAKLGL):

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



    def _build_npa(self):
        """The main function to create NPA's logic. The core of NPA
        is a user encoder and a news encoder.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and predict.
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
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_index = layers.Reshape((1, 1))(user_indexes)
        repeat_uindex = layers.Concatenate(axis=-2)(
            [nuser_index] * (hparams.npratio + 1)
        )
        pred_title_uindex = layers.Concatenate(axis=-1)(
            [pred_input_title, repeat_uindex]
        )
        pred_title_uindex_one = layers.Concatenate()(
            [pred_title_one_reshape, user_indexes]
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        user_embedding_layer = layers.Embedding(
            len(self.train_iterator.uid2index),
            hparams.user_emb_dim,
            trainable=True,
            embeddings_initializer="zeros",
        )

        vert_embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        vertencoder = self._build_vertencoder(vert_embedding_layer)

        titleencoder = self._build_newsencoder(embedding_layer, user_embedding_layer)
        userencoder = self._build_userencoder(titleencoder, user_embedding_layer)
        newsencoder = titleencoder
        self.uservertencoder = self._build_uservertencoder(vertencoder)
        self.vertencoder = vertencoder

        user_present = userencoder([his_input_title, user_indexes])

        news_present = layers.TimeDistributed(newsencoder)(pred_title_uindex)
        news_present_one = newsencoder(pred_title_uindex_one)

        uservert_present,uservert_present_individual = self.uservertencoder(his_input_title)
        newsvert_present = layers.TimeDistributed(self.vertencoder)(pred_input_title)
        newsvert_present_one = self.vertencoder(pred_title_one_reshape)

        uservert_present_individual = layers.Lambda(lambda k: k, name='dec_clicked')(uservert_present_individual)
        newsvert_present = layers.Lambda(lambda k: k, name='dec_candidate')(newsvert_present)

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
        pred_one = layers.Activation(activation="sigmoid")(pred_one)
        # pred_one = GroupScore(hparams.eval_clusters)([pred_one,pred_input_vert_one])
        

        model = keras.Model([user_indexes, his_input_title, pred_input_title], [preds,newsvert_present,uservert_present_individual])
        scorer = keras.Model([user_indexes, his_input_title, pred_input_title_one], [pred_one,newsvert_present_one])
              
        return model, scorer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='mind_type', default="demo", type=str, help='MIND type: demo, small, large')
    parser.add_argument('-e', action='store', dest='epochs', default=10, type=int, help='epochs')
    # parser.add_argument('-b', action='store', dest='batch_size', default=32, type=int, help='batch size')
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
    hparams,file_paths=nrms_init_params(args,script_file=__file__,seed=seed,init_dec=True,model_type="npa")

    train_news_file     = file_paths["train_news_file"]
    train_behaviors_file= file_paths["train_behaviors_file"] 
    valid_news_file     = file_paths["valid_news_file"]
    valid_behaviors_file= file_paths["valid_behaviors_file"]
    test_news_file      = file_paths["test_news_file"]
    test_behaviors_file = file_paths["test_behaviors_file"] 

    iterator = catiterator
    hparams.title_size= 30
    model = NPAGroupModelDECCLImprsVertWAKLGL(hparams, iterator, seed=seed,dec_raw=args.dec_raw)
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

    # model_eval(model, valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file, epoch=-1)

    if not args.dec_pretrain:
        model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file,test_news_file,test_behaviors_file)

        if args.mind_type=="large":
            test_news_file= valid_news_file.replace("valid","test")
            test_behaviors_file= valid_behaviors_file.replace("valid","test")
            model.test_iterator.is_test = True
            tmp_eval = model.run_slow_eval(
                test_news_file, test_behaviors_file, model.test_iterator,return_verts=False
            )
            group_impr_indexes, group_labels, group_preds, group_gpreds, clusters = tmp_eval
                    
            write_prediction_file(hparams.model_weights_path.replace(".h5","_test_preds.txt"),group_impr_indexes, group_preds)
            write_prediction_file(hparams.model_weights_path.replace(".h5","_test_gpreds.txt"),group_impr_indexes, group_gpreds)