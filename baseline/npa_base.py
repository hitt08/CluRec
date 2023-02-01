import tensorflow.keras as keras
from tensorflow.keras import layers


from recommenders.models.newsrec.models.layers import PersonalizedAttentivePooling
from tqdm import tqdm
import numpy as np
import sys

from recsys_utils import count_training_iters
from group_layers import group_score


from base_model import BaseModel
import pickle
from recommenders.models.deeprec.deeprec_utils import cal_metric


class NPAModel(BaseModel):
    """NPA model(Neural News Recommendation with Attentive Multi-View Learning)
    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie:
    NPA: Neural News Recommendation with Personalized Attention, KDD 2019, ADS track.
    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for MANL.
        Compared with the BaseModel, NPA need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train (object): NPA data loader class for train data.
            iterator_creator_test (object): NPA data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_graph(self):
        """Build NPA model and scorer.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_npa()
        return model, scorer

    def _build_userencoder(self, titleencoder, user_embedding_layer):
        """The main function to create user encoder of NPA.
        Args:
            titleencoder (object): the news encoder of NPA.
        Return:
            object: the user encoder of NPA.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        nuser_id = layers.Reshape((1, 1))(user_indexes)
        repeat_uids = layers.Concatenate(axis=-2)([nuser_id] * hparams.his_size)
        his_title_uid = layers.Concatenate(axis=-1)([his_input_title, repeat_uids])

        click_title_presents = layers.TimeDistributed(titleencoder)(his_title_uid)

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_indexes)
        )
        user_present = PersonalizedAttentivePooling(
            hparams.his_size,
            hparams.filter_num,
            hparams.attention_hidden_dim,
            seed=self.seed,
        )([click_title_presents, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer, user_embedding_layer):
        """The main function to create news encoder of NPA.
        Args:
            embedding_layer (object): a word embedding layer.
        Return:
            object: the news encoder of NPA.
        """
        hparams = self.hparams
        sequence_title_uindex = keras.Input(
            shape=(hparams.title_size + 1,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            sequence_title_uindex
        )
        user_index = layers.Lambda(lambda x: x[:, hparams.title_size :])(
            sequence_title_uindex
        )

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_index)
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)

        pred_title = PersonalizedAttentivePooling(
            hparams.title_size,
            hparams.filter_num,
            hparams.attention_hidden_dim,
            seed=self.seed,
        )([y, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        # pred_title = Reshape((1, feature_size))(pred_title)
        model = keras.Model(sequence_title_uindex, pred_title, name="news_encoder")
        return model

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

        titleencoder = self._build_newsencoder(embedding_layer, user_embedding_layer)
        userencoder = self._build_userencoder(titleencoder, user_embedding_layer)
        newsencoder = titleencoder

        user_present = userencoder([his_input_title, user_indexes])

        news_present = layers.TimeDistributed(newsencoder)(pred_title_uindex)
        news_present_one = newsencoder(pred_title_uindex_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer


##ENABLE ONLY BEFORE RUNNING BASELINE

    def run_eval(self, news_filename, behaviors_file, is_test=False, save_preds=False):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        iterator = self.test_iterator if is_test else self.val_iterator
        tmp_eval = self.run_slow_eval(
                news_filename, behaviors_file, iterator, return_verts=save_preds
            )
        if save_preds:
            _, group_labels, group_preds, group_gpreds, group_sgpreds, group_verts, group_subverts = tmp_eval
        else:
            _, group_labels, group_preds, group_gpreds, group_sgpreds = tmp_eval

        if save_preds:
            res = {}
            res["group_labels"] = group_labels
            res["group_preds"] = group_preds
            res["group_gpreds"] = group_gpreds
            res["group_sgpreds"] = group_sgpreds

            vertDict = self.train_iterator.load_dict(self.hparams.vertDict_file)
            vnum=max(vertDict.values()) + 1
            vertDict = self.train_iterator.load_dict(self.hparams.subvertDict_file)
            svnum = max(vertDict.values()) + 1
            del vertDict
            res["group_wise_preds"] = dict([(i, []) for i in range(vnum)])
            res["group_wise_gpreds"] = dict([(i, []) for i in range(vnum)])
            res["group_wise_labels"] = dict([(i, []) for i in range(vnum)])

            for vinps, pred, gpred, label in zip(group_verts, group_preds, group_gpreds, group_labels):
                for v, p, g, l in zip(vinps, pred, gpred, label):
                    res["group_wise_preds"][v].append(p)
                    res["group_wise_gpreds"][v].append(g)
                    res["group_wise_labels"][v].append(l)

            res["subgroup_wise_preds"] = dict([(i, []) for i in range(svnum)])
            res["subgroup_wise_gpreds"] = dict([(i, []) for i in range(svnum)])
            res["subgroup_wise_labels"] = dict([(i, []) for i in range(svnum)])

            for vinps, pred, gpred, label in zip(group_subverts, group_preds, group_sgpreds, group_labels):
                for v, p, g, l in zip(vinps, pred, gpred, label):
                    res["subgroup_wise_preds"][v].append(p)
                    res["subgroup_wise_gpreds"][v].append(g)
                    res["subgroup_wise_labels"][v].append(l)

            with open(self.hparams.model_weights_path.replace(".h5", f"_preds.p"), "wb") as f:
                pickle.dump(res, f)

        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        if self.group_eval:
            gres = cal_metric(group_labels, group_gpreds, self.hparams.metrics)
            # sgres = cal_metric(group_labels, group_sgpreds, self.hparams.metrics)
            # print(gres)
            # print(sgres)
            return res, gres
        else:
            return res

    def run_slow_eval(self, news_filename, behaviors_file, iterator, return_verts=False):
        print("slow eval")
        preds = []
        labels = []
        imp_indexes = []
        verts = []
        subverts = []
        with tqdm(total=count_training_iters(behaviors_file)) as tqdm_util:
            for batch_data_input in iterator.load_data_from_file(news_filename, behaviors_file):
                step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                imp_indexes.extend(np.reshape(step_imp_index, -1))
                verts.extend(np.reshape(batch_data_input["candidate_vert_batch"], -1))
                subverts.extend(np.reshape(batch_data_input["candidate_subvert_batch"], -1))

                tqdm_util.update(batch_data_input["count"])

        group_impr_indexes, group_labels, group_preds, group_verts, group_subverts  = self.group_labels(
            labels, preds, imp_indexes, verts, subverts
        )

        group_gpreds = []
        group_sgpreds = []
        if self.group_eval:
            vertDict = self.train_iterator.load_dict(self.hparams.vertDict_file)
            vnum=max(vertDict.values()) + 1
            vertDict = self.train_iterator.load_dict(self.hparams.subvertDict_file)
            svnum = max(vertDict.values()) + 1
            print(f"Clusters: {vnum}, SubCluster: {svnum}")
            del vertDict

            for pred, vinps, svinps in zip(group_preds, group_verts, group_subverts):
                # print(pred,vinps)
                gpred = group_score(np.asarray(pred), np.asarray(vinps, int), vnum)
                group_gpreds.append(gpred)

                sgpred = group_score(np.asarray(pred), np.asarray(svinps, int), svnum)
                group_sgpreds.append(sgpred)

        if return_verts:
            return group_impr_indexes, group_labels, group_preds, group_gpreds, group_sgpreds, group_verts, group_subverts
        else:
            return group_impr_indexes, group_labels, group_preds, group_gpreds, group_sgpreds

    def group_labels(self, labels, preds, group_keys,verts=None,subverts=None):
        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        if self.group_eval:
            group_verts = {k: [] for k in all_keys}
            group_subverts = {k: [] for k in all_keys}
            for label, p, v, sv, k in zip(labels, preds, verts,subverts, group_keys):
                group_labels[k].append(label)
                group_preds[k].append(p)
                group_verts[k].append(v)
                group_subverts[k].append(sv)
        else:
            for label, p, k in zip(labels, preds, group_keys):
                group_labels[k].append(label)
                group_preds[k].append(p)

        all_labels = []
        all_preds = []
        all_verts = []
        all_subverts = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
            if self.group_eval:
                all_verts.append(group_verts[k])
                all_subverts.append(group_subverts[k])

        return all_keys, all_labels, all_preds, all_verts, all_subverts