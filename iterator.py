import tensorflow as tf
import numpy as np
import pickle
import os
import sys
from recommenders.models.deeprec.io.iterator import BaseIterator
from recommenders.models.newsrec.newsrec_utils import word_tokenize #, newsample, random

from python_utils.data import read_json
# from transformers import BertTokenizer,AutoTokenizer
from pytorch.tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
import torch

def newsample(news, ratio, random_state):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random_state.choice(news, ratio, replace=False).tolist()

class MINDCatIterator(BaseIterator):
    """Train data loader for NAML model.
    The model require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts.
    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.
    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def __init__(
        self,
        hparams,
        npratio=-1,
        col_spliter="\t",
        ID_spliter="%",
        rnd_gen=None,
        is_test=False,
        load_body=False,
        dec_static_vec=False,
        bert_model=None,
        is_pytorch=False
    ):
        """Initialize an iterator. Create necessary placeholders for the model.
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
            col_spliter (str): column spliter in one line.
            ID_spliter (str): ID spliter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.batch_size = hparams.batch_size
        self.title_size = hparams.title_size
        if hasattr(hparams,"body_size"):
            self.body_size = hparams.body_size
        else:
            self.body_size=50
        self.his_size = hparams.his_size
        self.npratio = npratio

        self.word_dict = self.load_dict(hparams.wordDict_file)
        self.vert_dict = self.load_dict(hparams.vertDict_file)
        self.subvert_dict = self.load_dict(hparams.subvertDict_file)
        self.uid2index = self.load_dict(hparams.userDict_file)

        if hasattr(hparams,"use_random_verts"):
            if hparams.use_random_verts:
                self.random_vert_index = np.load(hparams.random_vert_file)
            self.use_random_verts=hparams.use_random_verts
        else:
            self.use_random_verts=False

        if rnd_gen is None:
            self.rnd_gen = np.random.RandomState()
        else:
            self.rnd_gen = rnd_gen

        self.load_body=load_body
        self.dec_static_vec=dec_static_vec
        self.bert_model=bert_model
        self.is_pytorch=is_pytorch
        self.is_test = is_test

    def load_dict(self, file_path):
        """load pickle file
        Args:
            file path (str): file path
        Returns:
            object: pickle loaded object
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def init_news(self, news_file):
        """init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """
        self.nid2index = {}
        news_title = [""]
        news_title_bert = [""]
        news_ab = [""]
        news_vert = [""]
        news_subvert = [""]

        

        tmp_nids=[]
        with tf.io.gfile.GFile(news_file, "r") as rd:
            for line in rd:
                nid, vert, subvert, title, ab, url, _, _ = line.strip("\n").split(
                    self.col_spliter
                )
                
                tmp_nids.append(nid)
                if nid in self.nid2index:
                    continue

                self.nid2index[nid] = len(self.nid2index) + 1
                if self.bert_model is not None:
                    news_title_bert.append(title)
                title = word_tokenize(title)
                ab = word_tokenize(ab)
                news_title.append(title)
                news_ab.append(ab)
                news_vert.append(vert)
                news_subvert.append(subvert)
        
        if self.load_body:
            with np.load(os.path.join(os.path.dirname(news_file),"news_article_emb.npz")) as data:
                newsarticles_vecs=data["arr_0"]

            newsarticles_ids=read_json(os.path.join(os.path.dirname(news_file),"nid2articleindex.json.gz"),compress=True)
            newsarticles_nid2index=dict([(n,i) for i,n in enumerate(newsarticles_ids)])
            self.newsarticles_vecs=np.zeros((len(news_title),newsarticles_vecs.shape[1]),dtype=newsarticles_vecs.dtype)
            self.newsarticles_vecs[1:]=newsarticles_vecs[[newsarticles_nid2index[n] for n in tmp_nids]]
        
        if self.dec_static_vec:
            with np.load(os.path.join(os.path.dirname(news_file),"news_title_unilm2_emb.npz")) as data:
                newsarticles_vecs=data["arr_0"]

            newsarticles_ids=read_json(os.path.join(os.path.dirname(news_file),"nid2articleindex.json.gz"),compress=True)
            newsarticles_nid2index=dict([(n,i) for i,n in enumerate(newsarticles_ids)])
            self.newstitle_decvecs=np.zeros((len(news_title),newsarticles_vecs.shape[1]),dtype=newsarticles_vecs.dtype)
            self.newstitle_decvecs[1:]=newsarticles_vecs[[newsarticles_nid2index[n] for n in tmp_nids]]


        self.news_title_index = np.zeros(
            (len(news_title), self.title_size), dtype="int32"
        )
        self.news_ab_index = np.zeros((len(news_ab), self.body_size), dtype="int32")
        self.news_vert_index = np.zeros(len(news_vert), dtype="int32")
        self.news_subvert_index = np.zeros(len(news_subvert), dtype="int32")

        if self.bert_model is not None:
            # tokenizer = BertTokenizer.from_pretrained(self.bert_model)
            tokenizer = TuringNLRv3Tokenizer.from_pretrained(self.bert_model,do_lower_case=True)
            
            print("Tokenizing")
            # return_tensor = "pt" if self.is_pytorch else "np"
                
            tk = tokenizer(news_title_bert,return_tensors="np",max_length=self.title_size,truncation=True,padding='max_length')
            print("Tokenizing Completed")
            self.news_title_bert_index = np.concatenate([tk["input_ids"],tk["token_type_ids"],tk["attention_mask"]],axis=1)
            # self.news_title_index = tk["input_ids"]#,tk["token_type_ids"],tk["attention_mask"]])
            del tk

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            ab = news_ab[news_index]

            # if self.bert_model is None:
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word_dict:
                    self.news_title_index[news_index, word_index] = self.word_dict[
                        title[word_index].lower()
                    ]
            
            if not self.use_random_verts:
                vert = news_vert[news_index]
                if vert in self.vert_dict:
                    self.news_vert_index[news_index] = self.vert_dict[vert]
            subvert = news_subvert[news_index]
            if subvert in self.subvert_dict:
                self.news_subvert_index[news_index] = self.subvert_dict[subvert]
                

        if self.use_random_verts:
            print("-- Using Random News Verts --")
            self.news_vert_index=self.random_vert_index


    def init_behaviors(self, behaviors_file):
        """init behavior logs given behaviors file.
        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.history_masks = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history_mask = [0] * (self.his_size-len(history)) + [1] * min(self.his_size, len(history))
                history = [0] * (self.his_size - len(history)) + history[: self.his_size]

                if self.is_test:
                    impr_news = [self.nid2index[i] for i in impr.split()]
                    label = [0 for _ in impr.split()]
                else:
                    impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                    label = [int(i.split("-")[1]) for i in impr.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.history_masks.append(history_mask)
                self.imprs.append(impr_news)
                self.labels.append(label)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def parser_one_line(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negtive sampled result.
        Args:
            line (int): sample index.
        Yields:
            list: Parsed results including label, impression id , user id,
            candidate_title_index, clicked_title_index.
        """
        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                candidate_title_index = []
                impr_index = []
                user_index = []
                label = [1] + [0] * self.npratio

                n = newsample(negs, self.npratio, self.rnd_gen)
                candidate_news_index = [p] + n
                candidate_title_index = self.news_title_index[[p] + n]
                candidate_ab_index = self.news_ab_index[[p] + n]
                candidate_vert_index = self.news_vert_index[[p] + n]
                candidate_subvert_index = self.news_subvert_index[[p] + n]
                click_news_index = self.histories[line]
                click_mask = self.history_masks[line]
                click_title_index = self.news_title_index[self.histories[line]]
                click_ab_index = self.news_ab_index[self.histories[line]]
                click_vert_index = self.news_vert_index[self.histories[line]]                
                click_subvert_index = self.news_subvert_index[self.histories[line]]

                if self.bert_model is not None:
                    candidate_title_bert_index = self.news_title_bert_index[[p] + n]
                    click_title_bert_index = self.news_title_bert_index[self.histories[line]]
                else:
                    candidate_title_bert_index = None
                    click_title_bert_index = None
                    

                if self.load_body:
                    candidate_body_vec = self.newsarticles_vecs[[p] + n]
                    click_body_vec = self.newsarticles_vecs[self.histories[line]]
                else:
                    candidate_body_vec=None
                    click_body_vec=None
                
                if self.dec_static_vec:
                    candidate_dec_vec = self.newstitle_decvecs[[p] + n]
                    click_dec_vec = self.newstitle_decvecs[self.histories[line]]
                else:
                    candidate_dec_vec=None
                    click_dec_vec=None
                
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])

                
                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_news_index,
                    candidate_title_index,
                    candidate_ab_index,
                    candidate_vert_index,
                    candidate_subvert_index,
                    click_news_index,
                    click_mask,
                    click_title_index,
                    click_ab_index,                    
                    click_vert_index,                    
                    click_subvert_index,
                    candidate_body_vec,
                    click_body_vec,
                    candidate_dec_vec,
                    click_dec_vec,
                    candidate_title_bert_index,
                    click_title_bert_index
                )

        else:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                candidate_news_index=[]
                candidate_title_index = []
                candidate_ab_index = []
                candidate_vert_index = []
                candidate_subvert_index = []
                candidate_body_vec = []
                candidate_title_bert_index = []
                impr_index = []
                user_index = []
                label = [label]

                candidate_news_index.append(news)
                click_news_index=self.histories[line]
                click_mask = self.history_masks[line]
                candidate_title_index.append(self.news_title_index[news])
                click_title_index = self.news_title_index[self.histories[line]]

                
                candidate_ab_index.append(self.news_ab_index[news])
                candidate_vert_index.append(self.news_vert_index[news])
                candidate_subvert_index.append(self.news_subvert_index[news])
                click_ab_index = self.news_ab_index[self.histories[line]]
                click_vert_index = self.news_vert_index[self.histories[line]]
                click_subvert_index = self.news_subvert_index[self.histories[line]]

                if self.bert_model is not None:
                    candidate_title_bert_index.append(self.news_title_bert_index[news])
                    click_title_bert_index = self.news_title_bert_index[self.histories[line]]
                else:
                    candidate_title_bert_index=None
                    click_title_bert_index=None

                if self.load_body:
                    candidate_body_vec.append(self.newsarticles_vecs[news])
                    click_body_vec = self.newsarticles_vecs[self.histories[line]]
                else:
                    candidate_body_vec=None
                    click_body_vec=None

                if self.dec_static_vec:
                    candidate_dec_vec.append(self.newstitle_decvecs[news])
                    click_dec_vec = self.newstitle_decvecs[self.histories[line]]
                else:
                    candidate_dec_vec=None
                    click_dec_vec=None
                
                impr_index.append(self.impr_indexes[line])
                user_index.append(self.uindexes[line])
               
                yield (
                    label,
                    impr_index,
                    user_index,
                    candidate_news_index,
                    candidate_title_index,
                    candidate_ab_index,
                    candidate_vert_index,
                    candidate_subvert_index,
                    click_news_index,
                    click_mask,
                    click_title_index,
                    click_ab_index,                    
                    click_vert_index,                    
                    click_subvert_index,
                    candidate_body_vec,
                    click_body_vec,
                    candidate_dec_vec,
                    click_dec_vec,
                    candidate_title_bert_index,
                    click_title_bert_index
                )

    def load_data_from_file(self, news_file, behavior_file):
        """Read and parse data from news file and behavior file.
        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.
        Yields:
            object: An iterator that yields parsed results, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        label_list = []
        imp_indexes = []
        user_indexes = []
        candidate_news_indexes = []
        candidate_title_indexes = []
        candidate_ab_indexes = []
        candidate_vert_indexes = []
        candidate_subvert_indexes = []
        click_news_indexes = []
        click_masks = []
        click_title_indexes = []
        click_ab_indexes = []
        click_vert_indexes = []
        click_subvert_indexes = []
        candidate_body_vecs=[]
        click_body_vecs=[]
        candidate_dec_vecs=[]
        click_dec_vecs=[]
        candidate_title_bert_indexes = []
        click_title_bert_indexes = []
                
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            self.rnd_gen.shuffle(indexes)

        for index in indexes:
            for (
                label,
                imp_index,
                user_index,
                candidate_news_index,
                candidate_title_index,
                candidate_ab_index,
                candidate_vert_index,
                candidate_subvert_index,
                click_news_index,
                click_mask,
                click_title_index,
                click_ab_index,
                click_vert_index,                
                click_subvert_index,
                candidate_body_vec,
                click_body_vec,
                candidate_dec_vec,
                click_dec_vec,
                candidate_title_bert_index,
                click_title_bert_index
            ) in self.parser_one_line(index):
                candidate_news_indexes.append(candidate_news_index)
                candidate_title_indexes.append(candidate_title_index)
                candidate_ab_indexes.append(candidate_ab_index)
                candidate_vert_indexes.append(candidate_vert_index)
                candidate_subvert_indexes.append(candidate_subvert_index)
                click_news_indexes.append(click_news_index)
                click_masks.append(click_mask)
                click_title_indexes.append(click_title_index)
                click_ab_indexes.append(click_ab_index)
                click_vert_indexes.append(click_vert_index)                
                click_subvert_indexes.append(click_subvert_index)

                if self.bert_model is not None:
                    candidate_title_bert_indexes.append(candidate_title_bert_index)
                    click_title_bert_indexes.append(click_title_bert_index)


                if self.load_body:
                    candidate_body_vecs.append(candidate_body_vec)
                    click_body_vecs.append(click_body_vec)

                if self.dec_static_vec:
                    candidate_dec_vecs.append(candidate_dec_vec)
                    click_dec_vecs.append(click_dec_vec)
                
                imp_indexes.append(imp_index)
                user_indexes.append(user_index)
                label_list.append(label)

                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(
                        label_list,
                        imp_indexes,
                        user_indexes,
                        candidate_news_indexes,
                        candidate_title_indexes,
                        candidate_ab_indexes,
                        candidate_vert_indexes,
                        candidate_subvert_indexes,
                        click_news_indexes,
                        click_masks,
                        click_title_indexes,
                        click_ab_indexes,
                        click_vert_indexes,                        
                        click_subvert_indexes,
                        candidate_body_vecs,
                        click_body_vecs,
                        candidate_dec_vecs,
                        click_dec_vecs,
                        candidate_title_bert_indexes,
                        click_title_bert_indexes
                    )
                    label_list = []
                    imp_indexes = []
                    user_indexes = []
                    candidate_news_indexes=[]
                    candidate_title_indexes = []
                    candidate_ab_indexes = []
                    candidate_vert_indexes = []
                    candidate_subvert_indexes = []
                    click_news_indexes=[]
                    click_masks=[]
                    click_title_indexes = []
                    click_ab_indexes = []
                    click_vert_indexes = []
                    click_subvert_indexes = []
                    candidate_body_vecs=[]
                    click_body_vecs=[]
                    candidate_dec_vecs=[]
                    click_dec_vecs=[]
                    candidate_title_bert_indexes = []
                    click_title_bert_indexes = []
                    cnt = 0

        if cnt > 0:
            yield self._convert_data(
                label_list,
                imp_indexes,
                user_indexes,
                candidate_news_indexes,
                candidate_title_indexes,
                candidate_ab_indexes,
                candidate_vert_indexes,
                candidate_subvert_indexes,
                click_news_indexes,
                click_masks,
                click_title_indexes,
                click_ab_indexes,
                click_vert_indexes,
                click_subvert_indexes,
                candidate_body_vecs,
                click_body_vecs,
                candidate_dec_vecs,
                click_dec_vecs,
                candidate_title_bert_indexes,
                click_title_bert_indexes
            )

    def _convert_data(
        self,
        label_list,
        imp_indexes,
        user_indexes,
        candidate_news_indexes,
        candidate_title_indexes,
        candidate_ab_indexes,
        candidate_vert_indexes,
        candidate_subvert_indexes,
        click_news_indexes,
        click_masks,
        click_title_indexes,
        click_ab_indexes,                        
        click_vert_indexes,                        
        click_subvert_indexes,
        candidate_body_vecs,
        click_body_vecs,
        candidate_dec_vecs,
        click_dec_vecs,
        candidate_title_bert_indexes,
        click_title_bert_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.
        Args:
            label_list (list): a list of ground-truth labels.
            imp_indexes (list): a list of impression indexes.
            user_indexes (list): a list of user indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            candidate_ab_indexes (list): the candidate news abstarcts' words indices.
            candidate_vert_indexes (list): the candidate news verts' words indices.
            candidate_subvert_indexes (list): the candidate news subverts' indices.
            click_title_indexes (list): words indices for user's clicked news titles.
            click_ab_indexes (list): words indices for user's clicked news abstarcts.
            click_vert_indexes (list): indices for user's clicked news verts.
            click_subvert_indexes (list):indices for user's clicked news subverts.

            Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        
        imp_indexes = np.asarray(imp_indexes, dtype=np.int32)
        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        candidate_news_index_batch = np.asarray(candidate_news_indexes, dtype=np.int64)
        click_news_index_batch = np.asarray(click_news_indexes, dtype=np.int64)

        labels = np.asarray(label_list, dtype=np.float32)
        candidate_title_index_batch = np.asarray(candidate_title_indexes, dtype=np.int64)
        candidate_ab_index_batch = np.asarray(candidate_ab_indexes, dtype=np.int64)
        candidate_vert_index_batch = np.expand_dims(np.asarray(candidate_vert_indexes, dtype=np.int64), axis=-1)
        candidate_subvert_index_batch = np.expand_dims(np.asarray(candidate_subvert_indexes, dtype=np.int64), axis=-1)
        click_mask_batch = np.asarray(click_masks, dtype=np.int64)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_ab_index_batch = np.asarray(click_ab_indexes, dtype=np.int64)
        click_vert_index_batch = np.expand_dims(np.asarray(click_vert_indexes, dtype=np.int64), axis=-1)
        click_subvert_index_batch = np.expand_dims(np.asarray(click_subvert_indexes, dtype=np.int64), axis=-1)
        
        if self.is_pytorch:
            labels = torch.FloatTensor(labels)
            candidate_title_index_batch = torch.LongTensor(candidate_title_index_batch)
            candidate_ab_index_batch = torch.LongTensor(candidate_ab_index_batch)
            candidate_vert_index_batch = torch.LongTensor(candidate_vert_index_batch)
            candidate_subvert_index_batch = torch.LongTensor(candidate_subvert_index_batch)
            click_mask_batch = torch.FloatTensor(click_mask_batch)
            click_title_index_batch = torch.LongTensor(click_title_index_batch)
            click_ab_index_batch = torch.LongTensor(click_ab_index_batch)
            click_vert_index_batch = torch.LongTensor(click_vert_index_batch)
            click_subvert_index_batch = torch.LongTensor(click_subvert_index_batch)
            
        
        res= {
            "impression_index_batch": imp_indexes,
            "user_index_batch": user_indexes,
            "clicked_news_index_batch": click_news_index_batch,
            "clicked_mask_batch": click_mask_batch,
            "clicked_title_batch": click_title_index_batch,
            "clicked_ab_batch": click_ab_index_batch,
            "clicked_vert_batch": click_vert_index_batch,
            "clicked_subvert_batch": click_subvert_index_batch,
            "candidate_news_index_batch": candidate_news_index_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_ab_batch": candidate_ab_index_batch,            
            "candidate_vert_batch": candidate_vert_index_batch,            
            "candidate_subvert_batch": candidate_subvert_index_batch,
            "labels": labels,
            "count":len(imp_indexes)
        }

        if self.bert_model is not None:
            candidate_title_bert_index_batch = np.asarray(candidate_title_bert_indexes, dtype=np.int64)
            click_title_bert_index_batch = np.asarray(click_title_bert_indexes, dtype=np.int64)
            if self.is_pytorch:
                candidate_title_bert_index_batch = torch.LongTensor(candidate_title_bert_index_batch)
                click_title_bert_index_batch = torch.LongTensor(click_title_bert_index_batch)

            res["clicked_title_bert_batch"]=click_title_bert_index_batch
            res["candidate_title_bert_batch"]=candidate_title_bert_index_batch

        if self.load_body:
            candidate_body_vec_batch = np.stack(candidate_body_vecs)
            click_body_vec_batch = np.stack(click_body_vecs)

            res["clicked_bodyvec_batch"]=click_body_vec_batch
            res["candidate_bodyvec_batch"]=candidate_body_vec_batch

        if self.dec_static_vec:
            candidate_dec_vec_batch = np.stack(candidate_dec_vecs)
            click_dec_vec_batch = np.stack(click_dec_vecs)
            if self.is_pytorch:
                candidate_dec_vec_batch = torch.FloatTensor(candidate_dec_vec_batch)
                click_dec_vec_batch = torch.FloatTensor(click_dec_vec_batch)

            res["clicked_decvec_batch"]=click_dec_vec_batch
            res["candidate_decvec_batch"]=candidate_dec_vec_batch

        return res



    def load_user_from_file(self, news_file, behavior_file):
        """Read and parse user data from news file and behavior file.
        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.
        Yields:
            object: An iterator that yields parsed user feature, in the format of dict.
        """

        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)

        user_indexes = []
        impr_indexes = []
        click_news_indexes = []
        click_masks = []
        click_title_indexes = []
        click_ab_indexes = []
        click_vert_indexes = []
        click_subvert_indexes = []
        click_body_vecs=[]
        click_dec_vecs=[]
        click_title_bert_indexes = []
        cnt = 0

        for index in range(len(self.impr_indexes)):
            click_news_indexes.append(self.histories[index])
            click_masks.append(self.history_masks[index])
            click_title_indexes.append(self.news_title_index[self.histories[index]])

            click_ab_indexes.append(self.news_ab_index[self.histories[index]])
            click_vert_indexes.append(self.news_vert_index[self.histories[index]])
            click_subvert_indexes.append(self.news_subvert_index[self.histories[index]])
            user_indexes.append(self.uindexes[index])
            impr_indexes.append(self.impr_indexes[index])

            if self.bert_model is not None:
                click_title_bert_indexes.append(self.news_title_bert_index[self.histories[index]])

            if self.load_body:
                click_body_vecs.append(self.newsarticles_vecs[self.histories[index]])
            if self.dec_static_vec:
                click_dec_vecs.append(self.newstitle_decvecs[self.histories[index]])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_user_data(
                    user_indexes,
                    impr_indexes,
                    click_news_indexes,
                    click_masks,
                    click_title_indexes,
                    click_ab_indexes,
                    click_vert_indexes,
                    click_subvert_indexes,
                    click_body_vecs,
                    click_dec_vecs,
                    click_title_bert_indexes
                )
                user_indexes = []
                impr_indexes = []
                click_news_indexes = []
                click_masks = []
                click_title_indexes = []
                click_ab_indexes = []
                click_vert_indexes = []
                click_subvert_indexes = []
                click_body_vecs=[]
                click_dec_vecs=[]
                click_title_bert_indexes=[]
                cnt = 0

        if cnt > 0:
            yield self._convert_user_data(
                user_indexes,
                impr_indexes,
                click_news_indexes,
                click_masks,
                click_title_indexes,
                click_ab_indexes,
                click_vert_indexes,
                click_subvert_indexes,
                click_body_vecs,
                click_dec_vecs,
                click_title_bert_indexes
            )

    def _convert_user_data(
        self,
        user_indexes,
        impr_indexes,
        click_news_indexes,
        click_masks,
        click_title_indexes,
        click_ab_indexes,
        click_vert_indexes,
        click_subvert_indexes,
        click_body_vecs,
        click_dec_vecs,
        click_title_bert_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.
        Args:
            user_indexes (list): a list of user indexes.
            click_title_indexes (list): words indices for user's clicked news titles.
            click_ab_indexes (list): words indices for user's clicked news abs.
            click_vert_indexes (list): words indices for user's clicked news verts.
            click_subvert_indexes (list): words indices for user's clicked news subverts.

        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        user_indexes = np.asarray(user_indexes, dtype=np.int32)
        impr_indexes = np.asarray(impr_indexes, dtype=np.int32)
        click_news_indexes = np.asarray(click_news_indexes, dtype=np.int32)

        click_mask_batch = np.asarray(click_masks, dtype=np.int64)
        click_title_index_batch = np.asarray(click_title_indexes, dtype=np.int64)
        click_ab_index_batch = np.asarray(click_ab_indexes, dtype=np.int64)
        click_vert_index_batch = np.expand_dims(np.asarray(click_vert_indexes, dtype=np.int64), axis=-1)
        click_subvert_index_batch = np.expand_dims(np.asarray(click_subvert_indexes, dtype=np.int64), axis=-1)

        if self.is_pytorch:
            click_mask_batch = torch.FloatTensor(click_mask_batch)
            click_title_index_batch = torch.LongTensor(click_title_index_batch)
            click_ab_index_batch = torch.LongTensor(click_ab_index_batch)
            click_vert_index_batch = torch.LongTensor(click_vert_index_batch)
            click_subvert_index_batch = torch.LongTensor(click_subvert_index_batch)


        res={
            "user_index_batch": user_indexes,
            "impr_index_batch": impr_indexes,
            "clicked_news_index_batch": click_news_indexes,
            "clicked_mask_batch": click_mask_batch,
            "clicked_title_batch": click_title_index_batch,
            "clicked_ab_batch": click_ab_index_batch,
            "clicked_vert_batch": click_vert_index_batch,
            "clicked_subvert_batch": click_subvert_index_batch,
            "count":len(user_indexes)
        }

        if self.bert_model is not None:
            click_title_bert_index_batch = np.asarray(click_title_bert_indexes, dtype=np.int64)
            if self.is_pytorch:
                click_title_bert_index_batch = torch.LongTensor(click_title_bert_index_batch)
            res["clicked_title_bert_batch"]=click_title_bert_index_batch


        if self.load_body:
            click_body_vec_batch = np.stack(click_body_vecs)

            res["clicked_bodyvec_batch"]=click_body_vec_batch

        if self.dec_static_vec:
            click_dec_vec_batch = np.stack(click_dec_vecs)
            if self.is_pytorch:
                click_dec_vec_batch = torch.FloatTensor(click_dec_vec_batch)

            res["clicked_decvec_batch"]=click_dec_vec_batch

        return res

    def load_news_from_file(self, news_file,start_idx=0):
        """Read and parse user data from news file.
        Args:
            news_file (str): A file contains several informations of news.
        Yields:
            object: An iterator that yields parsed news feature, in the format of dict.
        """
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        candidate_ab_indexes = []
        candidate_vert_indexes = []
        candidate_subvert_indexes = []
        candidate_body_vecs=[]
        candidate_dec_vecs=[]
        candidate_title_bert_indexes = []
        cnt = 0

        for index in range(start_idx,len(self.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(self.news_title_index[index])
            candidate_ab_indexes.append(self.news_ab_index[index])
            candidate_vert_indexes.append(self.news_vert_index[index])
            candidate_subvert_indexes.append(self.news_subvert_index[index])

            if self.bert_model is not None:
                candidate_title_bert_indexes.append(self.news_title_bert_index[index])

            if self.load_body:
                candidate_body_vecs.append(self.newsarticles_vecs[index])
            if self.dec_static_vec:
                candidate_dec_vecs.append(self.newstitle_decvecs[index])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_news_data(
                    news_indexes,
                    candidate_title_indexes,
                    candidate_ab_indexes,
                    candidate_vert_indexes,
                    candidate_subvert_indexes,
                    candidate_body_vecs,
                    candidate_dec_vecs,
                    candidate_title_bert_indexes
                )
                news_indexes = []
                candidate_title_indexes = []
                candidate_ab_indexes = []
                candidate_vert_indexes = []
                candidate_subvert_indexes = []
                candidate_body_vecs = []
                candidate_dec_vecs = []
                candidate_title_bert_indexes = []
                cnt = 0

        if cnt > 0:
            yield self._convert_news_data(
                news_indexes,
                candidate_title_indexes,
                candidate_ab_indexes,
                candidate_vert_indexes,
                candidate_subvert_indexes,
                candidate_body_vecs,
                candidate_dec_vecs,
                candidate_title_bert_indexes
            )

    def _convert_news_data(
        self,
        news_indexes,
        candidate_title_indexes,
        candidate_ab_indexes,
        candidate_vert_indexes,
        candidate_subvert_indexes,
        candidate_body_vecs,
        candidate_dec_vecs,
        candidate_title_bert_indexes
    ):
        """Convert data into numpy arrays that are good for further model operation.
        Args:
            news_indexes (list): a list of news indexes.
            candidate_title_indexes (list): the candidate news titles' words indices.
            candidate_ab_indexes (list): the candidate news abstarcts' words indices.
            candidate_vert_indexes (list): the candidate news verts' words indices.
            candidate_subvert_indexes (list): the candidate news subverts' words indices.
        Returns:
            dict: A dictionary, containing multiple numpy arrays that are convenient for further operation.
        """

        news_indexes_batch = np.asarray(news_indexes, dtype=np.int32)
        candidate_title_index_batch = np.asarray(candidate_title_indexes, dtype=np.int32)
        candidate_ab_index_batch = np.asarray(candidate_ab_indexes, dtype=np.int32)
        candidate_vert_index_batch = np.expand_dims(np.asarray(candidate_vert_indexes, dtype=np.int32), axis=-1)
        candidate_subvert_index_batch = np.expand_dims(np.asarray(candidate_subvert_indexes, dtype=np.int32), axis=-1)

        if self.is_pytorch:
            candidate_title_index_batch = torch.LongTensor(candidate_title_index_batch)
            candidate_ab_index_batch = torch.LongTensor(candidate_ab_index_batch)
            candidate_vert_index_batch = torch.LongTensor(candidate_vert_index_batch)
            candidate_subvert_index_batch = torch.LongTensor(candidate_subvert_index_batch)


        res= {
            "news_index_batch": news_indexes_batch,
            "candidate_title_batch": candidate_title_index_batch,
            "candidate_ab_batch": candidate_ab_index_batch,
            "candidate_vert_batch": candidate_vert_index_batch,
            "candidate_subvert_batch": candidate_subvert_index_batch,
            "count":len(news_indexes_batch)
        }

        if self.bert_model is not None:
            candidate_title_bert_index_batch = np.asarray(candidate_title_bert_indexes, dtype=np.int32)
            if self.is_pytorch:
                candidate_title_bert_index_batch = torch.LongTensor(candidate_title_bert_index_batch)
            res["candidate_title_bert_batch"]=candidate_title_bert_index_batch


        if self.load_body:
            candidate_body_vec_batch = np.stack(candidate_body_vecs)
            res["candidate_bodyvec_batch"]=candidate_body_vec_batch
        
        if self.dec_static_vec:
            candidate_dec_vec_batch = np.stack(candidate_dec_vecs)
            if self.is_pytorch:
                candidate_dec_vec_batch = torch.FloatTensor(candidate_dec_vec_batch)
            res["candidate_decvec_batch"]=candidate_dec_vec_batch
        
        return res


    def load_impression_from_file(self, behaivors_file):
        """Read and parse impression data from behaivors file.
        Args:
            behaivors_file (str): A file contains several informations of behaviros.
        Yields:
            object: An iterator that yields parsed impression data, in the format of dict.
        """

        if not hasattr(self, "histories"):
            self.init_behaviors(behaivors_file)

        indexes = np.arange(len(self.labels))

        for index in indexes:
            impr_label = np.array(self.labels[index], dtype="int32")
            impr_news = np.array(self.imprs[index], dtype="int32")

            yield (
                self.impr_indexes[index],
                impr_news,
                self.uindexes[index],
                impr_label,
            )