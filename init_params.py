import os
import sys
# from unittest.mock import _ArgsKwargs
from recommenders.models.newsrec.newsrec_utils import prepare_hparams
# from recommenders.models.newsrec.newsrec_utils import get_mind_data_set


from recsys_utils import get_subvert_labels, get_vert_labels
from iterator import MINDCatIterator as catiterator
from python_utils.data import count_file_lines

def nrms_init_params(args,script_file,seed=42,model_type="nrms",init_dec=False,dec_se=False,subverts=False,is_pytorch=False):
    epochs = args.epochs
    seed = seed
    # batch_size = args.batch_size

    # Options: demo, small, large
    MIND_type = args.mind_type

    data_path = f"/nfs/mind/{MIND_type}"

    file_paths={}

    file_paths["train_news_file"] = os.path.join(data_path, 'train', r'news.tsv')
    file_paths["train_behaviors_file"] = os.path.join(data_path, 'train', r'behaviors.tsv')
    file_paths["valid_news_file"] = os.path.join(data_path, 'valid', r'news.tsv')
    file_paths["valid_behaviors_file"] = os.path.join(data_path, 'valid', r'behaviors.tsv')

    # if MIND_type=="large":
    #     file_paths["test_news_file"] = os.path.join(data_path, 'test', r'news.tsv')
    #     file_paths["test_behaviors_file"] = os.path.join(data_path, 'test', r'behaviors.tsv')
    # else:
    file_paths["test_news_file"] = None
    file_paths["test_behaviors_file"] = None



    wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
    userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(data_path, "utils", f'{model_type}.yaml')

    # mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

    hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl"),
                          subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl"),
                        #   batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)

    # hparams.vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
    # hparams.subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
    hparams.vertEmb_file = os.path.join(data_path, "utils", "vert_embedding.npy")
    if hasattr(args,"random_verts"):
        hparams.use_random_verts=args.random_verts
        if args.random_verts:
            hparams.vertEmb_file=os.path.join(data_path,"utils","random_vert_embedding.npy")
            hparams.random_vert_file=os.path.join(data_path,"utils","random_vert_index.npy")    
    hparams.subvertEmb_file = os.path.join(data_path, "utils", "subvert_embedding.npy")
    hparams.dense_activation="relu"
    hparams.group_eval = True  # for group-based evaluation

    hparams.eval_clusters = args.n_clusters if hasattr(args,"n_clusters") else 0
    if init_dec:
        hparams.eval_clusters+=1
    if hparams.eval_clusters <= 1:
        tmp_iterator = catiterator(hparams)
        vertDict=tmp_iterator.load_dict(hparams.vertDict_file)
        hparams.vert_num = max(vertDict.values())+1      #+1 for 0th index
        hparams.vert_emb_dim = 100
        if subverts:
            subvertDict=tmp_iterator.load_dict(hparams.subvertDict_file)
            hparams.subvert_num = max(subvertDict.values())+1    #+1 for 0th index
            hparams.subvert_emb_dim = 100

            # train_news_verts = get_subvert_labels(tmp_iterator, file_paths["train_news_file"])
            hparams.eval_clusters = hparams.subvert_num
        else:
            # train_news_verts = get_vert_labels(tmp_iterator, file_paths["train_news_file"])
            hparams.eval_clusters = hparams.vert_num
        del tmp_iterator

    script=os.path.basename(script_file).replace(".py","")
    if hasattr(args,"random_verts") and args.random_verts:
        script+="_rnd"
    script_dir=os.path.dirname(script_file)
    if script_dir=="":
        script_dir=script

    nrm_pttrn=f"{script}_e{epochs}_b{hparams.batch_size}_c{hparams.eval_clusters}"

    if init_dec:
        hparams.n_clusters = args.n_clusters
        if hparams.n_clusters <= 1:
            # hparams.n_clusters = len(set(list(train_news_verts.values()))) #distinct clusters
            hparams.n_clusters = len(vertDict.values()) #distinct clusters
        print(f"Number of Clusters: {hparams.n_clusters}")
        hparams.dec_batch_size = args.dec_batch
        hparams.dec_learning_rate = args.dec_lr
        hparams.dec_decay_step = 20  # DEC model learning rate decay
        hparams.ae_layerwise_pretrain_iters = args.dec_ae_iters  # layer-wise pretrain weight for greedy layer wise auto encoder
        hparams.ae_finetune_iters = args.dec_ae_fineiters  # AE fine-tunning iteration
        if dec_se:
            hparams.body_emb_dim=384  #TODO: change to args
            
        dec_pttrn = f"c{hparams.n_clusters}_b{args.dec_batch}_l{args.dec_lr}"
        if dec_se:
            dec_pttrn = "se_"+dec_pttrn


        if is_pytorch:
            hparams.ae_pretrained_weights_path = os.path.join(data_path, "pytorch","dec_checkpoint", f"ae_{dec_pttrn}.pt")
            hparams.dec_init_weights_path = os.path.join(data_path, "pytorch","dec_checkpoint", f"dec_init_{dec_pttrn}.pt")
        else:
            hparams.ae_pretrained_weights_path = os.path.join(data_path, "dec_checkpoint", f"ae_{dec_pttrn}.h5")
            hparams.dec_init_weights_path = os.path.join(data_path, "dec_checkpoint", f"dec_init_{dec_pttrn}.h5")

        if args.update_interval_pct is not None:
            args.update_interval=round(args.update_interval_pct*count_file_lines(file_paths["train_news_file"])/100)
            print(f"Update Interval: {args.update_interval}")
            

        dec_pttrn+=f"_u{args.update_interval}"
        if is_pytorch:
            hparams.dec_pretrained_weights_path = os.path.join(data_path, "pytorch","dec_checkpoint", f"dec_{dec_pttrn}.pt")
        else:
            hparams.dec_pretrained_weights_path = os.path.join(data_path, "dec_checkpoint", f"dec_{dec_pttrn}.h5")
        if args.dec_raw:
            # hparams.ae_pretrained_weights_path=None
            hparams.dec_pretrained_weights_path=None
            dec_pttrn="raw"+dec_pttrn
        else:
            # dec_pttrn = f"c{hparams.n_clusters}_b{args.dec_batch}_l{args.dec_lr}"
            # if dec_se:
            #     dec_pttrn = "se_"+dec_pttrn
            # hparams.dec_pretrained_weights_path = os.path.join(data_path, "dec_checkpoint", f"dec_{dec_pttrn}.h5")
            # hparams.ae_pretrained_weights_path = os.path.join(os.path.dirname(hparams.dec_pretrained_weights_path), f"ae_{dec_pttrn}.h5")
            hparams.dec_eval_path = os.path.join(os.path.dirname(hparams.dec_pretrained_weights_path),f"eval_{dec_pttrn}.txt")
            os.makedirs(os.path.dirname(hparams.dec_pretrained_weights_path), exist_ok=True)
            
        if hasattr(args,"recsys_loss_weight"):
            hparams.recsys_loss_weight=args.recsys_loss_weight
            dec_pttrn+=f"_rwl{args.recsys_loss_weight}"
        if hasattr(args,"dec_news_loss_weight"):
            hparams.dec_news_loss_weight=args.dec_news_loss_weight
            dec_pttrn+=f"_nwl{args.dec_news_loss_weight}"
        if hasattr(args,"dec_user_loss_weight"):
            hparams.dec_user_loss_weight=args.dec_user_loss_weight
            dec_pttrn+=f"_uwl{args.dec_user_loss_weight}"

        nrm_pttrn+=f"_dec_{dec_pttrn}"

    
    if is_pytorch:
        hparams.model_weights_path = os.path.join(data_path, f"{script_dir}_checkpoint", f"{nrm_pttrn}.pt")
    else:
        hparams.model_weights_path = os.path.join(data_path, f"{script_dir}_checkpoint", f"{nrm_pttrn}.h5")

    hparams.eval_path = os.path.join(os.path.dirname(hparams.model_weights_path), f"eval_{nrm_pttrn}.jsonl")
    hparams.group_eval_path = os.path.join(os.path.dirname(hparams.model_weights_path), f"eval_grp_{nrm_pttrn}.jsonl")
    hparams.last_epoch_path = os.path.join(os.path.dirname(hparams.model_weights_path), f"saved_epoch_{nrm_pttrn}.txt")
    os.makedirs(os.path.dirname(hparams.model_weights_path), exist_ok=True)

    if init_dec:
        if is_pytorch:
            hparams.dec_weights_path = os.path.join(os.path.dirname(hparams.model_weights_path), f"dec_{nrm_pttrn}.pt")
        else:
            hparams.dec_weights_path = os.path.join(os.path.dirname(hparams.model_weights_path), f"dec_{nrm_pttrn}.h5")

    return hparams,file_paths