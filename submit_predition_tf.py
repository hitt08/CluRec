import argparse
import os
from tqdm import tqdm
import zipfile
import numpy as np
import sys

from iterator import MINDCatIterator as catiterator
from init_params import nrms_init_params


from baseline.nrms import NRMSModel
from clurec.nrms_group_wa import NRMSGroupModelDECCLImprsVertWAKLGL
from baseline.lstur import LSTURModel
from clurec.lstur_group_wa import LSTURGroupModelDECCLImprsVertWAKLGL

mind_type="large"
model_type="lstur"

def write_prediction_file(url,impr_indexes, preds):
    tmp_url= os.path.join("/tmp/prediction.txt")
    with open(tmp_url, 'w') as f:
        for impr_index, preds in tqdm(zip(impr_indexes, preds),total=len(preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank])+ '\n')
            
    ext=os.path.splitext(url)[-1]
    f = zipfile.ZipFile(url.replace(ext,".zip"), 'w', zipfile.ZIP_DEFLATED)
    f.write(tmp_url, arcname="prediction.txt")
    f.close()

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='mind_type', default=mind_type, type=str, help='MIND type: demo, small, large')
    parser.add_argument('-e', action='store', dest='epochs', default=3, type=int, help='epochs')
    parser.add_argument('-b', action='store', dest='batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-c', action='store', dest='n_clusters', type=int, default=0, help='number of clusters')
    parser.add_argument('-u', action='store', dest='update_interval', type=int, default=None, help='DEC update Interval')
    parser.add_argument('-t', action='store', dest='model_type', type=str, default=model_type, help='nrms, lstur')
    parser.add_argument('--upct', action='store', dest='update_interval_pct', type=float, default=None, help='Percentage of input for DEC update Interval')
    parser.add_argument('--dec_batch', action='store', dest='dec_batch', type=int,default=256, help='DEC batch size')
    parser.add_argument('--dec_lr', action='store', dest='dec_lr', type=float, default=0.0001, help='DEC learning rate')
    parser.add_argument('--dec_ae_iters', dest='dec_ae_iters', action='store', type=int, default=10000, help='DEC AE pretrain iters')
    parser.add_argument('--dec_ae_fineiters', dest='dec_ae_fineiters', action='store', type=int, default=20000, help='DEC AE finetrain iters')
    parser.add_argument('--dec_pretrain', dest='dec_pretrain', action='store_true', help='pretrain DEC')
    parser.add_argument('--dec_untrained', dest='dec_raw', action='store_true', help='use untrained DEC')
    parser.add_argument('--random_verts', dest='random_verts', action='store_true', help='Use random groups')
    parser.set_defaults(random_verts=False)
    parser.set_defaults(dec_pretrain=False)
    parser.set_defaults(dec_raw=False)
    args=parser.parse_args()

    print(args)



    model_type=args.model_type

    model_preds={}
    for script_file in [f"baseline/{model_type}.py"]:#,f"dec_cl_imprsvert_wa_kl_global/{model_type}_group_wa.py"]:   
    # for script_file in [f"dec_cl_imprsvert_wa_kl_global/{model_type}_group_wa.py"]:   
        print("\n\n"+"="*20)
        print(script_file)
        print("="*20,"\n")
        seed = 42
        init_dec= "dec" in script_file
        is_subvert= "subvert" in script_file
        args.random_verts = False
        args.update_interval_pct = 10
        args.n_clusters = 295 if init_dec else 0
        hparams,file_paths=nrms_init_params(args,script_file=script_file,seed=seed,init_dec=init_dec,subverts=is_subvert,model_type=model_type)
        
        train_news_file     = file_paths["train_news_file"]
        train_behaviors_file= file_paths["train_behaviors_file"] 
        valid_news_file     = file_paths["valid_news_file"]
        valid_behaviors_file= file_paths["valid_behaviors_file"] 
        test_news_file= valid_news_file.replace("valid","test")
        test_behaviors_file= valid_behaviors_file.replace("valid","test")
        
        
        hparams.model_weights_path=hparams.model_weights_path.replace(".h5",f"_{args.epochs}.h5")
        
        if init_dec:
            model_class = LSTURGroupModelDECCLImprsVertWAKLGL if args.model_type=="lstur" else NRMSGroupModelDECCLImprsVertWAKLGL
        else:
            model_class = LSTURModel  if args.model_type=="lstur" else NRMSModel
            
        model = model_class(hparams, catiterator, seed=seed)
        if init_dec:
            model.initialize_dec(train_news_file)
            model.support_quick_scoring = False
        
        model.model.load_weights(hparams.model_weights_path)
        
        model.test_iterator.is_test = True
        model.test_iterator.batch_size = 512
        if model.support_quick_scoring:
            print("Fast Eval")
            if script_file.startswith("baseline"):
                tmp_eval = model.run_fast_vert_svert_eval(
                    test_news_file, test_behaviors_file, model.test_iterator,return_verts=False,silent=False
                )
            else:
                tmp_eval = model.run_fast_eval(
                        test_news_file, test_behaviors_file, model.test_iterator,return_verts=False,silent=False
                    )
            if init_dec:
                group_impr_indexes, group_labels, group_preds, group_gpreds, clusters = tmp_eval
            else:
                if script_file.startswith("baseline"):
                    group_impr_indexes, group_labels, group_preds, group_gpreds, group_sgpreds  = tmp_eval
                else:
                    group_impr_indexes, group_labels, group_preds, group_gpreds = tmp_eval
                
        else:
            print("Slow Eval")
            tmp_eval = model.run_slow_eval(
                test_news_file, test_behaviors_file, model.test_iterator,return_verts=False
            )
            
            if init_dec:
                group_impr_indexes, group_labels, group_preds, group_gpreds, clusters = tmp_eval
            else:
                group_impr_indexes, group_labels, group_preds, group_gpreds = tmp_eval   
                
        write_prediction_file(hparams.model_weights_path.replace(".h5","_test_preds.txt"),group_impr_indexes, group_preds)
        write_prediction_file(hparams.model_weights_path.replace(".h5","_test_gpreds.txt"),group_impr_indexes, group_gpreds)
        if script_file.startswith("baseline"):
            write_prediction_file(hparams.model_weights_path.replace(".h5","_test_sgpreds.txt"),group_impr_indexes, group_sgpreds)
