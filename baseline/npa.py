import argparse
import sys

from recsys_utils import model_eval
from iterator import MINDCatIterator as catiterator
from init_params import nrms_init_params
from submit_predition_tf import write_prediction_file
from baseline.npa_base import NPAModel


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='mind_type', default="demo", type=str, help='MIND type: demo, small, large')
    parser.add_argument('-e', action='store', dest='epochs', default=10, type=int, help='epochs')
    # parser.add_argument('-b', action='store', dest='batch_size', default=32, type=int, help='batch size')
    args = parser.parse_args()
    print(args)


    seed = 42
    hparams,file_paths=nrms_init_params(args,script_file=__file__,seed=seed,model_type="npa")

    train_news_file     = file_paths["train_news_file"]
    train_behaviors_file= file_paths["train_behaviors_file"] 
    valid_news_file     = file_paths["valid_news_file"]
    valid_behaviors_file= file_paths["valid_behaviors_file"] 
    test_news_file      = file_paths["test_news_file"]
    test_behaviors_file = file_paths["test_behaviors_file"] 

    iterator = catiterator
    hparams.title_size= 30
    model = NPAModel(hparams, iterator, seed=seed)
    model.val_iterator.batch_size = 512
    model.test_iterator.batch_size = 512

       
    # model_eval(model, valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file, epoch=-1)
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file,test_news_file,test_behaviors_file,eval_every_epoch=True)

    if args.mind_type=="large":
        test_news_file= valid_news_file.replace("valid","test")
        test_behaviors_file= valid_behaviors_file.replace("valid","test")
        model.test_iterator.is_test = True
        tmp_eval = model.run_slow_eval(
                test_news_file, test_behaviors_file, model.test_iterator,return_verts=False
            )
        group_impr_indexes, group_labels, group_preds, group_gpreds, group_sgpreds = tmp_eval
        
                
        write_prediction_file(hparams.model_weights_path.replace(".h5","_test_preds.txt"),group_impr_indexes, group_preds)
        write_prediction_file(hparams.model_weights_path.replace(".h5","_test_gpreds.txt"),group_impr_indexes, group_gpreds)
        write_prediction_file(hparams.model_weights_path.replace(".h5","_test_sgpreds.txt"),group_impr_indexes, group_sgpreds)
