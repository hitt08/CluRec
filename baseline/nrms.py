import argparse
import sys

from recsys_utils import model_eval
from iterator import MINDCatIterator as catiterator
from init_params import nrms_init_params

from baseline.nrms_base import NRMSModel


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store', dest='mind_type', default="demo", type=str, help='MIND type: demo, small, large')
    parser.add_argument('-e', action='store', dest='epochs', default=10, type=int, help='epochs')
    parser.add_argument('-b', action='store', dest='batch_size', default=32, type=int, help='batch size')
    args = parser.parse_args()
    print(args)


    seed = 42
    hparams,file_paths=nrms_init_params(args,script_file=__file__,seed=seed)

    train_news_file     = file_paths["train_news_file"]
    train_behaviors_file= file_paths["train_behaviors_file"] 
    valid_news_file     = file_paths["valid_news_file"]
    valid_behaviors_file= file_paths["valid_behaviors_file"] 
    test_news_file      = file_paths["test_news_file"]
    test_behaviors_file = file_paths["test_behaviors_file"] 

    iterator = catiterator

    model = NRMSModel(hparams, iterator, seed=seed)

       
    model_eval(model, valid_news_file, valid_behaviors_file, test_news_file, test_behaviors_file, epoch=-1)
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file,test_news_file,test_behaviors_file)