#!/usr/bin/env python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast 
from torch.cuda.amp import GradScaler
from pytorch_metric_learning import samplers
import logging
import time
import os
import random
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import sys
sys.path.append("../") 
sys.path.append("el_test/el_test")
sys.path.append("../inference")
from transformers import AutoModel
import wandb
from inference import SnomedGraph
from src.data_loader import (
    DictionaryDataset,
    QueryDataset,
    QueryDataset_pretraining,
    MetricLearningDataset,
    MetricLearningDataset_pairwise,
)
from src.model_wrapper import (
    Model_Wrapper
)
from src.metric_learning import (
    Sap_Metric_Learning,
)
from el_test import get_el_datasets, el_evaluate
LOGGER = logging.getLogger()

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):
    """
    load dictionary
    
    Parameters
    ----------
    dictionary_path : str
        a path of dictionary
    """
    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def load_queries_pretraining(data_dir, filter_duplicate):
    """
    load query data
    
    Parameters
    ----------
    is_train : bool
        train or dev
    filter_composite : bool
        filter composite mentions
    filter_duplicate : bool
        filter duplicate queries
    """
    dataset = QueryDataset_pretraining(
        data_dir=data_dir,
        filter_duplicate=filter_duplicate
    )
    
    return dataset.data

def train(cfg, data_loader, model, scaler=None, model_wrapper=None, step_global=0):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        batch_x1, batch_x2, batch_y = data
        batch_x_cuda1, batch_x_cuda2 = {},{}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()

        batch_y_cuda = batch_y.cuda()
    
        if cfg.amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2, batch_y_cuda)  
        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        if cfg.use_wandb:
            wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1
        #if (i+1) % 10 == 0:
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1,train_loss / (train_steps+1e-9)))
        #LOGGER.info ("epoch: {} loss: {:.3f}".format(i+1, loss.item()))

        # save model every K iterations
        if step_global % cfg.checkpoint_step == 0:
            checkpoint_dir = os.path.join(cfg.output_dir, "checkpoint_iter_{}".format(str(step_global)))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global

@hydra.main(version_base='1.3', config_path="config", config_name="config")    
def main(cfg : DictConfig):
    if cfg.use_wandb:
        run = wandb.init(project=cfg.wandb_project)
    init_logging()
    #init_seed(args.seed)
    print(cfg)

    torch.manual_seed(cfg.random_seed)
    
    # prepare for output
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
        
    # load BERT tokenizer, dense_encoder
    model_wrapper = Model_Wrapper()
    encoder, tokenizer = model_wrapper.load_bert(
        path=cfg.model_dir,
        max_length=cfg.max_length,
        use_cuda=cfg.use_cuda,
        trust_remote_code=cfg.trust_remote_code,
        #lowercase=not cfg.cased
    )
    # load SAP model
    model = Sap_Metric_Learning(
            encoder = encoder,
            learning_rate=cfg.learning_rate, 
            weight_decay=cfg.weight_decay,
            use_cuda=cfg.use_cuda,
            pairwise=cfg.pairwise,
            loss=cfg.loss,
            use_miner=cfg.use_miner,
            miner_margin=cfg.miner_margin,
            type_of_triplets=cfg.type_of_triplets,
            agg_mode=cfg.agg_mode,
    )
    if cfg.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")
    
    def collate_fn_batch_encoding(batch):
        query1, query2, query_id = zip(*batch)
        query_encodings1 = tokenizer.batch_encode_plus(
                list(query1), 
                max_length=cfg.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        query_encodings2 = tokenizer.batch_encode_plus(
                list(query2), 
                max_length=cfg.max_length, 
                padding="max_length", 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
        #query_encodings_cuda = {}
        #for k,v in query_encodings.items():
        #    query_encodings_cuda[k] = v.cuda()
        query_ids = torch.tensor(list(query_id))
        return  query_encodings1, query_encodings2, query_ids

    if cfg.pairwise:
        train_set = MetricLearningDataset_pairwise(
                path=cfg.train_dir,
                tokenizer = tokenizer
        )
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=16,
            collate_fn=collate_fn_batch_encoding
        )
    else:
        train_set = MetricLearningDataset(
            path=cfg.train_dir,
            tokenizer = tokenizer
        )
        # using a sampler
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=cfg.train_batch_size,
            #shuffle=True,
            sampler=samplers.MPerClassSampler(train_set.query_ids,\
                2, length_before_new_iter=100000),
            num_workers=16, 
            )
    # mixed precision training 
    if cfg.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1,cfg.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,cfg.epoch))

        # train
        train_loss, step_global = train(cfg, data_loader=train_loader, model=model, scaler=scaler, model_wrapper=model_wrapper, step_global=step_global)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if cfg.save_checkpoint_all:
            checkpoint_dir = os.path.join(cfg.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            model_wrapper.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == cfg.epoch:
            model_wrapper.save_model(cfg.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    main()
