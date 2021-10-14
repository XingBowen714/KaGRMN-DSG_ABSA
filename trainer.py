import logging
import os
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import  my_collate_bert, my_collate_bert_eval
from torch.optim import Adam
from transformers import AdamW
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def reset_params(model,args):
    #for n,p in model.named_parameters():
     #   if 'bert' not in n:
    for gcn in model.gcn:  
    #       print(n)
        for p in gcn.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                    
def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_rels': batch[7], # adj no-reshape
                    'dep_heads': batch[8],
                    'aspect_position': batch[9],
                    'dep_dirs': batch[10]
                    }
        labels = batch[6]
    else: # bert
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'input_ids': batch[0],
                    'input_aspect_ids': batch[3],
                    'word_indexer': batch[1],
                    'w_idx': batch[2],
                    'aspect_indexer': batch[4],
                    'input_cat_ids': batch[5],
                    'segment_ids': batch[6],
                    'input_des_ids':batch[7],
                    'des_indexer':batch[8],
                    'dep_tags': batch[9],
                    'pos_class': batch[10],
                    'text_len': batch[11],
                    'aspect_len': batch[12],
                    'des_len': batch[13],
                    'dep_rels': batch[15],
                    'dep_heads': batch[16],
                    'aspect_position': batch[17],
                    'dep_dirs': batch[18],
                    'aspect_start':batch[19],
                    'sparse_graph': batch[20]}
        labels = batch[14]
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    if not args.one_bert: 
        des_bert_params_id = list(map(id,model.des_bert.parameters()))
    con_bert_params_id = list(map(id,model.bert.parameters()))
    
    if args.one_bert:
        base_params = filter(lambda p: id(p) not in con_bert_params,
                     model.parameters())
    else:
        base_params = filter(lambda p: id(p) not in des_bert_params + con_bert_params,
                     model.parameters())
    bert_decay_params, bert_no_decay_params, base_decay_params, base_no_decay_params = [], [], [], []
    
    for n, p in model.named_parameters():
       # print(n,p)
        if 'bert' in n:
            if not any(nd in n for nd in no_decay):
                bert_decay_params.append(p)
                #bert_decay_params_n.append(n)
            else:
                bert_no_decay_params.append(p)
                #bert_no_decay_params_n.append(n)
        else:
            if not any(nd in n for nd in no_decay):
                base_decay_params.append(p)
                #base_decay_params_n.append(n)
            else:
                base_no_decay_params.append(p)

    optimizer = AdamW([
        {'params': bert_decay_params, 'lr' : args.bert_lr,  'weight_decay': args.weight_decay},
        {'params': bert_no_decay_params, 'lr':args.bert_lr, 'weight_decay': 0.0},
        {'params': base_decay_params,'weight_decay': args.weight_decay},
        {'params': base_no_decay_params, 'weight_decay': 0.0}],
                      lr=args.learning_rate, eps=args.adam_epsilon)

    return optimizer


def train(args, train_dataset, model, test_dataset):
    '''Train the model'''
    #tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
        reset_params(model,args)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20], gamma = 0.5)
    epoch = -1
    for _ in train_iterator:
        epoch = epoch + 1
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels = get_input_from_batch(args, batch)
            #print(inputs)
            #assert 1==0
            logit = model(**inputs)
            loss = F.cross_entropy(logit, labels)
            #print(loss)
            #assert 1==0

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #train_results, train_loss = evaluate(args, train_dataset, model, 'Train', epoch)
                    results, eval_loss = evaluate(args, test_dataset, model, 'Eval', epoch)
                    all_eval_results.append(results)
                   
                    logging_loss = tr_loss



            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

        scheduler.step()
        print('*******')
        print('Updated learning rate: {0}'.format(scheduler.get_lr()))
    #tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results


def evaluate(args, eval_dataset, model, mode, epoch):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    #collate_fn = get_collate_fn(args)
    collate_fn = my_collate_bert_eval
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    case_study = False
     
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            pred_distri = F.softmax(logits, dim = 1).detach().cpu().numpy()
            #print(pred_distri)
            #assert 0==9
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
            pred_distri =np.append(pred_distri, F.softmax(logits, dim = 1).detach().cpu().numpy(), axis = 0)
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids, epoch)
    results.update(result)
    case_id = []
    case_distri = []
    case_id_bad = []
    case_distri_bad = []
    if case_study and result['acc'] > 0.855:
        for i in range(len(preds)):
            if preds[i] == out_label_ids[i]:
                case_id.append(i)
                case_distri.append(pred_distri[i])
            if preds[i] != out_label_ids[i]:
                case_id_bad.append(i)
                case_distri_bad.append(pred_distri[i])
        with open(args.dataset_name + '_id.goodcase', 'w') as f:
            for i in case_id:
                f.write(str(i))
                f.write('\n')
        
        with open(args.dataset_name + '_id.badcase', 'w') as f:
            for i in case_id_bad:
                f.write(str(i))
                f.write('\n')
        with open(args.dataset_name + '_distri.goodcase', 'w') as f:
            for i in case_distri:
                f.write(str(i))
                f.write('\n')
        with open(args.dataset_name + '_distri.badcase', 'w') as f:
            for i in case_distri_bad:
                f.write(str(i))
                f.write('\n')
        print('bad case writed, early stop.')
        assert 1==0
    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** {0} results *****'.format(mode))
        logger.info("  {0} loss: {1}".format(mode, str(eval_loss)))
        for key in sorted(result.keys()):
            
            if key == 'epoch':
                continue
            logger.info(" {0} {1} = {2}".format(mode,key, str(result[key])))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results, eval_loss


def evaluate_badcase(args, eval_dataset, model, word_vocab):

    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)

        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            if args.embedding_type == 'bert':
                sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = args.tokenizer.decode(sent_ids)
                case['aspect'] = args.tokenizer.decode(aspect_ids)
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)
            else:
                sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = ' '.join([word_vocab['itos'][i] for i in sent_ids])
                case['aspect'] = ' '.join([word_vocab['itos'][i] for i in aspect_ids])
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, epoch):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        'epoch': epoch,
        "acc": acc,
        "f1": f1
    }


def compute_metrics(preds, labels, epoch):
    return acc_and_f1(preds, labels, epoch)
