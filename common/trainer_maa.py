import os
import time
import torch
import datetime
import numpy as np
from common.utils import multi_acc, multi_mse, load_document4baseline_from_local, ensureDirs
from models.get_optim import get_Adam_optim, get_Adam_optim_v2
from models.model import MAAModel

SAVED_MODEL_PATH = '/saved_model'

class MAATrainer(object):
    def __init__(self, config):
        self.config = config
        from transformers import BertTokenizer
        pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

        self.train_itr, self.dev_itr, self.test_itr, self.usr_stoi, \
        self.prd_stoi, self.ctgy_stoi, self.keyword_itos, self.pos_embeddings, self.neg_embeddings = load_document4baseline_from_local(
            config)
        self.poskwcount = config.num_posembed
        self.negkwcount = config.num_negembed
        self.pad_tensor = torch.zeros((abs(self.poskwcount - self.negkwcount), self.config.hidden_size)).to(config.device)  # Padding with zeros
        diff = self.poskwcount - self.negkwcount
        positive_keywords = self.pos_embeddings.to(config.device) 
        negative_keywords = self.neg_embeddings.to(config.device) 
        if diff > 0:
            negative_keywords = torch.cat((negative_keywords, self.pad_tensor), dim=0)
        elif diff < 0:
            positive_keywords = torch.cat((positive_keywords , self.pad_tensor), dim=0)
        keyword_pool = torch.stack((positive_keywords, negative_keywords), dim=1).reshape(-1, positive_keywords.shape[1])
        
        
        
        model = MAAModel.from_pretrained(pretrained_weights, num_hidden_layers=config.n_totallayer, num_labels=config.num_labels, cus_config=config, positivekeyword_embeddings=positive_keywords, negativekeyword_embeddings=negative_keywords, interleaved_keyword_pool=keyword_pool)
        if self.config.n_gpu > 1:
            self.net = torch.nn.DataParallel(model).to(config.device)
            # self.net = model.to(config.device)
        else:
            self.net = model.to(config.device)
        self.optim = get_Adam_optim(config, self.net)
        self.optim, self.scheduler = get_Adam_optim_v2(config, self.net)

        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.oom_time = 0

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.config.version)
            self.train()
        else:
            try:
                model = MAAModel.from_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset),
                                                 num_hidden_layers=self.config.n_totallayer,
                                                 num_labels=self.config.num_labels, cus_config=self.config)
                if self.config.n_gpu > 1:
                    self.net = torch.nn.DataParallel(model).to(self.config.device)
                else:
                    self.net = model.to(self.config.device)
            except:
                print("Local model is miss. Please train a model first!")
                exit()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr if run_mode == 'val' else self.test_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, "validation" if run_mode == 'val' else 'test')
            print("\r" + eval_logs)

    def empty_log(self, version):
        log_dir = self.config.log_path
        log_filename = f"log_run_{self.config.dataset}_{self.config.version}_{self.config.attributes}_KW-{self.config.kw_attention_nums}_mmalayers-{self.config.n_mmalayer}_kwalayers-{self.config.n_kwalayer}_type-{self.config.type}.txt"
        log_path = os.path.join(log_dir, log_filename)

        if os.path.exists(log_path):
            # Loop to find an available name
            num = 1
            while os.path.exists(os.path.join(log_dir, f"{log_filename}_{num}")):
                num += 1
            new_log_path = os.path.join(log_dir, f"{log_filename}_{num}")
            os.rename(log_path, new_log_path)

        print('Initializing log file ........')
        print('Finished!\n')

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, loss, acc, rmse, eval='training'):
        logs = \
            '==={:10} phrase...'.format(eval) + "".center(60, " ") + "\n" + \
            '\t'.join(["{:<6}"] * 3).format("loss", "acc", "rmse") + '\n' + \
            '\t'.join(["{:^6.3f}"] * 3).format(loss, acc, rmse) + '\n'
        return logs

    def train(self):
        log_dir = self.config.log_path
        log_filename = f"log_run_{self.config.dataset}_{self.config.version}_{self.config.attributes}_KW-{self.config.kw_attention_nums}_mmalayers-{self.config.n_mmalayer}_kwalayers-{self.config.n_kwalayer}_type-{self.config.type}.txt"
        log_path = os.path.join(log_dir, log_filename)
        # Save log information
        logfile = open(
            log_path,
            'a+'
        )
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.close()
        for epoch in range(0, self.config.TRAIN.max_epoch):
            self.net.train()
            train_loss, train_acc, train_rmse = self.train_epoch()
            # train_loss, train_acc, train_rmse = 0.,0.,0.

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_loss, train_acc, train_rmse, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(log_path,logs)
            self.net.eval()
            with torch.no_grad():
                eval_loss, eval_acc, eval_rmse = self.eval(self.dev_itr)
            eval_logs = self.get_logging(eval_loss, eval_acc, eval_rmse, eval="evaluating")
            print("\r" + eval_logs)

            # logging evaluating logs
            self.logging(log_path, eval_logs)

            # early stopping
            if eval_acc > self.best_dev_acc:
                self.unimproved_iters = 0
                self.best_dev_acc = eval_acc
                # saving models
                ensureDirs(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
                self.tokenizer.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
                if self.config.n_gpu > 1:
                    self.net.module.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))
                else:
                    self.net.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset))

            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                    early_stop_logs = log_path + "\n" + \
                                      "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                    print(early_stop_logs)
                    self.logging(log_path, early_stop_logs)
                    break

    def train_epoch(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        start_time = 0.
        resume_batch = True
        
        for step, batch in enumerate(self.train_itr):
            
            
            torch.cuda.empty_cache()
            if resume_batch:
                start_time = time.time()
            input_ids, label, usr, prd, ctgy, keywordlist = batch
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != 100).long().to(self.config.device)  # id of <PAD> is 100
            labels = label.long().to(self.config.device)
            
            usr = torch.Tensor([self.usr_stoi[x] for x in usr]).long().to(self.config.device)
            prd = torch.Tensor([self.prd_stoi[x] for x in prd]).long().to(self.config.device)
            ctgy = torch.Tensor([self.ctgy_stoi[x] for x in ctgy]).long().to(self.config.device)
           
            
            try:
                logits = self.net(input_ids=input_ids,
                                    attrs=(usr, prd, ctgy),
                                    attention_mask=attention_mask)[0]
                

                loss = loss_fn(logits, labels)
                metric_acc = acc_fn(labels, logits)
                metric_mse = mse_fn(labels, logits)
                total_loss.append(loss.data.cpu().numpy())
                total_acc.append(metric_acc.data.cpu().numpy())
                total_mse.append(metric_mse.data.cpu().numpy())

                if self.config.TRAIN.gradient_accumulation_steps > 1:
                    loss = loss / self.config.TRAIN.gradient_accumulation_steps

                loss.backward()
                if (step + 1) % self.config.TRAIN.gradient_accumulation_steps == 0:
                    self.optim.step()
                    self.scheduler.step()
                    self.optim.zero_grad()

                    # Monitoring results on every batch * gradient_accumulation_steps
                    end_time = time.time()
                    span_time = (end_time - start_time) * (
                        int(len(self.train_itr) - step)) // self.config.TRAIN.gradient_accumulation_steps
                    h = span_time // (60 * 60)
                    m = (span_time % (60 * 60)) // 60
                    s = (span_time % (60 * 60)) % 60 // 1
                    print(
                        "\rIteration: {:>4}/{} ({:>4.1f}%) -- Loss: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                            step // self.config.TRAIN.gradient_accumulation_steps,
                            int(len(self.train_itr) // self.config.TRAIN.gradient_accumulation_steps),
                            100 * step / int(len(self.train_itr)),
                            loss,
                            int(h), int(m), int(s)),
                        end="")
                    resume_batch = True
                else:
                    resume_batch = False

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    self.oom_time += 1
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    print(str(exception))
                    raise exception

        return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())

    def eval(self, eval_itr):
        loss_fn = torch.nn.CrossEntropyLoss()
        acc_fn = multi_acc
        mse_fn = multi_mse
        total_loss = []
        total_acc = []
        total_mse = []
        for step, batch in enumerate(eval_itr):
            start_time = time.time()
            input_ids, label, usr, prd, ctgy, keywordlist = batch
            input_ids = input_ids.to(self.config.device)
            attention_mask = (input_ids != 100).long().to(self.config.device)  # id of <PAD> is 100
            labels = label.long().to(self.config.device)
            
            usr = torch.Tensor([self.usr_stoi[x] for x in usr]).long().to(self.config.device)
            prd = torch.Tensor([self.prd_stoi[x] for x in prd]).long().to(self.config.device)
            ctgy = torch.Tensor([self.ctgy_stoi[x] for x in ctgy]).long().to(self.config.device)
            logits = self.net(input_ids=input_ids,
                                attrs=(usr, prd, ctgy),
                                attention_mask=attention_mask)[0]
            

            loss = loss_fn(logits, labels)
            metric_acc = acc_fn(labels, logits)
            metric_mse = mse_fn(labels, logits)
            total_loss.append(loss.data.cpu().numpy())
            total_acc.append(metric_acc.data.cpu().numpy())
            total_mse.append(metric_mse.data.cpu().numpy())


            # monitoring results on every steps
            end_time = time.time()
            span_time = (end_time - start_time) * (
                    int(len(eval_itr)) - step)
            h = span_time // (60 * 60)
            m = (span_time % (60 * 60)) // 60
            s = (span_time % (60 * 60)) % 60 // 1
            print(
                "\rIteration: {:>4}/{} ({:>4.1f}%) -- Loss: {:.5f} -ETA {:>2}h-{:>2}m-{:>2}s".format(
                    step, int(len(eval_itr)),
                    100 * (step) / int(len(eval_itr)),
                    loss,
                    int(h), int(m), int(s)),
                end="")
        return np.array(total_loss).mean(), np.array(total_acc).mean(), np.sqrt(np.array(total_mse).mean())