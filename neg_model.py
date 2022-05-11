from config import CUE_MODEL, SCOPE_MODEL, TASK, F1_METHOD, SAVE_PATH, PRETRAINED_PATH
from transformers import BertForTokenClassification, RobertaForTokenClassification 
# from model import RobertaForTokenClassification
import torch
from torch.optim import Adam
from early_stopping import EarlyStopping
import numpy as np
from metrics import f1_score, f1_cues, f1_scope, flat_accuracy, flat_accuracy_positive_cues, report_per_class_accuracy, scope_accuracy
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, ReLU


class CueModel:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Cue_Detection.pickle', device = 'cuda', learning_rate = 3e-5, class_weight = [100, 100, 100, 1, 0], num_labels = 5):
        self.model_name = CUE_MODEL
        self.task = TASK
        if train == True:
        #     if 'xlnet' in CUE_MODEL:
        #         self.model = XLNetForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'xlnet-base-cased-model')
            if 'roberta' in CUE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'roberta-base-model')
            elif 'bert' in CUE_MODEL:
                self.model = BertForTokenClassification.from_pretrained(CUE_MODEL, num_labels=num_labels, cache_dir = 'bert_base_uncased_model')
            elif 'custom' in CUE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(PRETRAINED_PATH, num_labels=num_labels )
            
            else:
                raise ValueError("Supported model types are: roberta, bert")
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        if device == 'cuda':
            self.model.cuda()
        else:
            self.model.cpu()
            
        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            if intermediate_neurons == None:
                param_optimizer = list(self.model.classifier.named_parameters()) 
            else:
                param_optimizer = list(self.model.classifier.named_parameters())+list(self.model.int_layer.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

#     @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)  
    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):
        
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"{self.task} Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        loss_fn = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                if step % 100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                train_loss.append(loss.item())
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
            self.model.eval()
            eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
            predictions , true_labels, ip_mask = [], [], []
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                    with torch.no_grad():
                        logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                        active_labels = b_labels.view(-1)[active_loss]
                        tmp_eval_loss = loss_fn(active_logits, active_labels)
                        
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    mymasks = b_mymasks.to('cpu').numpy()
                    
                    if F1_METHOD == 'first_token':

                        logits = [list(p) for p in np.argmax(logits, axis=2)]
                        actual_logits = []
                        actual_label_ids = []
                        for l,lid,m in zip(logits, label_ids, mymasks):
                            actual_logits.append([i for i,j in zip(l,m) if j==1])
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                        logits = actual_logits
                        label_ids = actual_label_ids

                        predictions.append(logits)
                        true_labels.append(label_ids)
                    
                    elif F1_METHOD == 'average':

                        logits = [list(p) for p in logits]
                        
                        actual_logits = []
                        actual_label_ids = []
                        
                        for l,lid,m in zip(logits, label_ids, mymasks):
                            
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                            curr_preds = []
                            my_logits = []
                            in_split = 0
                            for i,j in zip(l,m):
                                if j==1:
                                    if in_split == 1:
                                        if len(my_logits)>0:
                                            curr_preds.append(my_logits[-1])
                                        mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                        if len(my_logits)>0:
                                            my_logits[-1] = mode_pred
                                        else:
                                            my_logits.append(mode_pred)
                                        curr_preds = []
                                        in_split = 0
                                    my_logits.append(np.argmax(i))
                                if j==0:
                                    curr_preds.append(i)
                                    in_split = 1
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                            actual_logits.append(my_logits)
                            
                        logits = actual_logits
                        label_ids = actual_label_ids
                        
                        predictions.append(logits)
                        true_labels.append(label_ids)
                    
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
                    eval_loss += tmp_eval_loss.mean().item()
                    valid_loss.append(tmp_eval_loss.mean().item())
                    eval_accuracy += tmp_eval_accuracy
                    if tmp_eval_positive_cue_accuracy!=None:
                        eval_positive_cue_accuracy+=tmp_eval_positive_cue_accuracy
                        steps_positive_cue_accuracy+=1
                    nb_eval_examples += b_input_ids.size(0)
                    nb_eval_steps += 1
                eval_loss = eval_loss/nb_eval_steps
                
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
            labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
            pred_flat = [p for p,l in zip(pred_flat, labels_flat) if l!=4]
            labels_flat = [l for l in labels_flat if l!=4]
            report_per_class_accuracy(labels_flat, pred_flat)
            print(classification_report(labels_flat, pred_flat))
            print("F1-Score Overall: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))
            p,r,f1 = f1_cues(labels_flat, pred_flat)
            if f1>return_dict['Best F1']:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            early_stopping(f1, self.model, SAVE_PATH)
        
            if early_stopping.early_stop:
                print("Early stopping")
                break

            labels_flat = [int(i!=3) for i in labels_flat]
            pred_flat = [int(i!=3) for i in pred_flat]
            print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat, average='weighted')))
            
        self.model.load_state_dict(torch.load(SAVE_PATH+'/state_dict.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

#     @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name):
        return_dict = {"Task": f"{self.task} Cue Detection",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        eval_loss, eval_accuracy, eval_scope_accuracy, eval_positive_cue_accuracy = 0, 0, 0, 0
        nb_eval_steps, nb_eval_examples, steps_positive_cue_accuracy = 0, 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss(weight=torch.Tensor(self.class_weight).to(self.device))
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch
            
            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)
                logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            mymasks = b_mymasks.to('cpu').numpy()

            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                actual_label_ids = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                logits = actual_logits
                label_ids = actual_label_ids

                predictions.append(logits)
                true_labels.append(label_ids)

            elif F1_METHOD == 'average':
                logits = [list(p) for p in logits]
                    
                actual_logits = []
                actual_label_ids = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                        
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j in zip(l,m):
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)

                logits = actual_logits
                label_ids = actual_label_ids
                
                predictions.append(logits)
                true_labels.append(label_ids)    
                
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            tmp_eval_positive_cue_accuracy = flat_accuracy_positive_cues(logits, label_ids)
        
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            if tmp_eval_positive_cue_accuracy != None:
                eval_positive_cue_accuracy += tmp_eval_positive_cue_accuracy
                steps_positive_cue_accuracy+=1
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy for Positive Cues: {}".format(eval_positive_cue_accuracy/steps_positive_cue_accuracy))
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        pred_flat = [p for p,l in zip(pred_flat, labels_flat) if l!=4]
        labels_flat = [l for l in labels_flat if l!=4]
        report_per_class_accuracy(labels_flat, pred_flat)
        print(classification_report(labels_flat, pred_flat))
        print("F1-Score: {}".format(f1_score(labels_flat,pred_flat,average='weighted')))
        p,r,f1 = f1_cues(labels_flat, pred_flat)
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        labels_flat = [int(i!=3) for i in labels_flat]
        pred_flat = [int(i!=3) for i in pred_flat]
        print("F1-Score Cue_No Cue: {}".format(f1_score(labels_flat,pred_flat,average='weighted')))
        print(return_dict)
        # print(len(true_labels))
        # print(true_labels)
        # print(len(predictions))
        # print(predictions)
        return return_dict

    def predict(self, dataloader):
        self.model.eval()
        predictions, ip_mask = [], []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            logits = logits.detach().cpu().numpy()
            mymasks = b_mymasks.to('cpu').numpy()
            #predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                for l,m in zip(logits, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                
                predictions.append(actual_logits)

            elif F1_METHOD == 'average':
                logits = [list(p) for p in logits]
                    
                actual_logits = []
                actual_label_ids = []
                for l,m in zip(logits, mymasks):
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j in zip(l,m):
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)
                predictions.append(actual_logits)
                
        return predictions

class ScopeModel:
    def __init__(self, full_finetuning = True, train = False, pretrained_model_path = 'Scope_Resolution_Augment.pickle', device = 'cuda', learning_rate = 3e-5):
        self.model_name = SCOPE_MODEL
        self.task = TASK
        self.num_labels = 2
        if train == True:
        #     if 'xlnet' in SCOPE_MODEL:
        #         self.model = XLNetForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'xlnet-base-cased-model')
            if 'roberta' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'roberta-base-model')
            elif 'bert' in SCOPE_MODEL:
                self.model = BertForTokenClassification.from_pretrained(SCOPE_MODEL, num_labels=self.num_labels, cache_dir = 'bert_base_uncased_model')
            elif 'custom' in SCOPE_MODEL:
                self.model = RobertaForTokenClassification.from_pretrained(PRETRAINED_PATH, num_labels=self.num_labels)
            
            else:
                raise ValueError("Supported model types are: roberta, bert")
        else:
            self.model = torch.load(pretrained_model_path)
        self.device = torch.device(device)
        if device=='cuda':
            self.model.cuda()
        else:
            self.model.cpu()

        if full_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

#     @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)    
    def train(self, train_dataloader, valid_dataloaders, train_dl_name, val_dl_name, epochs = 5, max_grad_norm = 1.0, patience = 3):
        self.train_dl_name = train_dl_name
        return_dict = {"Task": f"{self.task} Scope Resolution",
                       "Model": self.model_name,
                       "Train Dataset": train_dl_name,
                       "Val Dataset": val_dl_name,
                       "Best Precision": 0,
                       "Best Recall": 0,
                       "Best F1": 0}
        train_loss = []
        valid_loss = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        loss_fn = CrossEntropyLoss()
        for _ in tqdm(range(epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_mymasks = batch
                logits = self.model(b_input_ids, token_type_ids=None,
                             attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #2 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
                loss.backward()
                tr_loss += loss.item()
                train_loss.append(loss.item())
                if step%100 == 0:
                    print(f"Batch {step}, loss {loss.item()}")
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()
                self.model.zero_grad()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            
            self.model.eval()
            
            eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions , true_labels, ip_mask = [], [], []
            loss_fn = CrossEntropyLoss()
            for valid_dataloader in valid_dataloaders:
                for batch in valid_dataloader:
                    batch = tuple(t.to(self.device) for t in batch)
                    b_input_ids, b_input_mask, b_labels, b_mymasks = batch

                    with torch.no_grad():
                        logits = self.model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask)[0]
                        active_loss = b_input_mask.view(-1) == 1
                        active_logits = logits.view(-1, self.num_labels)[active_loss]
                        active_labels = b_labels.view(-1)[active_loss]
                        tmp_eval_loss = loss_fn(active_logits, active_labels)
                        
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    b_input_ids = b_input_ids.to('cpu').numpy()

                    mymasks = b_mymasks.to('cpu').numpy()
                        
                    if F1_METHOD == 'first_token':

                        logits = [list(p) for p in np.argmax(logits, axis=2)]
                        actual_logits = []
                        actual_label_ids = []
                        for l,lid,m in zip(logits, label_ids, mymasks):
                            actual_logits.append([i for i,j in zip(l,m) if j==1])
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                        # print('actual logits:', actual_logits)
                        logits = actual_logits
                        label_ids = actual_label_ids

                        predictions.append(logits)
                        true_labels.append(label_ids)
                    elif F1_METHOD == 'average':
                      
                        logits = [list(p) for p in logits]
                    
                        actual_logits = []
                        actual_label_ids = []
                        
                        for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):
                                
                            actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                            my_logits = []
                            curr_preds = []
                            in_split = 0
                            for i,j,k in zip(l,m, b_ii):
                                '''if k == 0:
                                    break'''
                                if j==1:
                                    if in_split == 1:
                                        if len(my_logits)>0:
                                            curr_preds.append(my_logits[-1])
                                        mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                        if len(my_logits)>0:
                                            my_logits[-1] = mode_pred
                                        else:
                                            my_logits.append(mode_pred)
                                        curr_preds = []
                                        in_split = 0
                                    my_logits.append(np.argmax(i))
                                if j==0:
                                    curr_preds.append(i)
                                    in_split = 1
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                            actual_logits.append(my_logits)
                        
                        logits = actual_logits
                        label_ids = actual_label_ids
                
                        predictions.append(logits)
                        true_labels.append(label_ids)    
                        # predictions.append(actual_logits)
                        # true_labels.append(actual_label_ids)    
                        
                    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                    tmp_eval_scope_accuracy = scope_accuracy(logits, label_ids)
                    eval_scope_accuracy += tmp_eval_scope_accuracy
                    valid_loss.append(tmp_eval_loss.mean().item())

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += len(b_input_ids)
                    nb_eval_steps += 1
                eval_loss = eval_loss/nb_eval_steps
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
            f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
            labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
            pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
            classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
            p = classification_dict["1"]["precision"]
            r = classification_dict["1"]["recall"]
            f1 = classification_dict["1"]["f1-score"]
            if f1>return_dict['Best F1']:
                return_dict['Best F1'] = f1
                return_dict['Best Precision'] = p
                return_dict['Best Recall'] = r
            print("F1-Score Token: {}".format(f1))
            print(classification_report(labels_flat, pred_flat))
            early_stopping(f1, self.model, SAVE_PATH)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        self.model.load_state_dict(torch.load(SAVE_PATH +'/state_dict.pt'))
        plt.xlabel("Iteration")
        plt.ylabel("Train Loss")
        plt.plot([i for i in range(len(train_loss))], train_loss)
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Validation Loss")
        plt.plot([i for i in range(len(valid_loss))], valid_loss)
        return return_dict

#     @telegram_sender(token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)
    def evaluate(self, test_dataloader, test_dl_name = "SFU"):
        return_dict = {"Task": f"{self.task} Scope Resolution",
                       "Model": self.model_name,
                       "Train Dataset": self.train_dl_name,
                       "Test Dataset": test_dl_name,
                       "Precision": 0,
                       "Recall": 0,
                       "F1": 0}
        self.model.eval()
        eval_loss, eval_accuracy, eval_scope_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels, ip_mask = [], [], []
        loss_fn = CrossEntropyLoss()
        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)[0]
                active_loss = b_input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] #5 is num_labels
                active_labels = b_labels.view(-1)[active_loss]
                tmp_eval_loss = loss_fn(active_logits, active_labels)
                
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            b_input_ids = b_input_ids.to('cpu').numpy()
            
            mymasks = b_mymasks.to('cpu').numpy()

            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                actual_label_ids = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])

                logits = actual_logits
                label_ids = actual_label_ids

                predictions.append(logits)
                true_labels.append(label_ids)

            elif F1_METHOD == 'average':
                
                logits = [list(p) for p in logits]
                
                actual_logits = []
                actual_label_ids = []
                
                for l,lid,m,b_ii in zip(logits, label_ids, mymasks, b_input_ids):
                        
                    actual_label_ids.append([i for i,j in zip(lid, m) if j==1])
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j,k in zip(l,m,b_ii):
                        '''if k == 0:
                            break'''
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)
                    
                predictions.append(actual_logits)
                true_labels.append(actual_label_ids)

            tmp_eval_accuracy = flat_accuracy(actual_logits, actual_label_ids)
            tmp_eval_scope_accuracy = scope_accuracy(actual_logits, actual_label_ids)
            eval_scope_accuracy += tmp_eval_scope_accuracy

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += len(b_input_ids)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        print("Validation Accuracy Scope Level: {}".format(eval_scope_accuracy/nb_eval_steps))
        f1_scope([j for i in true_labels for j in i], [j for i in predictions for j in i], level='scope')
        labels_flat = [l_ii for l in true_labels for l_i in l for l_ii in l_i]
        pred_flat = [p_ii for p in predictions for p_i in p for p_ii in p_i]
        classification_dict = classification_report(labels_flat, pred_flat, output_dict= True)
        p = classification_dict["1"]["precision"]
        r = classification_dict["1"]["recall"]
        f1 = classification_dict["1"]["f1-score"]
        return_dict['Precision'] = p
        return_dict['Recall'] = r
        return_dict['F1'] = f1
        print("Classification Report:")
        print(classification_report(labels_flat, pred_flat))
        print(return_dict)
        return return_dict

    def predict(self, dataloader):
        self.model.eval()
        predictions, ip_mask = [], []
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_mymasks = batch

            with torch.no_grad():
                logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]
            logits = logits.detach().cpu().numpy()
            mymasks = b_mymasks.to('cpu').numpy()

            if F1_METHOD == 'first_token':

                logits = [list(p) for p in np.argmax(logits, axis=2)]
                actual_logits = []
                for l,lid,m in zip(logits, label_ids, mymasks):
                    actual_logits.append([i for i,j in zip(l,m) if j==1])
                
                logits = actual_logits
                label_ids = actual_label_ids

                predictions.append(logits)
                true_labels.append(label_ids)

            elif F1_METHOD == 'average':
                
                logits = [list(p) for p in logits]
                
                actual_logits = []
                
                for l,m in zip(logits, mymasks):
                        
                    my_logits = []
                    curr_preds = []
                    in_split = 0
                    for i,j in zip(l,m):
                        
                        if j==1:
                            if in_split == 1:
                                if len(my_logits)>0:
                                    curr_preds.append(my_logits[-1])
                                mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                                if len(my_logits)>0:
                                    my_logits[-1] = mode_pred
                                else:
                                    my_logits.append(mode_pred)
                                curr_preds = []
                                in_split = 0
                            my_logits.append(np.argmax(i))
                        if j==0:
                            curr_preds.append(i)
                            in_split = 1
                    if in_split == 1:
                        if len(my_logits)>0:
                            curr_preds.append(my_logits[-1])
                        mode_pred = np.argmax(np.average(np.array(curr_preds, dtype = object), axis=0), axis=0)
                        if len(my_logits)>0:
                            my_logits[-1] = mode_pred
                        else:
                            my_logits.append(mode_pred)
                    actual_logits.append(my_logits)
                    
                predictions.append(actual_logits)
        return predictions
