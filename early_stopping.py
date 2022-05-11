import torch
from pathlib import Path

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = 0

    def __call__(self, score, model, save_path):

        #score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation F1 increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if Path(save_path).exists() is False:
            Path(save_path).mkdir()
        torch.save(model.state_dict(), save_path+'/state_dict.pt')
        # torch.save(model, save_path+'/model.pt')
        self.val_loss_min = val_loss
