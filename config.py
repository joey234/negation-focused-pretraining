# import torch

MAX_LEN = 128
bs = 8
EPOCHS = 30
PATIENCE = 30
INITIAL_LEARNING_RATE = 3e-5
NUM_RUNS = 1 #Number of times to run the training and evaluation code

# CUE_MODEL = 'bert-base-uncased'
CUE_MODEL = 'custom'
# SCOPE_MODEL = 'bert-base-uncased'
SCOPE_MODEL = 'custom'

SCOPE_METHOD = 'augment' # Options: augment, replace
F1_METHOD = 'average' # Options: average, first_token
TASK = 'negation' # Options: negation, speculation
SUBTASK = 'scope_resolution' # Options: cue_detection, scope_resolution
TRAIN_DATASETS = ['vetcompass']
TEST_DATASETS = ['bioscope_full_papers','bioscope_abstracts','sfu','sherlock', 'vetcompass']


BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-config.json"

}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin"
}

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin"
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json"
}

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json"
}

XLNET_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'xlnet-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin"
}

TF_WEIGHTS_NAME = 'model.ckpt'
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
SAVE_PATH = "scope_comb_vc"
# PRETRAINED_PATH = "roberta-maskneg-comb"
PRETRAINED_PATH = "random_mlm_full_ep5"
# PRETRAINED_PATH = "randomcomb_mlm_ep5"
# PRETRAINED_PATH = "roberta-maskneg-bioscope"
# PRETRAINED_PATH = "bioscope_random"


# PRETRAINED_PATH = "checkpoint-300000"
# device = torch.device("cuda")
# n_gpu = torch.cuda.device_count()

