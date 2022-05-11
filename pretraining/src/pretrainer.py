from transformers import Trainer, TrainingArguments, BertConfig, RobertaConfig, ElectraConfig
from transformers import HfArgumentParser
from transformers import BertTokenizerFast, RobertaTokenizerFast
from transformers import RobertaForMaskedLM, BertForMaskedLM
import transformers

from model.data_collator import DataCollatorForMaskedNegationCue
transformers.logging.set_verbosity_debug()

import torch
from torch.utils.data import DataLoader

import datasets
from datasets import concatenate_datasets
datasets.logging.set_verbosity(datasets.logging.ERROR)
from pathlib import Path
import time
import copy

from model import (
    RobertaForShuffledWordClassification,
    RobertaForShuffleRandomThreeWayClassification,
    RobertaForFourWayTokenTypeClassification,
    RobertaForFirstCharPrediction,
    RobertaForRandomWordClassification,
    RobertaForTwoWayTokenTypeClassification
)
from model import compute_metrics_fn_for_shuffle_random
from model import (
    DataCollatorForShuffledWordClassification,
    DataCollatorForShuffleRandomThreeWayClassification,
    DataCollatorForMaskedLanguageModeling,
    DataCollatorForFourWayTokenTypeClassification,
    DataCollatorForFirstCharPrediction,
    DataCollatorForRandomWordClassification,
    DataCollatorForNegationCuePrediction
)
from model import LoggingCallback

import logging
import os
import sys
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class AdditionalArguments:
    """Define additional arguments that are not included in `TrainingArguments`."""

    data_dir: str = field(
        metadata={"help": "Path to a processed dataset for pre-training"}
    )

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Local path to the model if the model to train has been instantiated from a local path. "
        + "If present, training will resume from the optimizer/scheduler states loaded here."}
    )

    hidden_size: int = field(
        default=768,
        metadata={"help": "Dimensionality of the encoder layers and the pooler layer."}
    )

    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "Number of hidden layers in the Transformer encoder."}
    )

    num_attention_heads: int = field(
        default=12,
        metadata={"help": "Number of attention heads for each attention layer in the Transformer encoder."}
    )

    intermediate_size: int = field(
        default=3072,
        metadata={"help": "Dimensionality of the intermediate (feed-forward) layer in the Transformer encoder."}
    )

    attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio for the attention probabilities."}
    )

    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler."}
    )

    pretrain_model: Optional[str] = field(
        default="RobertaForMaskedLM",
        metadata={"help": "The type of a model. Choose one from "
                  + "`RobertaForShuffledWordClassification`, "
                  + "`RobertaForShuffleRandomThreeWayClassification`, `RobertaForFirstCharPrediction`, "
                  + "`RobertaForMaskedLM`, `RobertaForFourWayTokenTypeClassification`, "
                  + "`RobertaForRandomWordClassification`."}
    )

    shuffle_prob: Optional[float] = field(
        default=0.15,
        metadata={"help": "[Shuffle] The ratio of shuffled words."}
    )

    mlm_prob: Optional[float] = field(
        default=0.15,
        metadata={"help": "[MLM] The ratio of masked tokens for MaskedLM."}
    )

    manipulate_prob: Optional[float] = field(
        default=0.10,
        metadata={"help": "[Shuffle+Random] The ratio of shuffled / random tokens over all tokens. "
        + "The resulting manipulated ratio will be twice larger than `manipulate_prob`."}
    )

    mask_prob: Optional[float] = field(
        default=0.15,
        metadata={"help": "[First Char, Token Type] The ratio of token masking."}
    )

    random_prob: Optional[float] = field(
        default=0.15,
        metadata={"help": "[Random] The ratio of random tokens."}
    )

    save_interval: Optional[float] = field(
        default=21600.0,
        metadata={"help": "An interval to save weights in seconds."}
    )

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
parser = HfArgumentParser((AdditionalArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    args, training_args = parser.parse_args_into_dataclasses()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)


def roberta_shuffled_cls():
    """Pre-train a RoBERTa model with shuffled word detection in a given sequence.

    Notes:
        * To see possible args, please run `python pretrainer.py --help`
        * To monitor training, run `tensorboard --logdir=/path/to/logging_dir/`

    References:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """

    # build a base model
    logger.info("Building a model...")
    if args.model_path is None:
        # pre-training from scratch
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig(
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        bos_token_id=0,
                        eos_token_id=2,
                        gradient_checkpointing=False,
                        hidden_act="gelu",
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        hidden_size=args.hidden_size,
                        initializer_range=0.02,
                        intermediate_size=args.intermediate_size,
                        layer_norm_eps=1e-05,
                        max_position_embeddings=514,
                        model_type="roberta",
                        num_attention_heads=args.num_attention_heads,
                        num_hidden_layers=args.num_hidden_layers,
                        pad_token_id=1,
                        type_vocab_size=1,
                        vocab_size=50265
                    )
        model = RobertaForShuffledWordClassification(config)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(config)
    else:
        # resume pre-training from a given checkpoint
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig.from_pretrained(args.model_path)
        model = RobertaForShuffledWordClassification.from_pretrained(args.model_path, config=config)

    logger.info(f"Save a checkpoint every {training_args.save_steps} steps.")
    logger.info(f"Logging every {training_args.logging_steps} steps.")

    # load datasets
    logger.info("Load the processed dataset...")
    full_dataset = []
    if Path(args.data_dir).exists() is False:
        raise FileNotFoundError("The specified dataset path does not exist!")
    # for ratio in range(0, 100, 10):
    #     temp_data_dir = Path(args.data_dir) / str(ratio)
    #     dataset = datasets.load_from_disk(temp_data_dir)
    #     if full_dataset != []:
    #         full_dataset = concatenate_datasets([full_dataset, dataset])
    #     else:
    #         full_dataset = dataset
    full_dataset = datasets.load_from_disk(args.data_dir)
    print(full_dataset)
    print(full_dataset.select([0])['text'])
    
    full_dataset.remove_columns_(['text'])
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    full_dataset = full_dataset.shuffle(seed=training_args.seed)

    # set up a trainer
    data_collator = DataCollatorForShuffledWordClassification(
        tokenizer=tokenizer,
        shuffle_prob=args.shuffle_prob
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=data_collator
    )
    
    # Add a callback
    trainer.add_callback(
        LoggingCallback(save_interval=args.save_interval)
    )

    # training
    # if `model_path` is not None, training will resume from the given checkpoint.
    logger.info("Training a model...")
    start_time = time.time()
    trainer.train(model_path=args.model_path)
    train_time = time.time() - start_time
    logger.info(f"Training time: {train_time}")

    # save final weights
    trainer.save_model(training_args.output_dir)


def roberta_maskedlm():
    """Pre-train a RoBERTa model with masked language modeling.

    Notes:
        * To see possible args, please run `python pretrainer.py --help`
        * To monitor training, run `tensorboard --logdir=/path/to/logging_dir/`

    References:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """

    # build a base model
    logger.info("Building a model...")
    if args.model_path is None:
        # pre-training from scratch
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig(
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        bos_token_id=0,
                        eos_token_id=2,
                        gradient_checkpointing=False,
                        hidden_act="gelu",
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        hidden_size=args.hidden_size,
                        initializer_range=0.02,
                        intermediate_size=args.intermediate_size,
                        layer_norm_eps=1e-05,
                        max_position_embeddings=514,
                        model_type="roberta",
                        num_attention_heads=args.num_attention_heads,
                        num_hidden_layers=args.num_hidden_layers,
                        pad_token_id=1,
                        type_vocab_size=1,
                        vocab_size=50265
                    )
        model = RobertaForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(config)
    else:
        # resume pre-training from a given checkpoint
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig.from_pretrained(args.model_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_path, config=config)
    
    cue_mask_token =  {'cue_token': '[CUE]'}
    num_added_toks = tokenizer.add_special_tokens(cue_mask_token)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Save a checkpoint every {training_args.save_steps} steps.")
    logger.info(f"Logging every {training_args.logging_steps} steps.")

    # load datasets
    logger.info("Load the processed dataset...")
    full_dataset = []
    if Path(args.data_dir).exists() is False:
        raise FileNotFoundError("The specified dataset path does not exist!")
    # for ratio in range(0, 100, 10):
    #     temp_data_dir = Path(args.data_dir) / str(ratio)
    #     dataset = datasets.load_from_disk(temp_data_dir)
    #     if full_dataset != []:
    #         full_dataset = concatenate_datasets([full_dataset, dataset])
    #     else:
    #         full_dataset = dataset
    full_dataset = datasets.load_from_disk(args.data_dir)
    print(full_dataset)
    print(full_dataset.select([0])['text'])
    
    full_dataset.remove_columns_(['text'])
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    full_dataset = full_dataset.shuffle(seed=training_args.seed)

    # set up a trainer
    data_collator = DataCollatorForMaskedLanguageModeling(
        tokenizer=tokenizer,
        mlm_prob=args.mlm_prob
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=data_collator
    )

    # Add a callback
    trainer.add_callback(
        LoggingCallback(save_interval=args.save_interval)
    )

    # training
    # if `model_path` is not None, training will resume from the given checkpoint.
    logger.info("Training a model...")
    start_time = time.time()
    trainer.train(model_path=args.model_path)
    train_time = time.time() - start_time
    logger.info(f"Training time: {train_time}")

    # save final weights
    trainer.save_model(training_args.output_dir)


def roberta_maskednegcue():
    """Pre-train a RoBERTa model with negation cue prediction.

    Notes:
        * To see possible args, please run `python pretrainer.py --help`
        * To monitor training, run `tensorboard --logdir=/path/to/logging_dir/`

    References:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """

    # build a base model
    logger.info("Building a model...")
    if args.model_path is None:
        # pre-training from scratch
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig(
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        bos_token_id=0,
                        eos_token_id=2,
                        gradient_checkpointing=False,
                        hidden_act="gelu",
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        hidden_size=args.hidden_size,
                        initializer_range=0.02,
                        intermediate_size=args.intermediate_size,
                        layer_norm_eps=1e-05,
                        max_position_embeddings=514,
                        model_type="roberta",
                        num_attention_heads=args.num_attention_heads,
                        num_hidden_layers=args.num_hidden_layers,
                        pad_token_id=1,
                        type_vocab_size=1,
                        vocab_size=50265
                    )
        model = RobertaForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(config)
    else:
        # resume pre-training from a given checkpoint
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig.from_pretrained(args.model_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_path, config=config)
    cue_mask_token =  {'additional_special_tokens': ['[CUE]']}
    num_added_toks = tokenizer.add_special_tokens(cue_mask_token)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Save a checkpoint every {training_args.save_steps} steps.")
    logger.info(f"Logging every {training_args.logging_steps} steps.")

    # load datasets
    logger.info("Load the processed dataset...")
    full_dataset = []
    if Path(args.data_dir).exists() is False:
        raise FileNotFoundError("The specified dataset path does not exist!")
    # for ratio in range(0, 100, 10):
    #     temp_data_dir = Path(args.data_dir) / str(ratio)
    #     dataset = datasets.load_from_disk(temp_data_dir)
    #     if full_dataset != []:
    #         full_dataset = concatenate_datasets([full_dataset, dataset])
    #     else:
    #         full_dataset = dataset
    full_dataset = datasets.load_from_disk(args.data_dir)
    print(full_dataset)
    print(full_dataset.select([0])['text'])
    
    full_dataset.remove_columns_(['text'])
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    full_dataset = full_dataset.shuffle(seed=training_args.seed)

    # set up a trainer
    data_collator = DataCollatorForMaskedNegationCue(
        tokenizer=tokenizer,
        mlm_prob=args.mlm_prob
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=data_collator
    )

    # Add a callback
    trainer.add_callback(
        LoggingCallback(save_interval=args.save_interval)
    )

    # training
    # if `model_path` is not None, training will resume from the given checkpoint.
    logger.info("Training a model...")
    start_time = time.time()
    trainer.train(model_path=args.model_path)
    train_time = time.time() - start_time
    logger.info(f"Training time: {train_time}")

    # save final weights
    trainer.save_model(training_args.output_dir)

def bert_maskednegcue():
    """Pre-train a BERT model with negation cue prediction.

    Notes:
        * To see possible args, please run `python pretrainer.py --help`
        * To monitor training, run `tensorboard --logdir=/path/to/logging_dir/`

    References:
        https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """

    # build a base model
    logger.info("Building a model...")
    if args.model_path is None:
        # pre-training from scratch
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        config = BertConfig()
        # config = BertConfig(
        #                 attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        #                 bos_token_id=0,
        #                 eos_token_id=2,
        #                 gradient_checkpointing=False,
        #                 hidden_act="gelu",
        #                 hidden_dropout_prob=args.hidden_dropout_prob,
        #                 hidden_size=args.hidden_size,
        #                 initializer_range=0.02,
        #                 intermediate_size=args.intermediate_size,
        #                 layer_norm_eps=1e-05,
        #                 max_position_embeddings=514,
        #                 model_type="bert",
        #                 num_attention_heads=args.num_attention_heads,
        #                 num_hidden_layers=args.num_hidden_layers,
        #                 pad_token_id=1,
        #                 type_vocab_size=1,
        #                 vocab_size=30522
        #             )
        model = BertForMaskedLM(config)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(config)
    else:
        # resume pre-training from a given checkpoint
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained(args.model_path)
        model = BertForMaskedLM.from_pretrained(args.model_path, config=config)
    cue_mask_token =  {'additional_special_tokens': ['[CUE]']}
    num_added_toks = tokenizer.add_special_tokens(cue_mask_token)
    model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Save a checkpoint every {training_args.save_steps} steps.")
    logger.info(f"Logging every {training_args.logging_steps} steps.")

    # load datasets
    logger.info("Load the processed dataset...")
    full_dataset = []
    if Path(args.data_dir).exists() is False:
        raise FileNotFoundError("The specified dataset path does not exist!")
    # for ratio in range(0, 100, 10):
    #     temp_data_dir = Path(args.data_dir) / str(ratio)
    #     dataset = datasets.load_from_disk(temp_data_dir)
    #     if full_dataset != []:
    #         full_dataset = concatenate_datasets([full_dataset, dataset])
    #     else:
    #         full_dataset = dataset
    full_dataset = datasets.load_from_disk(args.data_dir)
    print(full_dataset)
    print(full_dataset.select([0])['text'])
    
    full_dataset.remove_columns_(['text'])
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    full_dataset = full_dataset.shuffle(seed=training_args.seed)

    # set up a trainer
    data_collator = DataCollatorForMaskedNegationCue(
        tokenizer=tokenizer,
        mlm_prob=args.mlm_prob
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=data_collator
    )

    # Add a callback
    trainer.add_callback(
        LoggingCallback(save_interval=args.save_interval)
    )

    # training
    # if `model_path` is not None, training will resume from the given checkpoint.
    logger.info("Training a model...")
    start_time = time.time()
    trainer.train(model_path=args.model_path)
    train_time = time.time() - start_time
    logger.info(f"Training time: {train_time}")

    # save final weights
    trainer.save_model(training_args.output_dir)



def roberta_token_type_cls_neg():
    """Pre-train a RoBERTa model with two-way token type classification."""

    # build a base model
    logger.info("Building a model...")
    if args.model_path is None:
        # pre-training from scratch
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig(
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        bos_token_id=0,
                        eos_token_id=2,
                        gradient_checkpointing=False,
                        hidden_act="gelu",
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        hidden_size=args.hidden_size,
                        initializer_range=0.02,
                        intermediate_size=args.intermediate_size,
                        layer_norm_eps=1e-05,
                        max_position_embeddings=514,
                        model_type="roberta",
                        num_attention_heads=args.num_attention_heads,
                        num_hidden_layers=args.num_hidden_layers,
                        pad_token_id=1,
                        type_vocab_size=1,
                        vocab_size=50265
                    )
        model = RobertaForTwoWayTokenTypeClassification(config)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(config)
    else:
        # resume pre-training from a given checkpoint
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        config = RobertaConfig.from_pretrained(args.model_path)
        model = RobertaForTwoWayTokenTypeClassification.from_pretrained(args.model_path, config=config)

    logger.info(f"Save a checkpoint every {training_args.save_steps} steps.")
    logger.info(f"Logging every {training_args.logging_steps} steps.")

    # load datasets
    logger.info("Load the processed dataset...")
    full_dataset = []
    if Path(args.data_dir).exists() is False:
        raise FileNotFoundError("The specified dataset path does not exist!")
    # for ratio in range(0, 100, 10):
    #     temp_data_dir = Path(args.data_dir) / str(ratio)
    #     dataset = datasets.load_from_disk(temp_data_dir)
    #     if full_dataset != []:
    #         full_dataset = concatenate_datasets([full_dataset, dataset])
    #     else:
    #         full_dataset = dataset
    full_dataset = datasets.load_from_disk(args.data_dir)
    print(full_dataset)
    print(full_dataset.select([0])['text'])
    full_dataset.remove_columns_(['text'])
    full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    full_dataset = full_dataset.shuffle(seed=training_args.seed)

    # set up a trainer
    data_collator = DataCollatorForNegationCuePrediction(
        tokenizer=tokenizer,
        mask_prob=args.mask_prob
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        data_collator=data_collator
    )

    # Add a callback
    trainer.add_callback(
        LoggingCallback(save_interval=args.save_interval)
    )

    # training
    # if `model_path` is not None, training will resume from the given checkpoint.
    logger.info("Training a model...")
    start_time = time.time()
    trainer.train(model_path=args.model_path)
    train_time = time.time() - start_time
    logger.info(f"Training time: {train_time}")

    # save final weights
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    if args.pretrain_model == "RobertaForShuffledWordClassification":
        roberta_shuffled_cls()
    elif args.pretrain_model == "RobertaForShuffleRandomThreeWayClassification":
        roberta_shuffle_random_threeway_cls()
    elif args.pretrain_model == "RobertaForMaskedLM":
        roberta_maskedlm()
    elif args.pretrain_model == "RobertaForFourWayTokenTypeClassification":
        roberta_token_type_cls()
    elif args.pretrain_model == "RobertaForFirstCharPrediction":
        roberta_first_char_cls()
    elif args.pretrain_model == "RobertaForRandomWordClassification":
        roberta_random_cls()
    elif args.pretrain_model == "RobertaForMaskedNegationCue":
        roberta_maskednegcue()
    elif args.pretrain_model == "RobertaForNegationCuePrediction":
        roberta_token_type_cls_neg()
    elif args.pretrain_model == "BertForMaskedNegationCue":
        bert_maskednegcue()

    else:
        raise NotImplementedError()
