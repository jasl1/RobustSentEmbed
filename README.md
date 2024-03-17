# RobustSentEmbed
A self-supervised sentence embedding framework that enhances both generalization and robustness benchmarks


#### Train the RobustSentEmbed embeddings to generate robust text represnetation
```bash
LR=7e-6
MASK=0.30
LAMBDA=0.005

!python -m torch.distributed.launch --nproc_per_node 4 --master_port $(expr $RANDOM + 1000) train.py \
    --model_name_or_path bert-base-uncased \
    --generator_name distilbert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir /data/long/RNLP/result/RobustSCE3_bert \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate 7e-6 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir your_logging_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight 0.005 \
    --fp16 --masking_ratio 0.20
```

#### Evaluate the RobustSentEmbed embeddings on STS and Transfer tasks
```bash
LR=7e-6
MASK=0.30
LAMBDA=0.005

!python train.py \
    --model_name_or_path /data/long/RNLP/result/RobustSCE3_bert \
    --generator_name distilbert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir /data/long/RNLP/result/RobustSCE3_bert_eval \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --temp 0.05 \
    --do_eval \
    --batchnorm \
    --lambda_weight $LAMBDA \
    --fp16 --masking_ratio $MASK

```

#### Evaluate the RobustSentEmbed embeddings using various adversarial attack techniques.
The following code snippet evaluates the RobustSentEmbed embeddings using the TextFooler adversarial attack for the IMDB task. Users can switch to different adversarial attacks by uncommenting the corresponding attack technique in the code. Additionally, users can load another dataset (e.g., sst2 or cola) to assess the embeddings for a different task.

```python
import textattack
import random
import transformers
import datasets
from adversarial_fine_tunning import BertForAT
from datasets import load_dataset


mnli_dataset = load_dataset('imdb') #load different dataset
train_dataset = textattack.datasets.HuggingFaceDataset(mnli_dataset['train'].shuffle())
eval_dataset = textattack.datasets.HuggingFaceDataset(mnli_dataset['test'].shuffle())


model_name = '/data/long/RNLP/result/RobustSCE3_bert'
config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path = model_name, num_labels=num_labels)
model = BertForAT.from_pretrained(pretrained_model_name_or_path = model_name, config=config)         
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case= True)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

training_args = textattack.TrainingArgs(
    num_epochs=3,
    parallel=True,
    learning_rate=5e-5, #1e-5
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    log_to_tb=True,
)

trainer = textattack.Trainer(
    model_wrapper,
    "classification", # regression, classification
    None,
    train_dataset,
    eval_dataset,
    training_args
)
trainer.train()


#attack = textattack.attack_recipes.PWWSRen2019.build(trainer.model_wrapper)
attack = textattack.attack_recipes.TextFoolerJin2019.build(trainer.model_wrapper)
#attack = textattack.attack_recipes.TextBuggerLi2018.build(trainer.model_wrapper)
#attack = textattack.attack_recipes.BAEGarg2019.build(trainer.model_wrapper)
#attack = textattack.attack_recipes.BERTAttackLi2020.build(trainer.model_wrapper)

attack_args = textattack.AttackArgs(num_examples=1000, disable_stdout=True)
attacker = textattack.Attacker(attack, eval_dataset, attack_args)
attacker.attack_dataset()
```
