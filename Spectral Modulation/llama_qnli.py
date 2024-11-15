import os
import time
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer,AutoConfig
from dataset_utils.bigbench import get_bb_dataset
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress
import random
import json
from dataset_utils.SST_2 import csv_to_lists


torch.manual_seed(0)  
torch.cuda.manual_seed_all(0)  
np.random.seed(0) 
random.seed(0)  


class Results:

    def __init__(self, val_acc, val_logloss, test_acc, test_logloss):
        self.val_acc = val_acc
        self.val_logloss = val_logloss
        self.test_acc = test_acc
        self.test_logloss = test_logloss

    def to_dict(self):
        return {
            "val_acc": self.val_acc,
            "val_logloss": self.val_logloss,
            "test_acc": self.test_acc,
            "test_logloss": self.test_logloss
        }

    def to_str(self, only_test=False):
        if only_test:
            return f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"
        else:
            return f"Validation acc {self.val_acc:.3f}, Validation logloss {self.val_logloss:.3f}, " \
                   f"Test acc {self.test_acc:.3f}, Test logloss {self.test_logloss:.3f}"


class LlamaExperiment:

    def __init__(self, save_dir, logger):

        self.save_dir = save_dir
        self.logger = logger

        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)

        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)

        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)

        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_choice_tokens(self, choices, tokenizer):

        choice_token_ids = []
        for choice in choices:
            assert not choice.startswith(" "), f"Expecting choice token {choice} to not start with space"
            assert not choice.endswith(" "), f"Expecting choice token {choice} to not end with space"

            token_ids = tokenizer(f"{choice}")
            if not (len(token_ids["input_ids"]) == 2 and token_ids["input_ids"][0] == 128000):
                # This is a multi-token target and so must be evaluated differently
                return None
            else:
                token_id = int(token_ids["input_ids"][1])
                choice_token_ids.append(token_id)

        return choice_token_ids

    def single_token_eval(self, prompt, label, model_edit, choices, choice_token_ids, tokenizer):

        input_and_answer = tokenizer(prompt, return_tensors="pt").to('cuda:1')

        # Generate from the model
        # Compute log probability of question + answer
        results = model_edit(input_and_answer.input_ids)
        logits = results.logits[0]  # question + answer length x vocab
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)  # question + answer length x vocab

        choice_logprobs = [log_prob[-1, choice_token_id].item() for choice_token_id in choice_token_ids]

        prediction_label_id = int(np.argmax(choice_logprobs))
        label_id = choices.index(label)

        is_correct = label_id == prediction_label_id

        answer_log_prob = choice_logprobs[label_id]
        log_prob_results = ContextAnswerLogProb(total_log_prob=answer_log_prob,
                                                answer_log_prob=answer_log_prob,
                                                answer_len=1)

        return is_correct, log_prob_results

    def multi_token_eval(self, prompt, label, model_edit, choices, tokenizer):

        all_log_prob_results = []

        for choice in choices:

            input_and_answer = tokenizer(prompt + " " + choice, return_tensors="pt").to(self.device)

            # Generate from the model
            # Compute log probability of question + answer
            results = model_edit(input_and_answer.input_ids)
            logits = results.logits[0]  # question + answer length x vocab
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)  # question + answer length x vocab

            log_prob_results = self.metrics.answer_log_prob(log_prob=log_prob,
                                                            question_answer_token_ids=input_and_answer.input_ids[0],
                                                            answer=choice,
                                                            llm_tokenizer=tokenizer)
            all_log_prob_results.append(log_prob_results)

        choice_logprobs = [log_prob_results.answer_log_prob for log_prob_results in all_log_prob_results]
        prediction_label_id = int(np.argmax(choice_logprobs))
        label_id = choices.index(label)

        is_correct = label_id == prediction_label_id
        log_prob_results = all_log_prob_results[label_id]

        return is_correct, log_prob_results

    def intervene(self, model, tokenizer, questions, answers,args, choices):

        dataset_size = len(questions)
        self.logger.log(f"Starting a new intervention for layer number {args.lnum}, "
                        f"layer type {args.lname}, rate {args.rate}. Dataset size {dataset_size}.")

        time_edit_start = time.time()
        if args.lname != "dont":
            model = LaserWrapper.get_edited_model(model=model,
                                                    lname=args.lname,
                                                    lnum=args.lnum,
                                                    rate=args.rate,
                                                    intervention=args.intervention,
                                                    logger=self.logger,
                                                    in_place=True,
                                                    threshold = args.threshold)
             
        #model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model.device} in time {elapsed_from_str(time_edit_start)}")

        predictions = []

        choice_token_ids = self.get_choice_tokens(choices, tokenizer)
        if choice_token_ids is None:
            single_token_choices = False
            self.logger.log(f"Set of choices {choices} is a multi-token set.")
        else:
            single_token_choices = True
            self.logger.log(f"Set of choices {choices} is a single token set with token ids {choice_token_ids}.")

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(0, dataset_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            prompt = questions[i]
            label = answers[i]

            with torch.no_grad():

                if single_token_choices:
                    is_correct, log_prob_results = self.single_token_eval(prompt=prompt,
                                                                          label=label,
                                                                          model_edit=model,
                                                                          choices=choices,
                                                                          choice_token_ids=choice_token_ids,
                                                                          tokenizer = tokenizer)
                else:
                    is_correct, log_prob_results = self.multi_token_eval(prompt=prompt,
                                                                         label=label,
                                                                         model_edit=model,
                                                                         choices=choices,
                                                                         tokenizer = tokenizer)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            # correct_log_prob_results = [all_log_prob_results[answer_ix] for answer_ix in correct_answers]
            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=log_prob_results)

            predictions_ = {
                "ix": i,
                "question": prompt,
                "gold-answer": label,
                "generation": "N/A",
                "correct": is_correct,
                "f1_score": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                "white-space-strip": self.strip,              # We ignore white space when checking answer
                "total_logprob": log_prob_results.total_log_prob,
                "answer_logprob": log_prob_results.answer_log_prob,
                "answer_length": log_prob_results.answer_len
            }
            predictions.append(predictions_)

        # Save results and terminate
        self.terminate_and_save(predictions)

        return predictions

    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        # time_start = time.time()
        # # Save predictions
        # save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.lnum}-{args.lname}-{args.rate}.p"

        # with open(save_pred_fname, "wb") as f:
        #     pickle.dump(predictions, f)

        # # Save the summary
        # save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.lnum}-{args.lname}-{args.rate}.pkl"

        # results = self.dataset_metric.agg_to_dict()
        # for k, v in args.__dict__.items():
        #     results["args/%s" % k] = v

        # with open(save_summary_fname, "wb") as f:
        #     pickle.dump(results, f)

        # # Print final numbers and return
        # self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")

    @staticmethod
    def get_acc_log_loss(predictions):

        acc = np.mean([1.0 if prediction["correct"] else 0.0 for prediction in predictions]) * 100.0
        log_loss = np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"])
                            for prediction in predictions])

        return acc, log_loss

    @staticmethod
    def validate(predictions, split=0.2):

        val_size = int(split * len(predictions))
        validation_predictions = predictions[:val_size]
        test_predictions = predictions[val_size:]

        val_acc, val_logloss = LlamaExperiment.get_acc_log_loss(validation_predictions)
        test_acc, test_logloss = LlamaExperiment.get_acc_log_loss(test_predictions)

        return Results(val_acc=val_acc,
                       val_logloss=val_logloss,
                       test_acc=test_acc,
                       test_logloss=test_logloss)


# Step 1: Command line argument
parser = argparse.ArgumentParser(description='Process Arguments for experiments with Vicuna LLM on CounterFact')

parser.add_argument('--threshold', type=float, default=64, help='protection threshold')
parser.add_argument('--rate', type=float, default=0.95, help='reduction factor')
parser.add_argument('--split', type=str, default="qa_wikidata", help='big bench split to run on')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
parser.add_argument('--max_len', type=int, default=10, help='maximum length for generation')
parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
parser.add_argument('--intervention', type=str, default="do_low_pass",
                        choices=['dropout', 'rank-reduction','ksvd','do_low_pass'], help="what type of intervention to perform") 
parser.add_argument('--lname', type=str, default=None,
                    choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont'],
                    help="provided which type of parameters to effect")
parser.add_argument('--lnum', type=int, default=0, help='Layers to edit', choices=list(range(-1, 32)))
parser.add_argument('--model_path',
                    type=str,
                    default="model_path",
                    help="Place where model weights are stored")

parser.add_argument('--llm_name',
                    type=str,
                    default="vicuna_7b",
                    help="Model Name")

parser.add_argument('--home_dir', type=str,
                    default="./results/qnli/vicuna_7b_results",
                    help='Directory where the data is')
args = parser.parse_args()



# Step 2: Load model and tokenizer
llm_name = args.llm_name
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path)
model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,device_map="auto")

# Step 3: Create save directory and logger
home_dir = args.home_dir
# split = args.split

save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.intervention}-{args.lnum}-{args.lname}-{args.rate}.txt")

# Step 4: Create an experiment
experiment = LlamaExperiment(save_dir=save_dir, logger=logger)

logger.log("=" * 50)
logger.log(f"Created a new Experiment. Model {llm_name}")
logger.log("=" * 50)

for k, v in args.__dict__.items():
    logger.log(f">>>> Command line argument {k} => {v}")
logger.log("=" * 50)

# Step 5: Read the dataset
questions, answers = csv_to_lists('qnli.csv')


# Step 6: Run intervention
predictions = experiment.intervene(model=model,
                                    tokenizer=tokenizer,
                                    questions=questions,
                                    answers=answers,
                                    args=args,
                                    choices=["entailment", "non-entailment"])
results = experiment.validate(predictions)
logger.log(f"{llm_name}, Lnum {args.lnum}, Lname {args.lname}, Rate {args.rate} => "
            f"Model results {results.to_str()}.")
logger.log("Experimented Completed.")

summary = results.to_dict()
for k, v in vars(args).items():
    summary[f"args/{k}"] = v

