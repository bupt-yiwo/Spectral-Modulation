import os
import time
import torch
import pickle
import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer,AutoConfig
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from dataset_utils.truthfulqa import get_truthfulqa_pointwise_data
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics, ContextAnswerLogProb
from study_utils.time_utils import elapsed_from_str, Progress
#from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec
from copy import deepcopy
import torch.nn as nn
import json
import random
import numpy as np



torch.manual_seed(0)  
torch.cuda.manual_seed_all(0)  
np.random.seed(0) 
random.seed(0)   


class VicunaExperiment:

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

    def intervene(self, model, tokenizer, dataset, args):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

        time_edit_start = time.time()
        if args.ifdo == "do":
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

        # Answer tokens: true and false
        true_token_ids = tokenizer(" true")
        assert len(true_token_ids["input_ids"]) == 2 and true_token_ids["input_ids"][0] == 1
        true_token_id = int(true_token_ids["input_ids"][1])

        false_token_ids = tokenizer(" false")
        assert len(false_token_ids["input_ids"]) == 2 and false_token_ids["input_ids"][0] == 1
        false_token_id = int(false_token_ids["input_ids"][1])

        # Reset dataset metrics and set progress timestamp
        self.dataset_metric.reset()
        self.progress.start()

        for i in tqdm(range(0, dataset_size)):

            if (i - 1) % 100 == 0 and i > 1:
                # Print partial performance and telemetry data
                self.dataset_metric.print()
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            prompt = dataset[i][0]
            label = dataset[i][1]
            # We compute two types of metric
            # - LogLoss of all examples
            # - If argmax loss is correct
            with torch.no_grad():

                input_and_answer = tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate from the model
                # Compute log probability of question + answer
                results = model(input_and_answer.input_ids)
                logits = results.logits[0]                                      # question + answer length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question + answer length x vocab

                true_log_prob = log_prob[-1, true_token_id].item()
                false_log_prob = log_prob[-1, false_token_id].item()
                true_false_logprobs = {"true": true_log_prob, "false": false_log_prob}

                if label == 0:          # False
                    is_correct = false_log_prob > true_log_prob
                    answer_log_prob = false_log_prob
                else:                   # False
                    assert label == 1, f"Label must be either 0 or 1. Found {label}"
                    is_correct = true_log_prob > false_log_prob
                    answer_log_prob = true_log_prob

                log_loss = - answer_log_prob
                log_prob_results = ContextAnswerLogProb(total_log_prob=answer_log_prob,
                                                        answer_log_prob=answer_log_prob,
                                                        answer_len=1)

            # We compute 0-1 match, f1, precision, and recall score in addition to log-prob of the answer tokens
            # correct_log_prob_results = [all_log_prob_results[answer_ix] for answer_ix in correct_answers]
            self.dataset_metric.accept(is_correct=is_correct,
                                       f1pr_score=None,
                                       log_prob_results=log_prob_results)

            predictions_ = {
                "ix": i,
                "question": prompt,
                "log_losses": log_loss,
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
                "answer_length": log_prob_results.answer_len,
                "true_false_log_probs": true_false_logprobs,
                "question_answer_length": input_and_answer.input_ids.shape[1]
            }
            predictions.append(predictions_)

        # Save results and terminate
        self.terminate_and_save(predictions)

        return sum(1 for predictions_ in predictions if predictions_["correct"])/len(predictions) * 100, np.mean([-prediction["answer_logprob"]/float(prediction["answer_length"]) for prediction in predictions])
         
    def terminate_and_save(self, predictions):

        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()

        # time_start = time.time()
        # # Save predictions
        # save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"

        # with open(save_pred_fname, "wb") as f:
        #     pickle.dump(predictions, f)

        # Save the summary
        # save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        # results = self.dataset_metric.agg_to_dict()
        # for k, v in args.__dict__.items():
        #     results["args/%s" % k] = v

        # with open(save_summary_fname, "wb") as f:
        #     pickle.dump(results, f)

        # Print final numbers and return
        # self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")

    def evaluate(self, test_logits, temp):

        mean_log_prob = 0.0

        for indices, logit in test_logits:

            indices = torch.from_numpy(indices).to(self.device)
            logit = torch.from_numpy(logit).to(self.device)

            log_prob = torch.nn.functional.log_softmax(logit / temp, dim=1)  # answer_length x vocab
            indices = indices.view(-1, 1)                                    # answer_length x 1

            selected_log_prob = torch.gather(log_prob,
                                             index=indices,
                                             dim=1)  # answer_length x 1
            log_prob = selected_log_prob.sum().item()

            mean_log_prob += log_prob / float(indices.shape[0])

        mean_log_prob /= float(len(test_logits))

        self.logger.log(f"Temperature {temp}: Mean log prob {mean_log_prob} on test set of size {len(test_logits)}.")

    # def temperature_tuning(self, predictions, val=0.2):

    #     val_size = int(val * len(predictions))
    #     validation_predictions = predictions[:val_size]
    #     test_predictions = predictions[val_size:]
    #     self.logger.log(f"Starting temperature tuning with validation set of size {len(validation_predictions)} and"
    #                     f"a test set of size {len(test_predictions)}.")

    #     validation_logits = [answer_logits_ for prediction in validation_predictions
    #                          for answer_logits_ in prediction["answer_logits"]]

    #     test_logits = [answer_logits_ for prediction in test_predictions
    #                    for answer_logits_ in prediction["answer_logits"]]

    #     self.logger.log(f"Evaluating with temperature {1.0}")
    #     self.evaluate(test_logits, 1.0)

    #     lr = 0.001
    #     temp_logit = nn.Parameter(torch.FloatTensor([1.0]))
    #     optimizer = opt.Adam([temp_logit], lr=lr)

    #     for epoch in range(1000):

    #         total_loss = 0.0
    #         for indices, logit in validation_logits:

    #             indices = torch.from_numpy(indices).to(self.device)
    #             logit = torch.from_numpy(logit).to(self.device)

    #             temp = torch.nn.functional.sigmoid(temp_logit)
    #             log_prob = torch.nn.functional.log_softmax(logit / temp, dim=1)     # answer_length x vocab
    #             indices = indices.view(-1, 1)                                       # answer_length x 1

    #             selected_log_prob = torch.gather(log_prob,
    #                                              index=indices,
    #                                              dim=1)                             # answer_length x 1
    #             loss = - selected_log_prob.sum()

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item()

    #         temp = torch.nn.functional.sigmoid(temp_logit)
    #         self.logger.log(f"Epoch {epoch+1}, loss is {total_loss/float(len(validation_logits)):.3f}. "
    #                         f"Current value of temperature is {temp.item()}.")

    #         if epoch % 100 == 0:
    #             self.logger.log(f"Evaluating with temperature {temp.item()}")
    #             self.evaluate(test_logits, temp.item())



# Step 1: Command line argument
parser = argparse.ArgumentParser(description='Process Arguments for experiments with Vicuna LLM on CounterFact')
parser.add_argument('--ifdo', type=str, default="do", help='rates for intervention')
parser.add_argument('--threshold', type=float, default=64, help='rates for intervention')
parser.add_argument('--rate', type=float, default=0.95, help='rates for intervention')
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
                    default="./results/truthfulqa/vicuna_7b_results",
                    help='Directory where the data is')
args = parser.parse_args()



# Step 2: Load model and tokenizer
llm_name = args.llm_name
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path)
model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,device_map="auto")


# Step 3: Create save directory and logger
home_dir = args.home_dir
#dataset_loc = args.dataset_file

save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.intervention}-{args.lnum}-{args.lname}-{args.rate}.txt")

# Step 4: Create an experiment
experiment = VicunaExperiment(save_dir=save_dir, logger=logger)

logger.log("=" * 50)
logger.log(f"Created a new Experiment. Model {llm_name}")
logger.log("=" * 50)

for k, v in args.__dict__.items():
    logger.log(f">>>> Command line argument {k} => {v}")
logger.log("=" * 50)

# Step 5: Read the dataset
dataset = get_truthfulqa_pointwise_data(logger)

# Step 6: Run intervention
acc ,log_loss= experiment.intervene(model=model,
                                    tokenizer=tokenizer,
                                    dataset=dataset,
                                    args=args)
