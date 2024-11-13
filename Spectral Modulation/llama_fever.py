import os
import time
import torch
import pickle
import argparse
from tqdm import tqdm
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from dataset_utils.fever import FEVER
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
        # self.device = "cpu"

    def intervene(self, model, tokenizer, dataset, args):

        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")

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
    

        with torch.no_grad():
            #model_edit.to(self.device)
            
            
            self.logger.log(f"Edited and put model on {model.device} in time {elapsed_from_str(time_edit_start)}")

            predictions = []
            # Reset dataset metrics and set progress timestamp
            self.dataset_metric.reset()
            self.progress.start()

            # Answer tokens: true and false
            true_token_ids = tokenizer(" true")
            assert len(true_token_ids["input_ids"]) == 2 and true_token_ids["input_ids"][0] == 128000
            true_token_id = int(true_token_ids["input_ids"][1])

            false_token_ids = tokenizer(" false")
            assert len(false_token_ids["input_ids"]) == 2 and false_token_ids["input_ids"][0] == 128000
            false_token_id = int(false_token_ids["input_ids"][1])

            for i in tqdm(range(0, dataset_size)):

                if (i - 1) % 500 == 0 and i > 1:
                    # Print partial performance and telemetry data
                    self.dataset_metric.print()
                    self.progress.print(ex_done=i, ex_left=(dataset_size - i))

                question = dataset[i]["question"]

                # Answer is either 0 (False) or 1 (True)
                answer_ix = dataset[i]["answer"]
                # Given that we do 1-token look up we do the following:
                # - Compute log-prob of the gold token
                # - Compute top-1, top-5 and top-10 accuracies
                if question.strip().endswith(".") or question.strip().endswith("?"):
                    # prompted_question = "Is the following claim true or false: " + question.strip() + " The claim is "
                    prompted_question = "Consider the following claim: " + \
                                        question.strip() + " Is this claim true or false. The claim is"
                else:
                    # prompted_question = "Is the following claim true or false: " + question.strip() + ". The claim is "
                    prompted_question = "Consider the following claim: " + \
                                        question.strip() + ". Is this claim true or false. The claim is"
                assert answer_ix in [0, 1]

                inputs = tokenizer(prompted_question, return_tensors="pt").to(self.device)

                
                # Compute log probability of question
                results = model(inputs.input_ids,return_dict=True)
                logits = results.logits[0]                                      # question length x vocab
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab
                #print(torch.mean(results.loss["logits"]).item())
                # print(type(results.loss))
                # print(results.loss.dim)
                # print(results.loss.shape)
                last_token_logprob = log_prob[-1]                               # vocab

                true_logprob = last_token_logprob[true_token_id].item()
                false_logprob = last_token_logprob[false_token_id].item()

                if answer_ix == 1:     # Answer is True
                    answer_log_prob = true_logprob
                    is_correct = true_logprob > false_logprob
                    answer = "true"
                else:               # Answer is False
                    answer_log_prob = false_logprob
                    is_correct = true_logprob < false_logprob
                    answer = "false"

                sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)

                top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
                top_k_indices = sorted_indices[:10].detach()

                decoded_tokens = tokenizer.batch_decode(top_k_indices)
                top_k_tokens = [token for token in decoded_tokens]
                assert len(top_k_tokens) == 10

                top_1_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:1]])
                top_5_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:5]])
                top_10_acc = float(answer.lower().strip() in [token.lower().strip() for token in top_k_tokens[:10]])

                # Compute log-prob of question and answer
                selected_log_prob = log_prob[:-1, :]  # question - 1 x vocab
                indices = inputs.input_ids[0, 1:].unsqueeze(1)  # question - 1 x 1

                selected_log_prob = torch.gather(selected_log_prob,
                                                    index=indices,
                                                    dim=1)  # question - 1 x 1
                question_log_prob = selected_log_prob.sum().item()
                total_log_prob = question_log_prob + answer_log_prob

                logprob_results = ContextAnswerLogProb(total_log_prob=total_log_prob,
                                                        answer_log_prob=answer_log_prob,
                                                        answer_len=1)
                self.dataset_metric.accept(is_correct=is_correct,
                                        f1pr_score=None,
                                        log_prob_results=logprob_results,
                                        top_k_acc={1: top_1_acc, 5: top_5_acc, 10: top_10_acc})

                # if i % 10 == 0:
                #     print(f"Question: {question} and gold answer {answer}. Predicted top 10 tokens {top_k_tokens}.")

                predictions_ = {
                    "ix": i,
                    "question": question,
                    "prompted-question": prompted_question,
                    "gold-answer": answer,
                    "gold-answer-ix": answer_ix,
                    "generation": top_k_tokens[0],      # We can view the top token as the 1-step generation
                    "correct": is_correct,
                    "true_logprob": true_logprob,
                    "false_logprob": false_logprob,
                    "top_1_acc": top_1_acc,
                    "top_5_acc": top_5_acc,
                    "top_10_acc": top_10_acc,
                    "top_10_logprob": top_k_logprob,
                    "top_10_tokens": top_k_tokens,
                    "f1_score": None,
                    "precision": None,
                    "recall": None,
                    "case-sensitive": self.case_sensitive,        # We ignore case when checking answer
                    "white-space-strip": self.strip,              # We ignore white space when checking answer
                    "total_logprob": total_log_prob,
                    "question_logprob": question_log_prob,
                    "answer_logprob": answer_log_prob,
                    "answer_length": 1,
                    "question_answer_length": inputs.input_ids.shape[1] + 1
                }
                predictions.append(predictions_)
        # Save results and tercminate
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

        # # Save the summary
        # save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"

        # results = self.dataset_metric.agg_to_dict()
        # for k, v in args.__dict__.items():
        #     results["args/%s" % k] = v

        # with open(save_summary_fname, "wb") as f:
        #     pickle.dump(results, f)

        # # Print final numbers and return
        # self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


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
                    default="./results/fever/vicuna_7b_results",
                    help='Directory where the data is')
args = parser.parse_args()


# Step 2: Load model and tokenizer
llm_name = args.llm_name
tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path)
model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,device_map="auto")


# Step 3: Create save directory and logger
home_dir = args.home_dir

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

# # Step 5: Read the dataset
dataset_util = FEVER()
dataset = dataset_util.get_dataset(logger)

#Step 6: Run intervention
acc, log_loss = experiment.intervene(model=model,
                        tokenizer=tokenizer,
                        dataset=dataset,
                        args=args)
