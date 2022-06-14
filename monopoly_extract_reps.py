'''
This script serves to extract representations from BERT models and calculate self-similarities
See run_monopoly.sh
'''

from sklearn.metrics.pairwise import pairwise_distances
import pdb
from torch.utils.data.dataset import Dataset
from transformers import BertModel, BertTokenizer, BertConfig, FlaubertConfig, FlaubertModel, FlaubertTokenizer, AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, SequentialSampler
import torch
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from scipy.spatial.distance import cosine
import pandas
from matplotlib import pyplot as plt
import pickle
import sys
import random
from copy import deepcopy
import argparse


def get_data_and_outfns(dataset, control_type, language, model_name):
    folder = "Data/"
    if language == "en":
        corpus = 'semcor'        
    else:
        corpus = 'eurosense'
    langstr = "_" + language

    modelnamestr = dict()
    modelnamestr["bert-base-uncased"] = "bert"
    modelnamestr["bert-base-cased"] = "bbcased"
    modelnamestr["bert-base-multilingual-cased"] = "multicased"
    modelnamestr["nlpaueb/bert-base-greek-uncased-v1"] = "greekuncased"
    modelnamestr["dccuchile/bert-base-spanish-wwm-uncased"] = "betouncased"
    modelnamestr["flaubert/flaubert_base_uncased"] = "flaubertuncased"

    mm = modelnamestr[model_name]
    out_folder = folder + language + mm + "/"

    if dataset == "mono_poly":
        control_type = ""
    else:
        control_type = "_" + control_type

    in_fn = dataset + "_" + corpus + "_experiments_data_" + langstr + ".pkl"
    out_reps_fn = dataset + "_" + corpus + "_representations" + control_type + ".pkl"
    out_sims_fn = "similarities_" + dataset + "_" + corpus + control_type + ".pkl"
    out_csv = dataset + "_" + corpus + control_type + ".csv"

    data = pickle.load(open(folder + in_fn, "rb"))
    if dataset == "polysemy_bands":        
        data = data[control_type]

    return data, out_folder + out_reps_fn, out_folder + out_sims_fn, out_folder + out_csv


def check_correct_token_mapping(bert_tokenized_sentence, positions, word, tokenizer):
    # put together the pieces corresponding to the positions
    tokenized_word = list(tokenizer.tokenize(word))
    # check if they make the word we want
    berttoken = []
    for p in positions:
        berttoken.append(bert_tokenized_sentence[p])

    if berttoken == tokenized_word:
        return True
    else:
        return False



def aggregate_reps(reps_list):
    reps = torch.zeros([len(reps_list), 768])
    for i, wrep in enumerate(reps_list):
        w, rep = wrep
        reps[i] = rep
    # if a word is split into multiple wordpieces, average their representations
    if len(reps) > 1:
        reps = torch.mean(reps, dim=0)
    try:
        reps = reps.view(768)
    except RuntimeError:
        pdb.set_trace()
    return reps

def careful_tokenization(sentence, tokenizer, model_name, maxlen):
    map_ori_to_bert = []
    # initialise the tokenised sentence
    if "flaubert" in model_name:
        tok_sent = ['<s>']
    else:
        tok_sent = ['[CLS]']

    for orig_token in sentence.split():
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) # tokenize
        ##### check if adding this token will result in >= maxlen (=, because [SEP] goes at the end). If so, stop
        if len(tok_sent) + len(bert_token) >= maxlen:
            break
        tok_sent.extend(bert_token) # append the new token(s) to the partial tokenised sentence
        if len(bert_token) > 1: # if the word has been split into wordpieces
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1) # list of new positions of the target word in the new tokenisation
        map_ori_to_bert.append(tuple(current_tokens_bert_idx))

    if "flaubert" in model_name:
        tok_sent.append('</s>')
    else:
        tok_sent.append('[SEP]')

    return tok_sent, map_ori_to_bert


def extract_representations(infos, tokenizer, model_name, maxlen):
    reps = []
    if model_name in ["bert-base-uncased", "bert-base-cased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased"]:
        config_class, model_class = BertConfig, BertModel
    elif "flaubert" in model_name:
        config_class, model_class = FlaubertConfig, FlaubertModel
    elif "greek" in model_name or "spanish" in model_name:
        config_class, model_class = AutoConfig, AutoModel

    config = config_class.from_pretrained(model_name, output_hidden_states=True,max_position_embeddings=maxlen)
    model = model_class.from_pretrained(model_name, config=config)

    model.eval()
    with torch.no_grad():
        for info in infos:
            tok_sent = info['bert_tokenized_sentence']
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_sent)]).to(device)
            inputs = {'input_ids': input_ids}
            outputs = model(**inputs)

            if "flaubert" in model_name:
                hidden_states = outputs[1]
            else:
                hidden_states = outputs[2]

            bpositions = info["bert_position"]

            reps_for_this_instance = dict()
            for i, w in enumerate(info["bert_tokenized_sentence"]):
                if i in bpositions: #if it's one of the relevant positions where the target word is
                    for l in range(len(hidden_states)): # all layers
                        if l not in reps_for_this_instance:
                            reps_for_this_instance[l] = []

                        reps_for_this_instance[l].append((w, hidden_states[l][0][i].cpu()))

            reps.append(reps_for_this_instance)            


    return reps


if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True, help="mono_poly or polysemy_bands")
    parser.add_argument("--control_type", default=None, type=str, help="poly-bal, poly-rand, poly-same")
    parser.add_argument("--language", default='en', type=str, help="en, fr, es, el")
    parser.add_argument("--multilingual", action="store_true", help="whether we use the multilingual model or not")
    parser.add_argument("--cased", action="store_true", help="whether we use a cased model or not")

    args = parser.parse_args()
    dataset = args.dataset
    control_type = args.control_type
    language = args.language

    if args.cased:
        do_lower_case=False
    else:
        do_lower_case=True

    if args.multilingual:
        if args.cased:
            model_name = "bert-base-multilingual-cased"
        else:
            model_name = "bert-base-multilingual-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
    else:
        if language == "en":
       	    if args.cased:
       	       	model_name = "bert-base-cased"
       	    else:
       	       	model_name = "bert-base-uncased"
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)
       	elif language == "el":
            if args.cased:
                print("model not available")
            else:
                model_name = "nlpaueb/bert-base-greek-uncased-v1"
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

        elif language == "fr":
            if args.cased:
                print("model not available")
            else:
                model_name = "flaubert/flaubert_base_uncased"
            tokenizer = FlaubertTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

        elif language == "es":
            if args.cased:
                print("model not available")
            else:
                model_name = "dccuchile/bert-base-spanish-wwm-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1
    toplayer = 13
    maxlen = 512

   data, out_reps_fn, out_sims_fn, out_csv = get_data_and_outfns(dataset, control_type, language, model_name)


    infos = []
    for clas in data:
        for targetword in data[clas]:
            for i, instance in enumerate(data[clas][targetword]):
                info = dict()
                info["target_word"] = targetword
                info["class"] = clas
                info["index"] = i
                info["position"] = instance["position"]
                info["word"] = instance["word"]
                info["sentence"] = instance["sentence"]

                bert_tokenized_sentence, mapp = careful_tokenization(info["sentence"], tokenizer, model_name,maxlen=maxlen)
                info["bert_tokenized_sentence"] = bert_tokenized_sentence
                try:
                    bert_position = mapp[info["position"]]
                except IndexError:
                    pdb.set_trace()
                info["bert_position"] = bert_position
                if not check_correct_token_mapping(bert_tokenized_sentence, bert_position, info["word"], tokenizer):
                    print("Position mismatch!")
                    pdb.set_trace()
                infos.append(info)


    print("EXTRACTING REPRESENTATIONS...")
    reps = extract_representations(infos, tokenizer, model_name,maxlen=maxlen)
    print("...DONE")

    if len(reps) != len(infos):
         print("Serious mismatch")
         pdb.set_trace()
    # get representations of the relevant words using their position and complete the data dict with them
    for rep, instance in zip(reps, infos):
        for laynum in rep:
            k = "rep-"+str(laynum)
            representation = aggregate_reps(rep[laynum])
            try:
                data[instance['class']][instance['target_word']][instance['index']][k] = representation.cpu()
            except KeyError:
                pdb.set_trace()

    ######## store representations
    pickle.dump(data, open(out_reps_fn, "wb"))


    ######## extract similarities


    cosines_with_words = dict() # by word (for the classifier)
    cosines = dict() # by class
    for clas in data:
        cosines[clas] = dict()
        cosines_with_words[clas] = dict()

    for clas in data:
        for tw in data[clas]:
            for laynum in range(0, toplayer):
                k = "rep-" + str(laynum)
                reps_together = np.array([np.array(data[clas][tw][ii][k].cpu()) for ii in range(len(data[clas][tw]))])
                cosinedist_list = pairwise_distances(reps_together, metric='cosine')[np.triu_indices(reps_together.shape[0], k=1)] # calculate cosine distances and take upper triangular
                cosine_list = [1 - c for c in cosinedist_list] # turn cosine distances into similarities
                if laynum not in cosines[clas]:
                    cosines[clas][laynum] = []
                cosines[clas][laynum].append(np.average(cosine_list))
                if laynum not in cosines_with_words[clas]:
                    cosines_with_words[clas][laynum] = dict()
                cosines_with_words[clas][laynum][tw] = cosine_list

    for_dataframe = dict()

    for clas in cosines:
        for_dataframe[clas] = dict()
        for laynum in cosines[clas]:
            average_cosine = np.average(cosines[clas][laynum])
            for_dataframe[clas][laynum] = average_cosine

    df = pandas.DataFrame.from_dict(for_dataframe, orient='index')

    pickle.dump(cosines_with_words, open(out_sims_fn, "wb"))
    df.to_csv(out_csv, sep="\t", index_label="Type")
