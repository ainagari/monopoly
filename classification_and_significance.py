
import numpy as np
import random
import pickle
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
import pdb
from matplotlib import pyplot as plt
import math
import gzip
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
import argparse
import os
from pos_freq_analysis import load_freq_counts
import sys

random.seed(9)


def load_data(control_types, language):
	stats_results = dict()
	classif_results = dict()
	filename_dict = dict()	
	folder = "Similarities/"
	subfolder_models = [f for f in os.listdir(folder) if f.startswith(language) and not "multiuncased" in f]

	for subfolder_model in subfolder_models:
		model_name = subfolder_model.split("-")[1]
		filename_dict[model_name] = dict()
		filename_dict[model_name]["polysemy_bands"] = dict()
		stats_results[model_name] = dict()
		classif_results[model_name] = dict()	
		if language == "en":
			corpus = "semcor"
		else:
			corpus = "eurosense"

		#################### CHECK THIS LINE
		path = 	folder + subfolder_model + "/"
		
		filename_dict[model_name]["mono_poly"] = path + "similarities_mono_poly_"+corpus+".pkl"
		for control_type in control_types:
			filename_dict[model_name]["polysemy_bands"][control_type] = path + "similarities_polysemy_bands_"+corpus+"_" + control_type + ".pkl"
		
	return filename_dict, stats_results, classif_results


def read_files(filename_dict, model_name, mode):
	data = dict()	
	data["mono_poly"] = pickle.load(open(filename_dict[model_name]["mono_poly"], "rb"))
	data["polysemy_poly-bal"] = pickle.load(open(filename_dict[model_name]["polysemy_bands"]["poly-bal"], "rb"))
	if mode == "significance":
		data["polysemy_poly-rand"] = pickle.load(open(filename_dict[model_name]["polysemy_bands"]["poly-rand"], "rb"))
		data["polysemy_poly-same"] = pickle.load(open(filename_dict[model_name]["polysemy_bands"]["poly-same"], "rb"))
	return data

############################################################################
####################  1: STATISTICAL TESTS FUNCTIONS  ######################
############################################################################

def organize_by_dataset_and_class(data):
	values_by_dataset_and_clas = dict()
	for ds in data:
		values_by_dataset_and_clas[ds] = dict()
		for clas in data[ds]:
			values_by_dataset_and_clas[ds][clas] = dict()
			for laynum in data[ds][clas]:
				values_by_dataset_and_clas[ds][clas][laynum] = []
				for tw in data[ds][clas][laynum]:
					values_by_dataset_and_clas[ds][clas][laynum].append(np.average(data[ds][clas][laynum][tw]))

	return values_by_dataset_and_clas

def run_comparison_tests(values_by_dataset_and_clas, stats_results):
	# comparisons between datasets
	for ds in values_by_dataset_and_clas.keys():
		classes = list(values_by_dataset_and_clas[ds].keys())		
		combis = list(combinations(classes, 2)) # make pairs of classes
		stats_results[vector][ds] = dict()
		for laynum in values_by_dataset_and_clas[ds][classes[0]].keys():			
			for clas1, clas2 in combis:
				if (clas1, clas2) not in stats_results[vector][ds].keys():
					stats_results[vector][ds][(clas1, clas2)] = dict()
				# check for normality
				_, shapiro_p1 = shapiro(values_by_dataset_and_clas[ds][clas1][laynum]) 
				_, shapiro_p2 = shapiro(values_by_dataset_and_clas[ds][clas2][laynum])
				# If not normal, do mannwhitney instead of ttest
				if shapiro_p1 < 0.05 or shapiro_p2 < 0.05:
					res, p = mannwhitneyu(values_by_dataset_and_clas[ds][clas1][laynum], values_by_dataset_and_clas[ds][clas2][laynum])
				else:
					res, p = ttest_ind(values_by_dataset_and_clas[ds][clas1][laynum], values_by_dataset_and_clas[ds][clas2][laynum])
				stats_results[vector][ds][(clas1,clas2)][laynum] = (res, p, np.average(values_by_dataset_and_clas[ds][clas1][laynum]) - np.average(values_by_dataset_and_clas[ds][clas2][laynum]))
	# comparison between mono and poly bands
	clas1 = "monosemous"
	for ds in values_by_dataset_and_clas:
		if ds != "mono_poly":
			classes = list(values_by_dataset_and_clas[ds].keys())					
			for laynum in values_by_dataset_and_clas[ds][classes[0]].keys():				
				clas1_data = values_by_dataset_and_clas["mono_poly"]["monosemous"][laynum]
				for clas2 in classes:
					if ("monosemous", clas2) not in stats_results[vector][ds].keys():
						stats_results[vector][ds][("monosemous", clas2)] = dict()								
					_, shapiro_p1 = shapiro(values_by_dataset_and_clas["mono_poly"][clas1][laynum])
					_, shapiro_p2 = shapiro(values_by_dataset_and_clas[ds][clas2][laynum]) 
					if shapiro_p1 < 0.05 or shapiro_p2 < 0.05:
						res, p = mannwhitneyu(clas1_data, values_by_dataset_and_clas[ds][clas2][laynum])
					else:
						#print('ttest:', ds, clas1, clas2, laynum, vector)
						res, p = ttest_ind(clas1_data, values_by_dataset_and_clas[ds][clas2][laynum])
					stats_results[vector][ds][("monosemous",clas2)][laynum] = (res, p, np.average(clas1_data) - np.average(values_by_dataset_and_clas[ds][clas2][laynum]))

	return stats_results


def print_all_significance_results(stats_results, vector):
	print("*****" + vector + "*****")
	for ds in stats_results[vector].keys():
		print("---" + ds + "---")
		for clas in stats_results[vector][ds].keys():
			for laynum in stats_results[vector][ds][clas].keys():
				if "poly-rand" in ds or "poly-rand" in clas:
					print(clas, laynum, stats_results[vector][ds][clas][laynum][1])


def print_fdrcorrected_relevant_results(stats_results, vector):
	interesting = [("monosemous", "poly-rand"), ("monosemous", "low"), ("low","mid"), ("mid","high")]
	# one correction per vector type
	stored_pvals = []
	infos = []
	print("*****" + vector + "*****")
	for ds in stats_results[vector].keys():			
		if ds in ["mono_poly", "polysemy_poly-rand"]:
			for clas in stats_results[vector][ds].keys():
				if clas in interesting:
					for laynum in stats_results[vector][ds][clas].keys():
						stored_pvals.append(stats_results[vector][ds][clas][laynum][1])
						infos.append((clas, laynum))

	rejection, corrected_pvals, _, _ = multipletests(stored_pvals, method="fdr_bh",alpha=alpha_value)
	for info, ori_pval, rej, cor_pval in zip(infos, stored_pvals, rejection, corrected_pvals):
		print(info[0], info[1], ori_pval, rej, cor_pval)



##############################################################################
#########################  2: CLASSIFIER FUNCTIONS ###########################
##############################################################################



def split_traindevtest_targetwords():
	ref = list(filename_dict.keys())[0] if list(filename_dict.keys())[0] != "elmo" else list(filename_dict.keys())[1]
	data = dict()
	data["mono_poly"] = pickle.load(open(filename_dict[ref]["mono_poly"], "rb"))
	data["polysemy_poly-bal"] = pickle.load(open(filename_dict[ref]["polysemy_bands"]["poly-bal"], "rb"))

	words_by_set = dict()
	words_by_set['train'] = dict()
	words_by_set['test'] = dict()
	words_by_set['dev'] = dict()


	for clas in ['monosemous', 'poly-rand']:
		target_words = list(data["mono_poly"][clas][0].keys())
		num_train_words = int(len(target_words)* 0.7)
		if (len(target_words) - num_train_words) % 2 != 0:
			num_train_words +=1
		num_devtest_words = (len(target_words) - num_train_words)//2

		train_words = target_words[:num_train_words]
		dev_words = target_words[num_train_words:num_train_words+num_devtest_words]
		test_words = target_words[num_train_words+num_devtest_words:]
		words_by_set['train'][clas] = train_words
		words_by_set['dev'][clas] = dev_words
		words_by_set['test'][clas] = test_words
		if clas == "poly-rand":
			for clas2 in ["poly-bal", "poly-same"]:
				words_by_set['train'][clas2] = train_words
				words_by_set['dev'][clas2] = dev_words
				words_by_set['test'][clas2] = test_words
		if clas == "monosemous": # "monosemous_poly" is for the multiclass classifier:
			words_by_set['train']["monosemous_poly"] = train_words
			words_by_set['dev']["monosemous_poly"] = dev_words
			words_by_set['test']["monosemous_poly"] = test_words


	# MULTICLASS (4 classes: mono, low, mid, high)
	# set the number of words per band
	num_words_per_band = []
	for clas in data["polysemy_poly-bal"]:
		num_words_per_band.append(len(data["polysemy_poly-bal"][clas][0]))
	num_words_per_band.append(len(data["mono_poly"]["monosemous"][0]))
	min_words_per_band_multiclass = min(num_words_per_band)

	num_train_words = int(min_words_per_band_multiclass* 0.7) # number of training words *per band*
	if (min_words_per_band_multiclass - num_train_words) % 2 != 0:
		num_train_words +=1
	num_devtest_words = (min_words_per_band_multiclass - num_train_words)//2

	for clas in data["polysemy_poly-bal"]:
		pdb.set_trace()
		target_words = list(data["polysemy_poly-bal"][clas][0].keys())[:min_words_per_band_multiclass]
		train_words = target_words[:num_train_words]
		dev_words = target_words[num_train_words:num_train_words+num_devtest_words]
		test_words = target_words[num_train_words+num_devtest_words:]
		words_by_set['train'][clas] = train_words
		words_by_set['dev'][clas] = dev_words
		words_by_set['test'][clas] = test_words

	pdb.set_trace()
	words_by_set['train']["monosemous_poly"] = words_by_set['train']["monosemous_poly"][:num_train_words]
	words_by_set['dev']["monosemous_poly"] = words_by_set['dev']["monosemous_poly"][:num_devtest_words]
	words_by_set['test']["monosemous_poly"] = words_by_set['test']["monosemous_poly"][:num_devtest_words]
	pdb.set_trace()

	return words_by_set



def binary_classifier(words_by_set, classif_results, freq_counts=None):

	binary_classification_settings = [("monosemous", "poly-bal"), ("monosemous", "poly-rand"), ("monosemous", "poly-same")]
		
	training_words = []
	dev_words = []
	test_words = []

	for i, clas in enumerate(binary_classification_settings[0]):
		training_words.extend([(x, i) for x in words_by_set["train"][clas]])
		dev_words.extend([(x, i) for x in words_by_set["dev"][clas]])
		test_words.extend([(x, i) for x in words_by_set["test"][clas]])			 
	random.shuffle(training_words)
	random.shuffle(test_words)
	random.shuffle(dev_words)


	for vector in filename_dict.keys():
		data = dict()
		data["mono_poly"] = pickle.load(open(filename_dict[vector]["mono_poly"], "rb"))

		classif_results[vector]["mono_poly"] = dict()
		for classpair in binary_classification_settings:
			classif_results[vector]["mono_poly"][classpair] = dict()
			for laynum in data["mono_poly"][classpair[0]].keys():
				classif_results[vector]["mono_poly"][classpair][laynum] = dict()
				data_for_classifier = dict()
				for xy in ["training_x", "training_y", "dev_x", "dev_y", "test_x","test_y", "training_45feat_x", "dev_45feat_x", "test_45feat_x"]:
					data_for_classifier[xy] = []
				for subset_name, subset in [("training", training_words), ("dev", dev_words), ("test",test_words)]:
					for tw, i  in subset:						
						data_for_classifier[subset_name + "_x"].append(np.average(data["mono_poly"][classpair[i]][laynum][tw]))
						data_for_classifier[subset_name + "_45feat_x"].append(np.array(data["mono_poly"][classpair[i]][laynum][tw]))
						data_for_classifier[subset_name + "_y"].append(i)
					
				for xy in data_for_classifier:
					if xy.endswith("_x") and "45feat" not in xy:
						data_for_classifier[xy] = np.array(data_for_classifier[xy]).reshape(-1,1)
					else:
						data_for_classifier[xy] = np.array(data_for_classifier[xy])
				
				logreg = LogisticRegression(solver="lbfgs")
				logreg45 = LogisticRegression(solver="lbfgs")
				logreg.fit(data_for_classifier["training_x"], data_for_classifier["training_y"])
				logreg45.fit(data_for_classifier["training_45feat_x"], data_for_classifier["training_y"])
				
				predictions_dev = logreg.predict(data_for_classifier["dev_x"])
				predictions_test = logreg.predict(data_for_classifier["test_x"])
				predictions_dev45 = logreg45.predict(data_for_classifier["dev_45feat_x"])
				predictions_test45 = logreg45.predict(data_for_classifier["test_45feat_x"])

				accuracy_dev = accuracy_score(data_for_classifier["dev_y"], predictions_dev)
				accuracy_test = accuracy_score(data_for_classifier["test_y"], predictions_test)
				accuracy_dev45 = accuracy_score(data_for_classifier["dev_y"], predictions_dev45)
				accuracy_test45 = accuracy_score(data_for_classifier["test_y"], predictions_test45)
				
				classif_results[vector]["mono_poly"][classpair][laynum]["onefeat"] = (accuracy_dev, accuracy_test)
				classif_results[vector]["mono_poly"][classpair][laynum]["45feat"] = (accuracy_dev45, accuracy_test45)

	if freq_counts:
		for xy in ["training_x", "training_y", "dev_x", "dev_y", "test_x","test_y"]:
			data_for_classifier[xy] = []
		for subset_name, subset in [("training", training_words), ("dev", dev_words), ("test",test_words)]:
			for tw, i in subset:
				try:
					data_for_classifier[subset_name + "_x"].append(math.log(freq_counts[tw[:-2]]))
				except ValueError:
					data_for_classifier[subset_name + "_x"].append(-1)
				
				data_for_classifier[subset_name + "_y"].append(i)
				
		for xy in data_for_classifier:
			if xy.endswith("_x"):
				data_for_classifier[xy] = np.array(data_for_classifier[xy]).reshape(-1,1)

		logreg = LogisticRegression(solver="lbfgs")
		logreg.fit(data_for_classifier["training_x"], data_for_classifier["training_y"])
		predictions_dev = logreg.predict(data_for_classifier["dev_x"])
		predictions_test = logreg.predict(data_for_classifier["test_x"])
		accuracy_dev = accuracy_score(data_for_classifier["dev_y"], predictions_dev)
		accuracy_test = accuracy_score(data_for_classifier["test_y"], predictions_test)

		print("MONO-POLY-FREQ\nAccuracy dev:", accuracy_dev, "Accuracy test:", accuracy_test)

	return classif_results


def multi_classifier(words_by_set, classif_results, freq_counts=None):
	classset = ("monosemous", "low", "mid", "high")

	training_words = []
	dev_words = []
	test_words = []

	for i, clas in enumerate(classset):
		if clas == "monosemous":
			clas = "monosemous_poly"
		training_words.extend([(x, i) for x in words_by_set["train"][clas]])
		dev_words.extend([(x, i) for x in words_by_set["dev"][clas]])
		test_words.extend([(x, i) for x in words_by_set["test"][clas]])			 
	random.shuffle(training_words)
	random.shuffle(test_words)
	random.shuffle(dev_words)

	for vector in filename_dict.keys():
		data = read_files(filename_dict, vector, mode="significance")
		
		for ds in ["polysemy_poly-bal", "polysemy_poly-rand","polysemy_poly-same"]:
			classif_results[vector][ds] = dict()		
			classif_results[vector][ds][classset] = dict()
			for laynum in data[ds][classset[1]].keys():
				classif_results[vector][ds][classset][laynum] = dict()
				
				data_for_classifier = dict()
				for xy in ["training_x", "training_y", "dev_x", "dev_y", "test_x","test_y", "training_45feat_x", "dev_45feat_x", "test_45feat_x"]:
					data_for_classifier[xy] = []

				for subset_name, subset in [("training", training_words), ("dev", dev_words), ("test",test_words)]:
					for tw, i  in subset:
						if i == 0:
							data_for_classifier[subset_name + "_x"].append(np.average(data["mono_poly"]["monosemous"][laynum][tw]))
							data_for_classifier[subset_name + "_45feat_x"].append(np.array(data["mono_poly"]["monosemous"][laynum][tw]))
						else:						
							data_for_classifier[subset_name + "_x"].append(np.average(data[ds][classset[i]][laynum][tw]))
							data_for_classifier[subset_name + "_45feat_x"].append(np.array(data[ds][classset[i]][laynum][tw]))
											
						data_for_classifier[subset_name + "_y"].append(i)								

				for xy in data_for_classifier:
					if xy.endswith("_x") and "45feat" not in xy:
						data_for_classifier[xy] = np.array(data_for_classifier[xy]).reshape(-1,1)
					else:
						data_for_classifier[xy] = np.array(data_for_classifier[xy])	
			

				logreg = LogisticRegression(solver="lbfgs",multi_class="multinomial")
				logreg45 = LogisticRegression(solver="lbfgs",multi_class="multinomial")
				logreg.fit(data_for_classifier["training_x"], data_for_classifier["training_y"])
				logreg45.fit(data_for_classifier["training_45feat_x"], data_for_classifier["training_y"])
				
				predictions_dev = logreg.predict(data_for_classifier["dev_x"])
				predictions_test = logreg.predict(data_for_classifier["test_x"])
				predictions_dev45 = logreg45.predict(data_for_classifier["dev_45feat_x"])
				predictions_test45 = logreg45.predict(data_for_classifier["test_45feat_x"])

				accuracy_dev = accuracy_score(data_for_classifier["dev_y"], predictions_dev)
				accuracy_test = accuracy_score(data_for_classifier["test_y"], predictions_test)
				accuracy_dev45 = accuracy_score(data_for_classifier["dev_y"], predictions_dev45)
				accuracy_test45 = accuracy_score(data_for_classifier["test_y"], predictions_test45)

				classif_results[vector][ds][classset][laynum]["onefeat"] = (accuracy_dev, accuracy_test)
				classif_results[vector][ds][classset][laynum]["45feat"] = (accuracy_dev45, accuracy_test45)

	# frequency-based classifier
	if freq_counts:
		data_for_classifier = dict()
		for xy in ["training_x", "training_y", "dev_x", "dev_y", "test_x","test_y"]:
			data_for_classifier[xy] = []
		for subset_name, subset in [("training", training_words), ("dev", dev_words), ("test",test_words)]:
			for tw, i in subset:
				try:
					data_for_classifier[subset_name + "_x"].append(math.log(freq_counts[tw[:-2]]))
				except ValueError:
					data_for_classifier[subset_name + "_x"].append(-1)
				data_for_classifier[subset_name + "_y"].append(i)
								

		for xy in data_for_classifier:
			if xy.endswith("_x"):
				data_for_classifier[xy] = np.array(data_for_classifier[xy]).reshape(-1,1)
			
		logreg = LogisticRegression(solver="lbfgs",multi_class="multinomial")
		logreg.fit(data_for_classifier["training_x"], data_for_classifier["training_y"])

			
		predictions_dev = logreg.predict(data_for_classifier["dev_x"])
		predictions_test = logreg.predict(data_for_classifier["test_x"])
		accuracy_dev = accuracy_score(data_for_classifier["dev_y"], predictions_dev)
		accuracy_test = accuracy_score(data_for_classifier["test_y"], predictions_test)

		print("MULTI CLASSIF FREQ\nAccuracy dev:", accuracy_dev, "Accuracy test:", accuracy_test)


	return classif_results




def print_classification_results(classif_results):
	for vector in classif_results:
		print("-_-_-_- "+ vector.upper() + "-_-_-_-")
		for ds in classif_results[vector]:
			print("*****" + ds + "*****")
			for cla in classif_results[vector][ds].keys():
				for laynum in classif_results[vector][ds][cla].keys():
					print(cla, laynum, classif_results[vector][ds][cla][laynum])



def print_onlybest_classification_results(classif_results):
	# print the best result per vector/ feature type) according to results on dev set
	for vector in classif_results:
		print("-_-_-_- " + vector.upper() + "-_-_-_-")
		for ds in classif_results[vector]:
			for cla in classif_results[vector][ds].keys():
				if (ds == "mono_poly" and cla == ('monosemous', 'poly-rand')) or ds == "polysemy_poly-rand":
					print("*****" + ds + "*****")
					bestdev1feat = 0
					bestdev45feat = 0
					for laynum in classif_results[vector][ds][cla].keys():
						if classif_results[vector][ds][cla][laynum]['onefeat'][0] > bestdev1feat:
							bestdev1feat = classif_results[vector][ds][cla][laynum]['onefeat'][0]
						if classif_results[vector][ds][cla][laynum]['45feat'][0] > bestdev45feat:
							bestdev45feat = classif_results[vector][ds][cla][laynum]['45feat'][0]
					for laynum in classif_results[vector][ds][cla].keys():
						if classif_results[vector][ds][cla][laynum]['onefeat'][0] == bestdev1feat:
							print("SELFSIM ---- classes:", cla, "layer:", laynum, "test accuracy:", classif_results[vector][ds][cla][laynum]['onefeat'][1]) # test result
						if classif_results[vector][ds][cla][laynum]['45feat'][0] == bestdev45feat:
							print("PAIRCOS ---- classes:", cla, "layer:", laynum, "test accuracy:", classif_results[vector][ds][cla][laynum]['45feat'][1]) # test result


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--language", default='en', type=str, help="en,fr,es,el")
	parser.add_argument("--multilingual", action="store_true", help="whether we use the multilingual model or not")
	parser.add_argument("--cased", action="store_true", help="whether we use the cased model or not")
	parser.add_argument("--mode", default='classification',type=str, help="'significance' or 'classification'")
	parser.add_argument("--do_freq", action="store_true", help="whether we run the freq classifier")
	parser.add_argument("--english_freq_fn", default="", type=str, help="if do_freq is used and the language is English, you need to provide the path to the Google Ngrams frequency file")
	parser.add_argument("--alpha", default=0.01, type=float, help="alpha value for ttests/mannwhitney tests. Default: 0.01")

	args = parser.parse_args()

	if args.do_freq and args.language == "en" and not args.english_freq_fn:
		sys.out("Provide a path to a file containing Google Ngrams frequencies in English with --english_freq_fn")

	dataset = ["mono_poly", "polysemy_bands"]
	control_types = ["poly-bal", "poly-rand", "poly-same"]
	filename_dict, stats_results, classif_results = load_data(control_types, language=args.language)


	########################## STATISTICAL TESTS ############################
	if args.mode == "significance":
		for vector in filename_dict.keys():
			# load datasets for this vector type
			data = read_files(filename_dict, vector, mode="significance")
			values_by_dataset_and_clas = organize_by_dataset_and_class(data)
			# run the tests
			stats_results = run_comparison_tests(values_by_dataset_and_clas, stats_results)
			print_fdrcorrected_relevant_results(stats_results, vector)

	########################## CLASSIFIERS ############################
	elif args.mode == "classification":
		# prepare training data
		words_by_set = split_traindevtest_targetwords()
		if args.do_freq:
			freq_counts = load_freq_counts(args.language, args.english_freq_fn)
		else:
			freq_counts = None

		# classifiers: binary and multiclass
		classif_results = binary_classifier(words_by_set, classif_results, freq_counts=freq_counts)
		classif_results = multi_classifier(words_by_set, classif_results, freq_counts=freq_counts)

		print_onlybest_classification_results(classif_results)
		#print_classification_results(classif_results)