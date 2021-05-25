import pickle
from collections import defaultdict, Counter
import gzip
import pandas as pd
import numpy as np
import pdb
import sys

def load_data(language):
    if language == "en":
        corpus = "semcor"
    else:
        corpus = "eurosense"
    data = dict()

    monopoly_fn = "similarities_mono_poly_" + corpus + ".pkl"
    polybands_fn = "similarities_polysemy_bands_" + corpus + "_poly-rand.pkl"
    folder = "Similarities/"

    monovecnames = {"es":["betouncased"], "fr": ["flaubertuncased"], "el": ["greekuncased"], "en": ["bert","elmo","c2v"]}
    folders2 = []
    for mvn in monovecnames[language]:
        folders2.append(language + "-" + mvn)
    folders2.append(language + "-multicased")

    for folder2 in folders2:
        full_folder = folder + folder2 + "/"
        data[folder2] = dict()

        final_monopoly_fn = full_folder + monopoly_fn
        final_polybands_fn = full_folder + polybands_fn

        data[folder2]['monopoly'] = pickle.load(open(final_monopoly_fn, "rb"))
        data[folder2]['polybands'] = pickle.load(open(final_polybands_fn, "rb"))

    word_dict = dict()

    word_dict['mono'] = data[folder2]["monopoly"]['monosemous'][1]  # 1 is the layer, but it's irrelevant
    word_dict['poly'] = data[folder2]["monopoly"]['poly-rand'][1]
    word_dict['low'] = data[folder2]["polybands"]['low'][1]
    word_dict['mid'] = data[folder2]["polybands"]['mid'][1]
    word_dict['high'] = data[folder2]["polybands"]['high'][1]

    return data, word_dict


def load_freq_counts(language, english_freq_fn):
    if language in ["es","el","fr"]:
        frequency_fn = "freq_counts_"+language+ ".pkl"
        freq_counts = pickle.load(open(frequency_fn, "rb"))
    elif language == "en":
        # Load google unigrams
        with gzip.open(english_freq_fn) as f:
                bytecontents = f.read()
        contents = bytecontents.decode("utf-8")
        contents = contents.split("\n")
        freq_counts = defaultdict(int) 

        for tokencount in contents:
                s = tokencount.strip().split("\t")
                if len(s) == 2:
                        token, count = s
                        freq_counts[token] = int(count)

    return freq_counts


def posdist_overbands(word_dict):
    # Bands: mono, poly, low, mid, high
    pos_by_band = dict()
    band_by_pos = dict()
    pos_by_word = dict()
    for wordtype in word_dict:
        # get all observed parts of speech
        observed_pos = [w.split("_")[-1] for w in word_dict[wordtype]]
        count_pos = Counter(observed_pos)
        pos_by_band[wordtype] = count_pos
        for pos in count_pos:
            if pos not in band_by_pos:
                band_by_pos[pos] = dict()
            band_by_pos[pos][wordtype] = count_pos[pos]
        for wordpos in word_dict[wordtype]:
            pos_by_word[wordpos] = wordpos.split("_")[-1]

    # proportion
    pos_by_band_props = dict()
    for band in pos_by_band:
        pos_by_band_props[band] = dict()
        total = sum(pos_by_band[band].values())
        for pos in pos_by_band[band]:
            pos_by_band_props[band][pos] = pos_by_band[band][pos] / total

    return pos_by_word, pos_by_band, pos_by_band_props


def assign_to_freq_band(freq, quartiles):
    for i, quart in enumerate(quartiles):
        if i == 0:
            if freq <= quart:
                band = i
        else:
            if freq <= quart and freq >quartiles[i - 1]:
                band = i
    return band


def freqdist_overbands(word_dict, freq_counts):
    # Get frequencies observed in the data
    freqband_by_word_4 = dict()  # 4 frequency bands (4 quartiles)
    freqs_by_word = dict()
    for band in word_dict:
        for wordpos in word_dict[band]:
            word = wordpos.split("_")[0]
            freqs_by_word[wordpos] = freq_counts[word]
            if len(wordpos.split("_")) > 2:
                print(wordpos)

    freqs_df = pd.DataFrame(freqs_by_word.values())

    four_quartiles = freqs_df.quantile([0.25, 0.5, 0.75, 1]).values
    print("the four quartiles:", four_quartiles)
    freq_by_band_4 = dict()

    for wordtype in word_dict:
        freq_by_band_4[wordtype] = defaultdict(int)

        for wordpos in word_dict[wordtype]:
            word = wordpos.split("_")[0]
            freq = freq_counts[word]
            # situate this frequency in one of the freq bands
            try:
                band4 = assign_to_freq_band(freq, four_quartiles)
            except UnboundLocalError:
                pdb.set_trace()
            freq_by_band_4[wordtype][band4] += 1
            freqband_by_word_4[wordpos] = band4

    freq_by_band_4_props = dict() # proportion of words in each band by frequency range
    for band in freq_by_band_4:
        freq_by_band_4_props[band] = dict()
        total = sum(freq_by_band_4[band].values())
        for f in freq_by_band_4[band]:
            freq_by_band_4_props[band][f] = freq_by_band_4[band][f] / total

    return freqband_by_word_4, freq_by_band_4, freq_by_band_4_props


def sim_by_pos(pos_by_word, simlayer, vector):
    similarities_by_layer_and_pos = dict()

    for wordpos in pos_by_word:
        pos = pos_by_word[wordpos]
        if pos not in similarities_by_layer_and_pos:
            similarities_by_layer_and_pos[pos] = dict()

        # find the band that contains this word
        for b in ['mono', 'poly', 'low', 'mid', 'high']:
            if wordpos in simlayer[b][auxlayer]:
                band = b
                break
        for layer in simlayer[band]:
            if layer not in similarities_by_layer_and_pos[pos]:
                similarities_by_layer_and_pos[pos][layer] = []
            selfsim = np.average(simlayer[band][layer][wordpos])
            similarities_by_layer_and_pos[pos][layer].append(selfsim)

    avgsimilarities_by_layer_and_pos = dict()

    for pos in similarities_by_layer_and_pos:
        avgsimilarities_by_layer_and_pos[pos] = dict()
        for layer in similarities_by_layer_and_pos[pos]:
            avgsimilarities_by_layer_and_pos[pos][layer] = np.average(similarities_by_layer_and_pos[pos][layer])

    posdf = pd.DataFrame.from_dict(avgsimilarities_by_layer_and_pos)

    posdf.to_csv(args.out_dir + "/" + vector + "/pos_by_sim.csv", sep="\t")

    return posdf


def get_posbalanced_bands(pos_by_band, simlayer):
    allpos = set()
    for band in pos_by_band:
        for pos in pos_by_band[band]:
            allpos.add(pos)

    balanced_words = dict()
    for pos in allpos:
        values = [pos_by_band[band][pos] for band in pos_by_band]
        minvalue = min(values)

        for band in ['mono', 'poly', 'low', 'mid', 'high']:
            if band not in balanced_words:
                balanced_words[band] = dict()
            if pos not in balanced_words[band]:
                balanced_words[band][pos] = []
            for wordpos in simlayer[band][auxlayer]: # look for the band with the smallest number of words of this pos
                pos2 = pos_by_word[wordpos]
                if pos2 == pos and len(balanced_words[band][pos]) < minvalue:
                    balanced_words[band][pos].append(wordpos)
                if len(balanced_words[band][pos]) >= minvalue:
                    break
            if len(balanced_words[band][pos]) > minvalue:
                print("SOMETHING'S WRONG")
                pdb.set_trace()

    balanced_by_band = dict()
    for band in balanced_words:
        balanced_by_band[band] = []
        for pos in balanced_words[band]:
            balanced_by_band[band].extend(balanced_words[band][pos])

    print("number of words left in each band after balancing for pos:")
    for band in balanced_by_band:
        print(band, len(balanced_by_band[band]))
    print("number of words per band and pos after balancing:")
    for band in balanced_words:
        print(band, [(pos, len(balanced_words[band][pos])) for pos in balanced_words[band]])


    return balanced_by_band


def get_posbalanced_similarities(balanced_by_band, simlayer, vector):
    posbalanced_similarities_by_layer_and_poly = dict()
    posbalanced_similarities_by_layer_and_poly_by_word = dict()

    for band in ['mono', 'poly', 'low', 'mid', 'high']:
        if band not in posbalanced_similarities_by_layer_and_poly:
            posbalanced_similarities_by_layer_and_poly[band] = dict()
            posbalanced_similarities_by_layer_and_poly_by_word[band] = dict()

        for layer in simlayer[band]:
            if layer not in posbalanced_similarities_by_layer_and_poly[band]:
                posbalanced_similarities_by_layer_and_poly[band][layer] = []
            if layer not in posbalanced_similarities_by_layer_and_poly_by_word[band]:
                posbalanced_similarities_by_layer_and_poly_by_word[band][layer] = dict()

            for wordpos in balanced_by_band[band]:
                posbalanced_similarities_by_layer_and_poly_by_word[band][layer][wordpos] = simlayer[band][layer][wordpos]
                selfsim = np.average(simlayer[band][layer][wordpos])
                posbalanced_similarities_by_layer_and_poly[band][layer].append(selfsim)

    pickle.dump(posbalanced_similarities_by_layer_and_poly_by_word, open(args.out_dir + "/" + vector + "/similarities_posbalanced.pkl", "wb"))
    avg_posbalanced = dict()

    for polyband in posbalanced_similarities_by_layer_and_poly:
        avg_posbalanced[polyband] = dict()
        for layer in posbalanced_similarities_by_layer_and_poly[polyband]:
            avg_posbalanced[polyband][layer] = np.average(posbalanced_similarities_by_layer_and_poly[polyband][layer])

    balposdf = pd.DataFrame.from_dict(avg_posbalanced)

    balposdf.to_csv(args.out_dir + "/" + vector + "/poly_pos-bal.csv", sep="\t")

    return balposdf


def sim_by_freq(freqband_by_word_4, simlayer, vector):
    similarities_by_layer_and_freq = dict()

    for wordpos in freqband_by_word_4:
        freqband = freqband_by_word_4[wordpos]
        if freqband not in similarities_by_layer_and_freq:
            similarities_by_layer_and_freq[freqband] = dict()

        for b in ['mono', 'poly', 'low', 'mid', 'high']:
            if wordpos in simlayer[b][auxlayer]:
                band = b
                break
        for layer in simlayer[band]:
            if layer not in similarities_by_layer_and_freq[freqband]:
                similarities_by_layer_and_freq[freqband][layer] = []
            selfsim = np.average(simlayer[band][layer][wordpos])
            similarities_by_layer_and_freq[freqband][layer].append(selfsim)

    avgsimilarities_by_layer_and_freq = dict()

    for freqband in similarities_by_layer_and_freq:
        avgsimilarities_by_layer_and_freq[freqband] = dict()
        for layer in similarities_by_layer_and_freq[freqband]:
            avgsimilarities_by_layer_and_freq[freqband][layer] = np.average(
                similarities_by_layer_and_freq[freqband][layer])

    freqdf = pd.DataFrame.from_dict(avgsimilarities_by_layer_and_freq)
    freqdf.to_csv(args.out_dir + "/" + vector + "/freq_by_sim.csv", sep="\t")

    return freqdf



def get_freqbalanced_bands(freq_by_band_4, simlayer):
    allranges = set()
    for band in freq_by_band_4:
        for range in freq_by_band_4[band]:
            allranges.add(range)

    balanced_words = dict()
    for range in allranges:
        values = [freq_by_band_4[band][range] for band in freq_by_band_4]
        minvalue = min(values)

        for band in ['mono', 'poly', 'low', 'mid', 'high']:
            if band not in balanced_words:
                balanced_words[band] = dict()
            if range not in balanced_words[band]:
                balanced_words[band][range] = []
            for wordpos in simlayer[band][auxlayer]:
                range2 = freqband_by_word_4[wordpos]
                if range == range2 and len(balanced_words[band][range]) < minvalue:
                    balanced_words[band][range].append(wordpos)
                if len(balanced_words[band][range]) >= minvalue:
                    break
                if len(balanced_words[band][range]) > minvalue:
                    print("SOMETHING WENT WRONG")

    balanced_by_band = dict()
    for band in balanced_words:
        balanced_by_band[band] = []
        for range in balanced_words[band]:
           balanced_by_band[band].extend(balanced_words[band][range])

    print("Words left in each band after balancing for frequency range:")
    for band in balanced_by_band:
        print(band, len(balanced_by_band[band]))
    print("Number of words per band and freq after balancing:")
    for band in balanced_words:
        print(band, [(freqr, len(balanced_words[band][freqr])) for freqr in balanced_words[band]])

    return balanced_by_band


def get_freqbalanced_similarities(balanced_by_band, simlayer, vector):
    freqbalanced_similarities_by_layer_and_poly = dict()
    freqbalanced_similarities_by_layer_and_poly_by_word = dict()

    for band in ['mono', 'poly', 'low', 'mid', 'high']:
        if band not in freqbalanced_similarities_by_layer_and_poly:
            freqbalanced_similarities_by_layer_and_poly[band] = dict()
            freqbalanced_similarities_by_layer_and_poly_by_word[band] = dict()
        for layer in simlayer[band]:
            if layer not in freqbalanced_similarities_by_layer_and_poly[band]:
                freqbalanced_similarities_by_layer_and_poly[band][layer] = []
            if layer not in freqbalanced_similarities_by_layer_and_poly_by_word[band]:
                freqbalanced_similarities_by_layer_and_poly_by_word[band][layer] = dict()

            for wordpos in balanced_by_band[band]:
                    freqbalanced_similarities_by_layer_and_poly_by_word[band][layer][wordpos] = simlayer[band][layer][wordpos]
                    selfsim = np.average(simlayer[band][layer][wordpos])
                    freqbalanced_similarities_by_layer_and_poly[band][layer].append(selfsim)

    pickle.dump(freqbalanced_similarities_by_layer_and_poly_by_word, open(args.out_dir + "/" + vector + "/similarities_freq-bal.pkl", "wb"))

    avg_freqbalanced = dict()

    for polyband in freqbalanced_similarities_by_layer_and_poly:
        avg_freqbalanced[polyband] = dict()
        for layer in freqbalanced_similarities_by_layer_and_poly[polyband]:
            avg_freqbalanced[polyband][layer] = np.average(freqbalanced_similarities_by_layer_and_poly[polyband][layer])

    balfreqdf = pd.DataFrame.from_dict(avg_freqbalanced)

    balfreqdf.to_csv(args.out_dir + "/" + vector + "/poly_freq-bal.csv", sep="\t")

    return balfreqdf



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='en', type=str, help="en,fr,es,el")
    parser.add_argument("--out_dir", default="posfreq_results", type=str, help="directory where results will be saved")
    parser.add_argument("--english_freq_fn", default=0.01, type=float, help="if do_freq is used and the language is English, you need to provide the path to the Google Ngrams frequency file")

    args = parser.parse_args()

    if args.language == "en" and not args.english_freq_fn:
        sys.out("Provide a path to a file containing Google Ngrams frequencies in English with --english_freq_fn")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Load data for that language
    data, word_dict = load_data(language)
    freq_counts = load_freq_counts(language, args.english_freq_fn)

    ######### 1. DATASET ANALYSIS
    # 1.1 CHECK THE DISTRIBUTION OF POS OVER BANDS
    pos_by_word, pos_by_band, pos_by_band_props = posdist_overbands(word_dict)
    # 1.2 CHECK THE DISTRIBUTION OF FREQ OVER BANDS
    freqband_by_word_4, freq_by_band_4, freq_by_band_4_props = freqdist_overbands(word_dict, freq_counts)

    ######### 2. SIMILARITIES ANALYSIS (different for each vector type)

    vector = list(data.keys())[0] # whichever vector
    auxlayer = 1
    simlayer = dict()
    simlayer['mono'] = data[vector]["monopoly"]['monosemous'] 
    simlayer['poly'] = data[vector]["monopoly"]['poly-rand']
    simlayer['low'] = data[vector]["polybands"]['low']
    simlayer['mid'] = data[vector]["polybands"]['mid']
    simlayer['high'] = data[vector]["polybands"]['high']

    # 2.1 Balance pos/freq distributions
    words_pos_balanced = get_posbalanced_bands(pos_by_band, simlayer)
    words_freq_balanced = get_freqbalanced_bands(freq_by_band_4, simlayer)

    for vector in data:
        auxlayer = 1 if "c2v" not in vector else 0
        simlayer = dict()
        simlayer['mono'] = data[vector]["monopoly"]['monosemous']  
        simlayer['poly'] = data[vector]["monopoly"]['poly-rand']
        simlayer['low'] = data[vector]["polybands"]['low']
        simlayer['mid'] = data[vector]["polybands"]['mid']
        simlayer['high'] = data[vector]["polybands"]['high']

        # 2.2 Get avg selfsim in pos/freq balanced bands
        posdf = sim_by_pos(pos_by_word, simlayer, vector)
        balposdf = get_posbalanced_similarities(words_pos_balanced, simlayer, vector)

        freqdf = sim_by_freq(freqband_by_word_4, simlayer, vector)
        balfreqdf = get_freqbalanced_similarities(words_freq_balanced, simlayer, vector)

