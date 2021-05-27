# 🎲 Let’s Play mono-poly 🎩

Data and code for the paper:

Aina Garí Soler and Marianna Apidianaki (2021). [Let’s Play mono-poly: BERT Can Reveal Words’ Polysemy Level and Partitionability into Senses](https://arxiv.org/abs/2104.14694). To appear in _Transactions of the Association for Computational Linguistics_ (TACL).


### Data/

The sentences used in our experiments are found in the `data` folder. Files starting with `mono_poly` contain sentences for the `mono` and `poly` sets of words (introduced in Section 3). Files starting with `polysemy_bands` contain sentences for the `low`, `mid` and `high` polysemy bands. The language of the sentences is indicated at the end of the filename (`en/fr/es/el`). 
The files, which are pickled Python dictionaries, include sentences from all sentence pools (`poly-same`, `poly-rand`, `poly-bal`)


### Results/

The `Results`folder contains the average self-similarities by layer obtained by every model. Folder names indicate the language and the model used (bbcased = bert-base-cased, bert = bert-base-uncased) 


### Obtaining vector representations and similarities

In order to obtain representations and similarities from all the models used in the paper, you can run the script `run_multi_monopoly.sh.`
This runs `monopoly_extract_reps.py` with the necessary arguments to run all experiments.

### Analysis by POS and frequency

The script `pos_freq_analysis.py` serves for:
- checking the distribution of pos and frequency in the bands for the dataset of a specific language (Section 5.1 in the paper),
- balancing the dataset by pos/frequency (as explained in Section 5.3), 
- calculating SelfSim values by pos and frequency range (Section 5.2) 
- calculating SelfSim values by band in the balanced dataset (Section 5.3).

Some information is printed in the terminal, and results are saved in the `posfreq_results/` (or `--out_dir`) directory. The script can be run as follows:

`pos_freq_analysis.py --language en --out_dir [OUTPUT DIRECTORY] --english_freq_fn [PATH TO ENGLISH FREQUENCIES]`

where `--language` can be `[en|fr|es|el]`.

#### Note on frequency counts

We provide the frequency counts for French, Spanish and Greek (`freq_counts_[LANGUAGE].pkl`), which were calculated from the [OSCAR corpus](https://oscar-corpus.com/) (Ortiz Suárez et al, 2019). We, however, cannot provide the English counts. A path to [Google Ngrams frequencies](https://catalog.ldc.upenn.edu/LDC2006T13) (Brants and Franz, 2006) needs to be provided for English with the flag `--english_freq_fn`. 


### Significance and classifiers

The script `pos_freq_analysis.py` can be used to run significance tests  on the similarity values obtained (with `--mode significance`, Sections 3 and 4 of the paper) and to train classifiers for a given language (`--mode classification`, Section 6). See the note on frequency counts above if you want to run the frequency classifier (with `--do_freq`).
The script can be run as follows:

`python classification_and_significance.py --mode classification --language fr --do_freq`


### Clusterability experiments

Coming soon


### Citation

If you use the code in this repository, please cite our paper:
```
@article{garisoler-2021-let,
  title={{Let's Play Mono-Poly: BERT Can Reveal Words' Polysemy Level and Partitionability into Senses}},
  author={Gar{\'\i} Soler, Aina and Apidianaki, Marianna},
  journal={arXiv preprint arXiv:2104.14694},
  year={2021}
}
```


### References

Pedro Javier Ortiz Suárez, Benoît Sagot, and Laurent Romary. 2019. Asynchronous pipeline for processing huge corpora on medium to low resource infrastructures. In Proceedings of the 7th Workshop on the Challenges in the Management of Large Corpora (CMLC-7), Cardiff, UK. Leibniz-Institut für Deutsche Sprache.

Thorsten Brants and Alex Franz. 2006. Web 1T 5-gram Version 1. In LDC2006T13, Philadelphia, Pennsylvania. Linguistic Data Consortium.


### Contact

For any questions or requests feel free to contact me: aina dot gari at limsi dot fr

