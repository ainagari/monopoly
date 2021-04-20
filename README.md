# 🎲 Let’s Play mono-poly 🎩

Data and code (coming soon) for the paper:

Aina Garí Soler and Marianna Apidianaki (2021). Let’s Play mono-poly: BERT Can Reveal Words’ Degree of Polysemy and Partitionability into Senses. To appear in _Transactions of the Association for Computational Linguistics_ (TACL).


### Data

The sentences used in our experiments are found in the `*\_data` folder. Files starting with `*\_mono_poly` contain sentences for the `*\_mono` and `*\_poly` sets of words (introduced in Section 3). Files starting with `*\_polysemy_bands` contain sentences for the `*\_low`, `*\_mid` and `*\_high` polysemy bands. The language of the sentences is indicated at the end of the filename (`*\_en/fr/es/el`). 
The files, which are pickled Python dictionaries, include sentences from all sentence pools (`*\_poly-same`, `*\_poly-rand`, `*\_poly-bal`)


### Results

The `*\_Results`folder contains the average self-similarities by layer obtained by every model. Folder names indicate the language and the model used (bbcased = bert-base-cased, bert = bert-base uncased) 



To be continued
