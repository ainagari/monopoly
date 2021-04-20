# 🎲 Let’s Play mono-poly 🎩

Data and code (coming soon) for the paper:

Aina Garí Soler and Marianna Apidianaki (2021). Let’s Play mono-poly: BERT Can Reveal Words’ Degree of Polysemy and Partitionability into Senses. To appear in _Transactions of the Association for Computational Linguistics_ (TACL).


### Data

The sentences used in our experiments are found in the `data` folder. Files starting with `mono_poly` contain sentences for the `mono` and `poly` sets of words (introduced in Section 3). Files starting with `polysemy_bands` contain sentences for the `low`, `mid` and `high` polysemy bands. The language of the sentences is indicated at the end of the filename (`en/fr/es/el`). 
The files, which are pickled Python dictionaries, include sentences from all sentence pools (`poly-same`, `poly-rand`, `poly-bal`)


### Results

The `Results`folder contains the average self-similarities by layer obtained by every model. Folder names indicate the language and the model used (bbcased = bert-base-cased, bert = bert-base uncased) 



To be continued
