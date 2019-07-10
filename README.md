# disease_normalization
Disease mention normalization project<br>
This is the code for a master's thesis project. The thesis can be found at <a href=" ">https://www.utupub.fi/handle/10024/147720</a> <br>
<br>
To replicate the results from the thesis, please<br>
(1) download the necessay data as specified in the `Data` section below,<br>
(2) move the relevent scripts from `src/baseline` or `src/expt` to `src`,<br>
(3) change the PATH_TO_SOMETHING's in `defaults.cfg` to the corresponding files,<br>
(4) move the cahced data from `pickled` to `src` as described in the `Note` section below,<br>
(5) create folders for outputs as described in the `Output` section below.

### Data <br/>
Data set: NCBI disease corpus (mention level) <a href=" ">https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip</a> <br/>
Controlled vocabulary: MEDIC, the version containing 9,664 disease concepts that come with downloaded DNorm (filename `CTD_diseases.tsv`) <br>
Embeddings: the word vectors used are listed in the thesis<br>
For the scripts to work, a folder called `data` needs to be created. The following files need to be placed inside: `CTD_diseases.tsv`, `NCBIdevelopset_corpus.txt`, `NCBItestset_corpus.txt`, and `NCBItrainset_corpus.txt`.

### Folders <br/>
`analysis`: contains analysis of experimental results <br/>
`baseline`: baselines, the scripts need to be moved to the folder `src` to work <br/>
`expt`: experiments reported in thesis, the scripts work in `src` <br/>
`scripts`: model evaluation, pretraining, test architecture, the scripts work in `src` <br/>
`unimplemented`: old code or not fully implemented code, the scripts work in `src`

### Files <br/>
`run*`: run the normalization programme, under different setups. Recommend to use the scripts in `expt` instead.<br>

### Output <br/>
For every run, the configurations, models, and results are saved to `gitig/log`.<br>
If the model is saved, it is saved to `gitig`.<br>
For these functions to run correctly, please create a folder `gitig`, and `log` inside `gitig`.

### Note <br/>
The scripts may need cached processed data to run successfully. The code for processing those data are usually commented out in the scripts (or in other scripts). Occasionally the codes are not documented (when the data processing is straight forward). <br>
Some of the cached processed (those used in the scripts in `expt`) can be found inside `pickled`. Move the relevant ones to `src` before running the script. Decompress the file when appropriate.<br>
<br>
Known bug: the callback used in the scripts in `expt` does not save the best model.
