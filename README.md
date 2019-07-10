# disease_normalization
Disease mention normalization project

### Data <br/>
Data set: NCBI disease corpus (mention level) <a href=" ">https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip</a> <br/>
Controlled vocabulary: MEDIC, the version containing 9,664 disease concepts that come with downloaded DNorm (filename `CTD_diseases.tsv`) <br>
For the scripts to work, a folder called `data` needs to be created. The following files needs to be placed inside: `CTD_diseases.tsv`, `NCBIdevelopset_corpus.txt`, `NCBItestset_corpus.txt`, and `NCBItrainset_corpus.txt`.

### Folders <br/>
`analysis`: contains analysis of experimental results <br/>
`baseline`: baselines, the scripts need to be moved to the folder `src` to work <br/>
`expt`: experiments reported in thesis, the scripts work in `src` <br/>
`scripts`: model evaluation, pretraining, test architecture, the scripts work in `src` <br/>
`unimplemented`: old code or not fully implemented code, the scripts work in `src`

### Files <br/>
`run*`: run the normalization programme, under different setups. Recommend to use the scripts in `expt` instead.

### Note <br/>
The scripts may need cached processed data to run successfully. The code for processing those data are usually commented out in the scripts (or in other scripts). Occasionally the codes are not documented (when the data processing is straight forward). 
