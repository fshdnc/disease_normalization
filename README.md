# disease_normalization
Disease mention normalization project

### Data <br/>
Data set: NCBI disease corpus <br/>
Controlled vocabulary: MEDIC

### Folders <br/>
analysis: contains analysis of experimental results <br/>
baseline: baselines, the scripts need to be moved to /src to work <br/>
expt: experiments reported in thesis, the scripts work in /src <br/>
scripts: model evaluation, pretraining, test architecture, the scripts work in /src <br/>
unimplemented: old code or not fully implemented code, the scripts work in /src

### Files <br/>
run*: run the normalization programme, under different setups. Recommend to use the scripts in `expt` instead.

### Note <br/>
The scripts may need cached processed data to run successfully. The code for processing those data are usually commented out in the scripts (or in other scripts). Occasionally the codes are not documented (when the data processing is straight forward. 
