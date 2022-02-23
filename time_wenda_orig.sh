#!/bin/bash

prefix=$1 #This should be set to a specific, meaningful identifier for the dataset and classification problem.

filename=runtimes/${prefix}_wenda_orig.txt

python3 wenda_orig/train_models.py ${prefix}

# Confirm all feature models have been trained and confidence scores generated
conf_files=`ls -1q confidences/${prefix}_wenda_orig | wc -l`

if [[ $conf_files != 1 ]]; then
	echo "Error: not all models trained." > $filename
	exit 1
fi

end=$SECONDS
echo "Model training complete. Time to run: $end" > $filename
