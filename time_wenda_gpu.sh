#!/bin/bash

prefix=$1 #This should be changed to a specific, meaningful identifier for the dataset and classification problem.
batch_size=$2 #How many feature models to train in one batch. For small datasets (i.e. few number of samples), this number can be raised without risk of memory flow errors, but very large datasets may require a number <100.
logistic=1 #This value should be 1 for logistic net regression (binary label data) and 0 for elastic net regression (continuous label data).

filename=runtimes/${prefix}_wenda_gpu.txt

start=$SECONDS

# Get number of columns
source_features=`awk -F"\t" '{print NF;exit}' $data_path/$prefix/source_data.tsv`
target_features=`awk -F"\t" '{print NF;exit}' $data_path/$prefix/target_data.tsv`
if [ $source_features -ne $target_features ]; then
	echo "Error: Source and target datasets have different numbers of features."
	echo "Source dataset has ${source_features} features, target dataset has ${target_features}."
	echo "Confirm your datasets are laid out so samples are rows and features are columns."
	exit 1
else
	echo "Preparing to train models for ${source_features} features..."
fi

echo $source_features
echo $batch_size

# Calculate number of batches to run
batches=$(( $source_features / $batch_size ))
if [ $(expr $source_features % $batch_size) != "0" ]; then
	batches=$((batches + 1))
fi

# Train feature models in batches
for (( i=0; i<$batches; i++))
do
	start=$(( $i * $batch_size ))
	stop=$(( $start + $batch_size - 1 ))
	echo "Training models ${start} to ${stop}..."
	python3 train_feature_models.py -p ${prefix} -s ${start} -r ${batch_size}
done

# Confirm all feature models have been trained and confidence scores generated
conf_files=`ls -1q $confidence_path/${prefix}_wenda_gpu | wc -l`

if [ $conf_files -ne $source_features ]; then
	echo "Error: not all models trained. Number of models trained: ${conf_files}" > $filename
	exit 1
fi

end=$SECONDS

echo "Model training complete. Time to run: $end" > $filename
