Automated Kelp canopy Mapping: A unet approach with Landsat & Citizen Science

	short 1-2 sentence summary of: This study developed and evaluated an
	end-to-end deep learning pipeline, utilizing a UNet architecture with pre-trained
	ResNet backbones, for semantic segmentation of kelp canopy in Landsat 7 im-
	agery using these Floating Forests labels, incorporating rigorous data preprocessing
	and augmentation. We found that a ResNet34 backbone, trained on cleaned and
	augmented data, achieved an Intersection over Union (IoU) of 0.5028, with data
	preprocessing and augmentation proving essential for optimal performance. Our
	study suggests that deep learning, leveraged with citizen-science-derived ground
	truth, offers a viable and scalable approach to automate kelp canopy mapping,
	which can enhance the efficiency of conservation efforts by reallocating resources
	towards direct ecological interventions.

	display image figures/good_predictions/TK423110.png

How to use the tool

	
	1 obtain landsat7 images you would like to train kelp on

		Must be shape (350,350,7) with the ordering of bands being:

			# 	0: Short-wave infrared (SWIR)
			# 	1: Near infrared (NIR)
			# 	2: Red
			# 	3: Green
			# 	4: Blue
			# 	5: Cloud Mask (binary - is there cloud or not)
			# 	6: Digital Elevation Model (meters above sea-level)

		if cloud mask and DEM are unavailable they can be substituted with layers of 0s

		store these in data/cleaned/train_satellite/

		for proof of concept you can use data_copy.py to copy training data to the directory

	2 clean data

		cleaning the raw, noisy landsat7 data has been automated through the use of the data_clean.py script

			cd data_cleaning
			python data_clean.py

	3 run inference

		download the model (https://drive.google.com/drive/folders/1TlhEzbolgPp9DIkzbnAVSdZWoLRjPctm?usp=sharing)

		store the 34_clean_aug folder in the runs/ directory

		load the model weights, and run the script

			cd ../models
			python generate_masks.py

	4 view results

		run script for to view generated masks

			cd ../data_visualization
			python output_view.py




How to reproduce my results

	download data (https://drive.google.com/drive/folders/12OTsSu9QpEbhQWyeh9ZKtMVTI74oBX-K?usp=sharing)

	save in cleaned folder

		cleaned/train_satellite1
		cleaned/train_kelp1

		(test_satellite unused)

	clean

		need to clean data in train_satellite1

			cd ../data_clean
			python data_clean.py

			(may need to adjust directory strucure in the file)

	split

		need to split data into training, validation, and testing sets

			cd ../utils
			python split_data.py

	train

		We can start training! adjust parameters at the top of 350resnet.py as needed to begin training simply run the file.

		once done training, all relelvant information is stored in the runs/ directory.

	Testing

		first, find the optimal threshold for segmenting the logits. adjust parameters in find_threshold.py according to what is in the runs/ directory. Then, run the file

		Then, adjust params in test.py and run the file. outputed masks are outputted to the output/ directory

		Lastly, to view the outputted masks, run data_compare.py.

	


