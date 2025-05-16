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

	2 clean data

		cleaning the raw, noisy landsat7 data has been automated through the use of the data_clean.py script

			cd data_cleaning
			python data_clean.py


