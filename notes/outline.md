Abstract

	overview (paragraph summary)

	one sentence each that summarizes intro, background, methods, results, discussion

		intro -> x is important because of y

		background -> previous studies demonstrated x

		methods -> we have done x

		results -> we find that x

		discussion -> our study suggests that X

intro

	what topic has ur work been about

	what are the most important aspects of this topic

	why is this topic relevant to society

	what is known and what is still unclear

	how does your thesis connect to what is known/unknown

	provide a preview of results without providing too many technical details



background

	what has been done before? what is known

	- Cavanaugh et al. (2023) - CubeSats show persistence of bull kelp refugia amidst a regional collapse in California (https://doi.org/10.1016/j.rse.2023.113521)

		sing high-resolution (3m) PlanetScope CubeSat imagery from 2016-2021

		using a Naïve Bayes classifier (trained on manually labeled pixels) to predict kelp presence.

		calculates R^2 value against various datasets

		their method vs UAV: R² = 0.70

			https://doi.org/10.1002/rse2.295
		
		their method vs Landsat: R² = 0.49

			a decision tree to identify potential kelp pixels, and Multiple Endmember Spectral Mixture Analysis (MESMA) to estimate the fraction of each 30m pixel covered by kelp canopy. (not binary)
		
		their method vs CDFW: R² = 0.38

			they took the final CDFW shapefile product for 2016 (the year their PlanetScope data starts and the last year of successful CDFW surveys)

			https://wildlife.ca.gov/Conservation/Marine/Kelp/Aerial-Kelp-Surveys

	
	- Bell et al. (2020) - Three decades of variability in California's giant kelp forests from the Landsat satellites (https://doi.org/10.1016/j.rse.2018.06.039)

		used a decision tree to initially classify potential kelp pixels, followed by Multiple Endmember Spectral Mixture Analysis (MESMA) to estimate the continuous fractional cover of kelp within each 30m pixel.

		compared against the The Santa Barbara Coastal Long Term Ecological Research 

		scuba divers counted individual kelp plants in two distinct areas. (https://sbclter.msi.ucsb.edu/)

		calculate r^2 against the results of this study, 0.64



	- Floating Forests: Quantitative Validation of Citizen Science Data Generated From Consensus Classifications - https://arxiv.org/abs/1801.08522

methods

	what did you do? how did you do it?

	research -> build off someone elses work. explain what you were inspired by and what you changed

	ABSOLUTELY need details. someone should be able to easily reproduce it
	
	visualization of pipeline/workflow
	----------------------------------------------------------

	data => 5635 pairs. one binary mask (1 = kelp, 0 = not kelp). and one 7 layer landsat image. 

		# Band description:
		# 	0: Short-wave infrared (SWIR)
		# 	1: Near infrared (NIR)
		# 	2: Red 
		# 	3: Green 
		# 	4: Blue 
		# 	5: Cloud Mask (binary - is there cloud or not)
		# 	6: Digital Elevation Model (meters above sea-level)

		Lets talk about the format of landsat7 images

		Typical images from a phone or camera are only RGB
		There is a red, green, and blue channel, which creates a composite

		Again, Landsat7 data has extra layers that provide us valuable information




		Next, Near Infared measures wave lengths to large for us to see.
		This again is super advantageous,
		chlorophyll in vegetation reflects this wavelength super well. 
		So, kelp, which is high in clorophyll, will be extremely visible
		This will help the model “see” kelp easier than just by color

		Next is the cloud mask
		Clouds sometimes block the satellite's view of the earths surface. 
		If a pixel is marked as a cloud
		we know that the shortwave infared, near infared, Red, Green, and Blue values for that pixel are unreliable and should be ignored 
		Obviously, You can't measure an accurate reading of the earths surface through clouds

		Lastly, the Digital evolution model is a representation of the terrain's elevation height above sea level. 
		This will help, as we can rule anything above sea-level as not kelp

		Hopefully this demonstrates the benefits of extra spectral information compared to traditional RGB images.`

			


	
results

	what did you find out / how did it work. very dry, numbers

discussion

	speculative -> what do the results mean. we think ____ happened because...

	discuss weaknesses / what could be improved


------------------------------------------------------------------------------

Paper Title: (Needs to be specific - e.g., "Automated Kelp Canopy Segmentation from Landsat 7 Imagery using Transfer Learning on Citizen Science Data")

Abstract (~250-300 words)
	
	- Intro: Monitoring kelp forest dynamics is critical for coastal ecosystem health, but large-scale mapping remains challenging.

	- Background (CHANGE THIS SUCKS): While methods like MESMA and citizen science projects like Floating Forests exist for kelp mapping from Landsat, automated pixel-level segmentation using deep learning on this data source is less explored.

	- Methods: We processed Landsat 7 data classified by the Floating Forests project and trained UNet models with pre-trained ResNet backbones (18, 34, 50) using data augmentation to segment kelp canopy.

	- Results: We found that a UNet with a ResNet-X backbone achieved the best performance (e.g., Dice coefficient of Y, IoU of Z) on a held-out test set, significantly outperforming models trained from scratch, with data augmentations providing a notable improvement.

	- Discussion: Our study suggests that leveraging transfer learning/FINE TUNING and data augmentation is highly effective for automated kelp segmentation from Landsat 7  data, offering a promising avenue for large-scale monitoring despite data imbalance and sensor limitations.

=========================================================================
Introduction (~1.5-2 pages)
=========================================================================

	- The kelp crisis
		
		- we are expierencing a large die-off of kelp in california

		- due to an explosion of pacific purple sea urchin populations

		- Their natural predator, the sunflower sea star  suffered from sea star wasting syndrome starting in 2013

		- This almost lead to their extinction, they are now critically endangered

		- With no predators, the urchin population exploded. Urchins feed on kelp

		- So many urchins eating kelp is destroying kelp forests
		
		- estimated they have destroyed 80% of the kelp population in Northern California since 2013

	- The Need for Intervention:

		- Human divers can manually destroy urchin populations.

		- With limited divers, the question becomes where to send divers.

		- As the West Coast Region Kelp Team puts it we need an efficient way 
		to conduct annual assessments of kelp forest ecosystem health

		- If we find areas with poor kelp ecosystem health,
		we know their are high concentrations of urchins there.

		- Then, we can have divers target these areas to destroy the urchins
		in hopes the kelp can grow back

	- Challenges with Existing Monitoring Methods and Why New Methods Are Needed

		- SCUBA surveys - costly, small spatial scale.

		- Aerial surveys - costly, repetitive human annotation

			- The California Department of Fish and wildlife annually flies planes over the coast of california 

				- Pictures of the ocean’s surface are taken from the plane,

				- Then, scientists manually detect kelp and calculate its area in square feet. 

				- This is expensive, time consuming, and repetitive. 

				- This raises the question, Can this task be automated?


	- existing digital data and automation techniques for kelp maps
	
		- California Department of Fish and Wildlife shapefiles derived from aerial surveys (https://wildlife.ca.gov/Conservation/Marine/Kelp/Aerial-Kelp-Surveys)

			- Labor-Intensive & Time-Consuming to produce

		- Using unoccupied aerial vehicles to map and monitor changes in emergent kelp canopy

			- extremely small sites "The surveyed sites varied in size from 0.16 to 1.48 km²

		- MEMSA LAndsat data 
		
			- MESMA relies on spectral physics and pre-defined endmembers

			- In satellite imaging, endmembers represent pure materials, such as kelp or water. Because water itself doesn't always look the same – it might be clear, cloudy with mud, or have bright sun reflections – there are many types of "pure water" endmembers. The correct endmember must be selected in order for the MESMA process to be accurate. the paper uses an automatic selection process to choose the correct set of endmembers per image. This fixed, automated process could lead to an incorrect endmember selection due to noise or the sampling process not perfectly represent all water variations in that specific scene. 

	- Bridging the Data Gap: Citizen Science:

		Floating Forests offers a distinct approach, as its labels are derived from human visual interpretation
		
		The study showed that accurate kelp labels could be constructed from the consensus of multiple untrained participants. 

		- leveraging historical data from Earth-observing satellite programs like Landsat

		- based on landsat 7 sattalite images of falkand islands 
			
		- Studies (https://doi.org/10.48550/arXiv.1801.08522) show this consensus method produces accurate kelp maps.
			
		- Relies on human pattern recognition, potentially handling some visual variations differently than the automated spectral approach.

		- Human-in-the-loop approach potentially offers different robustness characteristics to noise or visual variations compared to purely spectral methods like MESMA

	- Remote Sensing as a Solution:

		- I propose using Machine Learning, specifically Semantic Segmentation, to measure the area of surface level kelp on pre-existing satellite imagery

		- It can then find areas in low ecosystem health, making it easier to allocate diver resources.

	- Known: 
	
		- landsat7 data is plentiful

		- citizen science via Floating Forests provides large-scale classifications (validated by Rosenthal et al.). 
		
		- Traditional automated methods like MESMA exist (Bell et al. 2020).

	- Unknown: 
	
		- Can modern deep learning techniques (specifically semantic segmentation) effectively automate *pixel-level* kelp mapping using landsat7 data

		- Your Contribution: State the objective clearly: "This study aims to develop, train, and evaluate a deep learning pipeline based on the UNet architecture with pre-trained ResNet backbones for semantic segmentation of kelp canopy in Landsat 7 imagery, using ground truth labels derived from the Floating Forests citizen science project."

	- Preview of Approach and Findings: 
	
		- explored different ResNet backbones
		
		- investigated data augmentation strategies
		
		- found using pretrained weights for the encoder + fine tuning essential
		
		- developed a functional model implemented in a testing pipeline, demonstrating the feasibility of the approach.


=============================
Background (~2-3 pages)
=============================


	- Benefits of Landsat 7 

		Rather than having the typical 3 Red and Green Blue channels, The Landsat 7 ETM+ derived data product used in this study comprised seven distinct channels, which provide more spectral information beyond color

			Near-Infrared (NIR) 

				- Near Infared measures wave lengths to large for us to see. This again is super advantageous, chlorophyll in vegetation reflects this wavelength super well. So, kelp, which is high in clorophyll, will be extremely visible. This will help the model “see” kelp easier than just by color

			Short-Wave Infrared (SWIR) 
			
				- First, Short Wave Infared measures wave lengths too small for us to see. Whata good about short wave infrared? water heavily absorbs these wavelengths and the land highly reflects this wavelength. This Creates a high contrast between land and water

			Red
				
				- Absorbed by chlorophyll in kelp for photosynthesis, providing contrast with NIR reflectance and contributing to vegetation indices.

			Green
			
				- Reflects moderately from healthy vegetation (giving kelp its color) and can also indicate water column properties like sediment or phytoplankton.

			Blue
				
				- Penetrates water deepest, potentially useful for submerged features (less so for surface canopy), but most affected by atmospheric scattering/haze.

			Cloud Mask (binary)
			
				- A critical quality indicator used to identify and exclude pixels where clouds obscure the view of the surface, preventing misinterpretation.

			Digital Elevation Model (DEM)
			
				- Provides elevation data used primarily to accurately mask out land areas, ensuring the analysis focuses only on water pixels where kelp could exist.

		WRS - Nasa's landsat series of sattelite utilize the Worldwide Reference System, which poses the potential for us to translate image pixels to coordinates on earth's surface

		- Landsat 7 Satellite photographs the same location on earth every 16 days, making measuring changes over time possible

	
	- Drawbacks of Landsat 7

		- Scan Line Corrector (SLC) failure (post-May 2003):

			The SLC was an electro-mechanical component designed to compensate for the satellite's forward motion, ensuring complete image coverage without gaps between scan lines.

			On May 31, 2003, the SLC mechanism permanently failed.

			All Landsat 7 ETM+ images acquired after this date ("SLC-off" data) exhibit a zig-zag scan pattern, resulting in wedge-shaped gaps of missing data.

			These gaps are narrowest at the image center (nadir) and widen towards the edges, causing approximately 22% of pixel data to be missing in a typical SLC-off scene.

			this significantly impacts the usability of SLC-off imagery for applications requiring complete spatial coverage unless specific gap-filling techniques are applied.

			The dataset used in this study included Landsat 7 imagery which was affected by the SLC failure, presenting an additional challenge for consistent feature extraction and labeling.
			
			https://www.usgs.gov/landsat-missions/landsat-collection-2-known-issues

	- Established Automated Method On Landsat Data
	
		- Spectral Mixture Analysis (SMA/MESMA)

			- Model a pixel's spectrum as a linear combination of pure reference spectra ('endmembers').

			- pure reference spectra could be 'water' or 'kelp'. Each pixel in the image is some combination of these pure materials

			- Water is so volatile we can't represent it as a single endmember. We have endmembers for clear water, shallow water, reflective water, cloudy water... etc

			- MESMA approach Uses a single, constant kelp endmember

			- Uses a multiple water endmembers unique to each scene.
        
				- As described by Bell et al. (2020), Spectra are extracted automatically from 30 pre-defined locations within each scene known to be consistently water-covered. This set of 30 scene-specific spectra forms the library for that image's MESMA run.

			- pros

				- Provides quantitative sub-pixel *fractional cover*
				
				- can be linked to biomass (citing Bell et al. 2020 validation), suitable for long-term, large-scale automated analysis.

   			- cons  
   
   				- Accuracy depends heavily on the *representativeness* of the chosen endmembers (both the static kelp and the scene-specific water spectra extracted from fixed points).

				- The automated extraction from fixed points might not always capture the *full range* of water variability

				- could be affected by localized noise/haze at those points in a specific image

				-  Known struggles with very low fractional cover (<~20%, citing Hamilton et al. 2020, Cavanaugh et al. 2023)

		
	
	- A new Approach with an alternative Data Source: The Floating Forests Project
    
		- Floating Forests is a large-scale citizen science project on the Zooniverse platform, specifically designed to generate kelp labels for Landsat imagery.

    	- Volunteers visually inspected Landsat 7 images, displayed using SWIR/NIR/Red bands, and manually traced kelp canopy borders.
    
		- The Consensus Mechanism: ressulting Data relied on agreement among multiple (up to 15) untrained volunteers. The final label for a pixel is derived from this consensus (e.g., requiring >=4 votes, as determined by Rosenthal et al.).

   		- Summarize Validation Results: The Rosenthal et al. study (preprint/report) confirmed that this consensus approach yields kelp classifications with accuracy (measured by MCC) comparable to expert-derived methods, establishing the dataset's utility.

		- Distinct from other data: it relies on human visual pattern recognition and consensus judgment, rather than spectral physics and automated endmember extraction. 
		
			- This makes it a fundamentally different type of label source, potentially capturing different information or having different sensitivities to noise/ambiguity compared to MESMA.

		
	- Deep Learning for Image Segmentation

    	- Semantic Segmentation: The task of assigning a class label (e.g., kelp or background) to every pixel in an image.
    
		- The UNet Architecture: encoder-decoder structure with skip connections. 
		
			A UNET is a CNN that has skip connections. 
			
			Normally, in a CNN, layers only exchange information with their neighbors. 

			Skip connections allow layers to talk to more than its neighbors to combine coarse and fine-grained feature information.

			



    - Transfer Learning and Fine-Tuning:
        
		- Tranfer Learning:

			-  Using models (like ResNet-18, -34, -50) pre-trained on large, general datasets (ImageNet).

			-   Leverages powerful, pre-learned visual features, reducing the need for labeled data specific to the target task.

        - Fine-Tuning
		
			Allow pre-trained encoder/model weights to be updated during training

			- model (specifically the encoder) adapt more precisely to the specific nuances and patterns of kelp photographed from orbit.
    

	- Data Augmentation: 
	
		- Data augmentation is crucial for  increasing data diversity. 
		
		- we can artificially expand the training dataset by applying random transformations (flips, rotations, brightness changes, etc.) to input images. 
		
		- while the difference is trivial to us, the model sees this as an entirely new image.
		
		- supplemental training data improves model robustness, generalization, and helps prevent overfitting



===============================================================
Methods (~3-4 pages) - *Needs high detail for reproducibility*
===============================================================

	flowchart diagram illustrating this entire pipeline (Data Download -> Clean -> Split -> Train -> Threshold -> Test)

	intro

		The methodology involved several key stages: data acquisition and preprocessing from the Floating Forests dataset, model architecture design based on a UNet with a ResNet backbone, a detailed training procedure incorporating data augmentation and learning rate scheduling, determination of an optimal prediction threshold, and finally, evaluation on an unseen test set."

	pipeline

		floating forests data

			Feature Data (Satellite Imagery):
		
				Imagery was derived from the Landsat satellite missions (Landsat 5, 7, and 8) and provided as 350 x 350 pixel, unreferenced GeoTIFF tiles. Each tile corresponded to coastal waters around the Falkland Islands.

				Each GeoTIFF contained seven co-referenced bands at a 30-meter spatial resolution:

				Bands 0-4: Spectral bands representing surface reflectance, rescaled to 16-bit integers. These included Short-Wave Infrared 1 (SWIR1), Near-Infrared (NIR), Red, Green, and Blue. A value of -32768 indicated missing data.
				
				Band 5: A binary cloud mask (1 = cloud, 0 = no cloud).

				Band 6: A Digital Elevation Model (DEM) in meters above sea-level, derived from ASTER data.

				Filename schema: <tile_id>_satellite.tif.

			Label Data (Ground Truth Masks):

				Binary segmentation masks indicating the presence (1) or absence (0) of kelp canopy.

				These labels were generated by citizen scientists on the Floating Forests platform through visual interpretation and tracing of kelp in the corresponding Landsat imagery.

				The masks were provided as single-band, 350 x 350 pixel TIFF images.

				Filename schema: <tile_id>_kelp.tif.

			data source provided a 'train' set with features and labels, and a 'test' set with only features (labels withheld). For this study, the provided 'train' set was further subdivided into our own training, validation, and test partitions."

			should be saved in .../data/cleaned/

		cleaning data

			to address potential missing values, noise, and outliers inherent in the raw data, data cleaning was necessary

			data clean script reads in the sattelite train folder (no processign needed for ground truth)

			the script:

				Identifies Invalid Pixels: Pixels were marked as invalid based on:
				
					A specific marker value (-32768) is used as a no-data value in the original Landsat processing
			
					Negative values (< 0) in spectral bands (Bands 0-4).
			
					Cloud & Water Masking: Pixels corresponding to clouds (using Band 5) and water bodies (using DEM Band 6 <= 0) were identified.
			
				performs Statistics Calculation: Dataset-wide mean and standard deviation were calculated for each band designated for normalization (Bands 0-4 and 6). 
				
					Invalid pixels (marker, negative, cloud, water) were excluded from these statistical calculations.
			
				Prior to normalization, pixel values in the target bands were clipped based on pre-defined global thresholds (corresponding to approximate 1st and 99th percentiles) to mitigate the impact of extreme outliers.

					percentiles were derived from derived from an initial exploratory analysis of the entire dataset

			Standardization (Z-score): Spectral bands (0-4) and the DEM band (6) were standardized by subtracting the calculated dataset mean and dividing by the dataset standard deviation for that band.
			
			Handling Missing Data Post-Normalization: Pixels originally identified as invalid (marker value or negative) were replaced with 0.0 after standardization.

				0.0 was chosen as zero represents the mean after Z-score standardization, this minimizes its impact on subsequent layers"
			
			Cloud Mask Preservation: The original Cloud Mask band (Band 5) was preserved without numerical normalization or clipping.
			
			Output Format: Processed images, with normalized bands and the original cloud mask, were saved back in-place as 32-bit floating-point TIFF files in (Height, Width, Channels) format.


		split data 

			rename original train folders, add 1 to end "train_satellite1", "train_kelp1"

			run split data script

			creates a 70 15 15 split (train, val, test)

				train -> model trains from this folder

				val-> used for balidation after each epoch. also used to calculate the threshold for assiging a given pixel as kelp.

				tets-> model never sees this until after training and acts as the hold out test set. is kept seperated so data is truly never seen before

			resulting 6 folders (sattelite/gt for train, for val, for test.)

			Images were randomly assigned to train, validation, and test sets


		model training

			Model Architecture: 
			
				A U-Net like segmentation model was employed, utilizing a pre-trained ResNet (configurable to ResNet18, ResNet34, or ResNet50) as its encoder backbone. 

					The UNet architecture was selected due to its demonstrated effectiveness in biomedical image segmentation and its increasing application in remote sensing. 
			
					Its encoder-decoder structure with skip connections allows for the capture of both contextual information and precise localization of features, making it well-suited for pixel-wise kelp canopy segmentation."

					ResNet encoders are known for their strong performance on image recognition tasks, and using their pre-trained weights allows for effective transfer learning, often leading to better generalization and faster convergence, especially with limited domain-specific training data."
				
				The ResNet's initial convolutional layer was modified to accept 7 input channels corresponding to the hyperspectral satellite bands. 
				
				A custom decoder composed of sequential ConvTranspose2d, BatchNorm2d, and ReLU layers was used to upsample features and produce a single-channel output for binary segmentation.

				The decoder comprised four upsampling blocks. Each block consisted of a 2D transposed convolution to double the spatial resolution, followed by batch normalization and a ReLU activation. This architecture progressively reduced the channel depth from the encoder's output (512 for ResNet18/34 or 2048 for ResNet50) through intermediate channel depths of 256, 128, and 64, down to 32 channels before a final 1x1 convolutional layer produced the single-channel logit output."

			Data Handling:

				Training data was subjected to on-the-fly augmentations, including horizontal flips, vertical flips, random 90-degree rotations, and the addition of small Gaussian noise to spectral bands 0-4. any given transformation had a 50% probability of being appied to the input image.
				
				Validation data was not augmented.

				Data was loaded using PyTorch DataLoaders, with a batch size of 8 for training and 16 for validation.

			Training Setup:

				The model was trained using PyTorch Lightning.

				The loss function was Binary Cross-Entropy with Logits (BCEWithLogitsLoss).

				During validation, Intersection over Union (IoU) for the "kelp" class was calculated and logged.

					Intersection over Union (IoU), also known as the Jaccard Index, was selected as it is a standard metric for assessing the accuracy of semantic segmentation models. It measures the overlap between the predicted and ground truth regions.

				Learning Rate 
					
					The Adam optimizer was used with an initial learning rate of 1e-4.

						The Adam (Adaptive Moment Estimation) optimizer combines the advantages of two other extensions of stochastic gradient descent:
						
							AdaGrad, (Adaptive Gradient) dynamically adjusts the learning rate for each parameter based on its gradient history
							
							RMSProp, (Root Mean Squared Propagation) computes a moving average of squared gradients to scale parameter updates, preventing drastic fluctuations and promoting faster convergence. 
							
						By computing adaptive learning rates for each parameter from estimates of first and second moments of the gradients, Adam often achieves efficient convergence and robust performance across a variety of deep learning architectures and datasets, making it a common default choice.

					A Cosine Annealing learning rate scheduler (CosineAnnealingLR) was employed to gradually reduce the learning rate over MAX_EPOCHS down to a minimum of 1e-7.

						A custom LearningRateMonitor callback logged and printed learning rate changes.

					Training was performed for a maximum of MAX_EPOCHS or until early stopping criteria were met.

						An EarlyStopping callback monitored the validation loss, halting training if the loss did not improve by at least EARLY_STOPPING_MIN_DELTA (e.g., 0.001) for EARLY_STOPPING_PATIENCE (e.g., 10) consecutive epochs.

				A ModelCheckpoint callback saved the model checkpoint corresponding to the best validation loss observed during training.

			Resuming Training: The script includes functionality to resume training from the latest available checkpoint if a previous run in the same output directory was interrupted.

			Hardware & Precision: Training utilized a GPU if available. For GPU training, 32-bit precision was specified

			Final Model: After training, the state dictionary of the model that achieved the best validation loss was saved as a .pth file.


		Finding threshold

			The model outputs continuous probability scores (or logits that are converted to probabilities via sigmoid) for each pixel representing the likelihood of it being kelp. A threshold is necessary to convert these probabilities into a definitive binary decision (kelp or not-kelp) for evaluation and practical application.

			A dedicated script was used to determine the optimal probability threshold for converting model logits to binary (kelp/no-kelp) predictions.

			The script loaded the best model weights (best_weights.pth) saved from a specific training run, identified by RUN_NAME and BACKBONE_NAME.

			A range of potential thresholds (0.2 to 0.6 in steps of 0.004) was tested.

			The optimal threshold was determined by applying the fully trained model (using the weights corresponding to the best validation loss achieved during the entire training process) to the complete validation set after training was complete. 
			
				The sigmoid output probabilities from the model for every pixel in the validation set were first collected. 
				
				Then, a range of potential thresholds was applied to these pre-computed probabilities to generate binary masks, and the threshold that maximized the IOU score on the validation set was selected."

			The threshold that maximized the IOU on the validation set was selected as the optimal threshold.

		test phase

			Purpose: A dedicated script was used for evaluating the trained segmentation model on an unseen test dataset and for generating final prediction masks.

			loads the appropriate trained model weights and associated optimal threshold.

			the test dataset was loaded, consisting of satellite images and corresponding ground truth kelp masks from pre-defined split directories. 

			The optimal probability threshold, previously determined on the validation was used to convert the model's sigmoid output probabilities into binary (kelp/no-kelp) predictions.

			The model performed inference on the entire test set, generating raw probability outputs for each pixel.

			optional Land Mask Post-processing is available to post-process the binary predictions. If enabled, any pixels predicted as kelp that corresponded to land areas (DEM band value > 0) were set to no-kelp (0).

			Four standard segmentation metrics were calculated between the (potentially land-masked) binary predictions and the ground truth test masks:
				
				Intersection over Union (IoU) / Jaccard Index
				Precision
				Recall
				F1-Score

			The final binary prediction masks (after thresholding and optional land masking) were saved as TIFF files.



===============================================================
Results (~2-3 pages) - *Focus on objective findings*
===============================================================

	Overview of Experimental Setup

		the main experimental variables tested:

			The effect of data preprocessing (cleaned vs. original) on model accuracy.

			The impact of applying random data augmentation during training on model accuracy.

			A comparison of performance across different ResNet backbones (ResNet18, ResNet34, ResNet50).

			Evaluation of the land mask post-processing step on correcting potential misclassifications over land.

	

		performance was evaluated using Intersection over Union (IoU), Precision, Recall, and F1-Score on the held-out test set.

	
	Impact of Data Preprocessing (Cleaning)

		Backbone	Preprocessing	Augmentation	IoU	Precision	Recall	F1-Score
		ResNet18	Original	No	0.3043	0.4192	0.5261	0.4666
		ResNet18	Cleaned	No	0.4437	0.5874	0.6446	0.6147
		ResNet34	Original	No	0.3239	0.4645	0.5168	0.4893
		ResNet34	Cleaned	No	0.4492	0.5827	0.6622	0.6200
		ResNet50	Original	No	0.3656	0.5178	0.5543	0.5354
		ResNet50	Cleaned	No	0.4419	0.5782	0.6521	0.6130

		The application of data cleaning and normalization procedures consistently improved model performance across all ResNet backbones when augmentations were not used.

		For instance, with the ResNet18 backbone, IoU increased from 0.3043 (original data) to 0.4437 (cleaned data), representing a substantial improvement of 45%. Similarly, ResNet34 saw an improvement of 38%, and Resnet50 saw 20%.
	
		data cleaning was a crucial step for achieving better baseline performance.

	Impact of Data Augmentation

		Backbone	Preprocessing	Augmentation	IoU	Precision	Recall	F1-Score
		ResNet18	Cleaned	No	0.4437	0.5874	0.6446	0.6147
		ResNet18	Cleaned	Yes	0.4983	0.6387	0.6939	0.6652
		ResNet34	Cleaned	No	0.4492	0.5827	0.6622	0.6200
		ResNet34	Cleaned	Yes	0.5028	0.6355	0.7066	0.6692
		ResNet50	Cleaned	No	0.4419	0.5782	0.6521	0.6130
		ResNet50	Cleaned	Yes	0.5010	0.6386	0.6993	0.6676

		Data augmentation applied during training on the cleaned dataset further enhanced model performance for all backbones.
	
		With the ResNet18 backbone, IoU improved from 0.4437 (no augmentation) to 0.4983 (with augmentation).

		The ResNet34 backbone showed the highest overall performance when combined with cleaned data and augmentations, achieving an IoU of 0.5028 and an F1-Score of 0.6692. Resnet50 saw a very similar increase in performance.

		data augmentation provided a significant boost to the models trained on cleaned data.

	Comparison of ResNet Backbones

		Backbone	Preprocessing	Augmentation	IoU	Precision	Recall	F1-Score
		ResNet18	Cleaned	Yes	0.4983	0.6387	0.6939	0.6652
		ResNet34	Cleaned	Yes	0.5028	0.6355	0.7066	0.6692
		ResNet50	Cleaned	Yes	0.5010	0.6386	0.6993	0.6676

		When comparing the performance of different ResNet backbones using cleaned data and augmentations, the ResNet34 backbone achieved the highest IoU (0.5028) and F1-Score (0.6692).

		The ResNet50 backbone performed comparably (IoU: 0.5010, F1: 0.6676), while the ResNet18 backbone was slightly lower (IoU: 0.4983, F1: 0.6652).

		This suggests that while deeper models offered a slight advantage, the gains diminished beyond ResNet34 for this specific task and dataset.

	Effect of Land Mask Post-processing

		The application of a land mask as a post-processing step, designed to remove any kelp predictions over land areas (DEM > 0), was evaluated.

		Across all experimental conditions (different backbones, preprocessing, and augmentation strategies), the land mask had a negligible or no impact on the reported evaluation metrics (IoU, Precision, Recall, F1-Score). 
		
		For example, for the best performing model (ResNet34, cleaned data, augmentations), the IoU remained 0.5028 both with and without the land mask."

		Minor variations (e.g., an increase in IoU from 0.3043 to 0.3091 for the ResNet18 original data no-augmentation run) were observed in some lower-performing models, potentially due to the model incorrectly predicting kelp on land pixels which were then corrected. 
		
		However, for the higher-performing models, this effect was not apparent.

		for the primary test set evaluation, the land mask did not significantly alter the overall segmentation performance on kelp itself, suggesting the models were largely not predicting kelp over land areas. 
		
		it is possible that such predictions were being made, however they did not substantially impact the metrics calculated.

	Summary of Best Performing Model

		The best overall performance was achieved using a UNet with a ResNet34 backbone, trained on cleaned and normalized data with data augmentations applied. This configuration yielded an Intersection over Union (IoU) of 0.5028, a Precision of 0.6355, a Recall of 0.7066, and an F1-Score of 0.6692 on the held-out test set, using an prediction threshold of 0.3414.

	Qualitative Results 
	
		To visually assess the model's performance, representative examples of predictions from the best-performing model (ResNet34, cleaned, augmented) on test set images are presented below

			*Successful segmentation of clear kelp patches*

				*Even works on SLC failure images!*

			*Poor Performance & missed detections of small patches of kelp*


===================================================================
Discussion (~2-3 pages) - *Interpret and contextualize*
===================================================================

	Summary of Key Findings and Performance

		Successful development of a deep learning pipeline (ResNet-UNet) for kelp segmentation using Landsat 7 and Floating Forests data.
		
		ResNet34 with cleaned, augmented data achieved an IoU of 0.5028

		Data cleaning, normalization, and augmentation proved crucial to acheiving a high accuracy. 

		early findings showed training a model from scratch was not feasible on such little labeled data; Transfer learning and find tuning a pre-trained encoder was key to creating accurate kelp masks.

		there was a negligible impact of the land mask post-processing on the best models, suggesting good specificity regarding land.

	
	
	Interpretation of Results: Why did it work?

		Stabilizing Input Distributions: 
		
			Normalization (Z-score standardization) rescales input features to have zero mean and unit variance. 
			
			This helps stabilize the learning process, preventing certain features with larger numerical ranges from disproportionately influencing weight updates during backpropagation, leading to faster and more stable convergence.

			Clipping extreme values (1st/99th percentiles) before normalization reduces the skewing effect of sensor artifacts or rare environmental conditions

			replacing specific marker values (-32768) and negative values (often indicative of errors or no-data in reflectance) with a neutral value (0.0 post-standardization) ensures these pixels don't introduce NaNs or extreme values into calculations, allowing the model to learn more effectively from valid data points.

			Satellite imagery inherently contains noise from sensor imperfections, atmospheric interference, and bi-directional reflectance effects. Cleaning steps help reduce this noise, allowing the model to learn more meaningful underlying features related to kelp presence rather than spurious artifacts.

		Power of Data Augmentation:

			Augmentations (flips, rotations, noise injection) artificially expand the training dataset by creating modified but still plausible versions of existing images. This exposes the model to a wider range of visual variations without needing to collect more raw labeled data.

			By training on these varied examples, the model learns to become invariant to such transformations. This helps it generalize better to unseen test data, which may exhibit similar variations in kelp orientation, illumination, or minor sensor noise.

			Citizen science labels (even consensus-based) can have slight inconsistencies in exact boundary placement. Augmentations that slightly shift or rotate features might make the model less sensitive to these minor label variations, forcing it to learn more fundamental features of kelp.

			With a limited dataset, models can easily overfit (memorize the training data). Augmentations act as a form of regularization, making it harder for the model to overfit and encouraging it to learn more generalizable representations.

		Importance of Transfer Learning (Pre-trained ResNet Backbones):

			ResNet backbones pre-trained on ImageNet (a massive dataset of diverse natural images) have already learned robust hierarchical features – edges, textures, shapes, and object parts – in their early and mid-level layers. These low-level visual primitives are often transferable and useful even for very different domains like satellite imagery.

			Training a deep network from scratch (randomly initialized weights) requires a very large amount of domain-specific labeled data to learn meaningful features. With a specialized and smaller dataset like the Floating Forests project, starting from pre-trained weights provides a much better initialization, helping the model converge faster and to a better solution.

			Because the backbone already understands basic visual concepts, the model needs fewer examples from the target domain (kelp images) to learn the specific features that distinguish kelp from water, compared to learning everything from scratch.

		Backbone Choice (ResNet34 as a Sweet Spot):

			Sufficient Capacity for the Task: ResNet34 likely offered enough model capacity (depth and number of parameters) to learn the complex features needed to distinguish kelp in 30m Landsat imagery from the Floating Forests labels.

			While ResNet50 is deeper and theoretically more powerful, the additional complexity might not have translated to significantly better performance on this specific dataset and task. The subtle differences it could learn might have been outweighed by the increased risk of overfitting to the training data

			ResNet18, while still effective due to transfer learning, might have had slightly insufficient capacity to capture all the necessary nuances compared to ResNet34, leading to slightly lower performance.

		Thresholding:

			The optimal threshold is typically chosen to maximize a specific evaluation metric (like IoU or F1-score) on a validation set. This ensures the binary predictions are as aligned as possible with the ground truth according to that metric.

			The ideal threshold value is not universal; it often varies depending on the specific model architecture, the training data characteristics (e.g., class imbalance), and the loss function used. This was evident in the results where different runs had different optimal thresholds.

			Changing the threshold directly impacts the trade-off between precision (minimizing false positives) and recall (minimizing false negatives). A lower threshold increases recall but may decrease precision, and vice-versa. The optimal threshold finds a good balance for the chosen metric, IOU.

	
	The Developed End-to-End Solution and its Significance

		This study successfully developed and demonstrated an end-to-end pipeline for automated kelp canopy segmentation, from raw satellite image tile input to final binary prediction masks. 
		
		This automated approach offers significant advancements over traditional kelp monitoring methods.
		
		Unlike aerial surveys (e.g., CDFW), our method utilizes pre-existing, freely available historical Landsat imagery, eliminating the substantial costs and logistical complexities associated with dedicated aerial image acquisition campaigns. 
		
		The model automates the segmentation process, removing the need for ongoing, time-consuming manual delineation of kelp by experts for each new satellite image—a major bottleneck in methods like the traditional CDFW aerial survey processing. 
		
		This automated, model-based approach is inherently scalable, capable of processing large archives of satellite imagery much more rapidly than manual methods. 
		
		The consistent revisit cycle of satellites like Landsat further opens possibilities for tracking changes in kelp extent over time, which could aid in identifying areas undergoing stress.

		By reducing the expenditure on image acquisition and manual annotation, resources can be reallocated to more direct conservation efforts.
		
		Specifically, the money saved from these prior expenses could be used to fund more diver-based interventions, such as targeted urchin removal in areas identified by the model as having low kelp cover or being at risk. 
		
		Our solution can therefore contribute to a more cost-effective and responsive cycle of monitoring and management, enabling more efficient allocation of limited conservation funds to where they are most needed on the seafloor.

		
	Comparison with Existing Remote Sensing Methods

		MESMA (Bell et al.) as an established automated method providing valuable fractional cover.

			Our deep learning approach, trained on human-consensus labels from Floating Forests, differs fundamentally from MESMA's reliance on spectral unmixing and pre-defined endmembers. It learns spatial patterns and visual cues directly.

			Potential advantages:
			
				maybe different robustness to certain atmospheric/water conditions if visual patterns remain distinct, no need for per-scene endmember refinement once trained.

				Our deep learning model, trained on Floating Forest labels which included SLC-off imagery, demonstrated an ability to produce segmentations on corrupted data.

			Disadvantage: 
			
				Produces binary output (vs. fractional)
				
				reliant on the quality and nature of FF labels.

		Floating Forests

			Acknowledge FF provides validated labels (Rosenthal et al.).

			While FF provides historical labels, our work automates the process for new imagery, leveraging the past citizen science effort to build a predictive tool. 
			
			It also provides a consistent, algorithmic interpretation rather than relying on ongoing volunteer consensus for every new image."

		CDFW Aerial / UAV:

			While CDFW aerial surveys and UAVs provide valuable high-resolution snapshots, our satellite-based automated pipeline offers superior scalability for consistent, large-area regional monitoring due to the systematic and broad coverage of Landsat imagery.

	 		Our approach significantly lowers operational costs by eliminating the need for dedicated aircraft/drone flights, field crew deployment, and extensive manual image processing inherent in CDFW aerial and UAV-based mapping efforts.
			
			Leveraging the regular revisit cycle of satellites like Landsat, our automated method has the potential for more frequent assessments across broad areas compared to the often intermittent and logistically constrained survey schedules of aerial and UAV campaigns.

	
	Limitations of the Study

		The model's performance is inherently tied to the quality and characteristics of the Floating Forests labels. While validated, these citizen-science-derived consensus labels may contain inconsistencies or differ from expert delineations in some cases.

		The 30m resolution of Landsat 7 limits the ability to detect very small or sparse kelp patches.

		The current model provides a binary (presence/absence) output, which does not quantify sub-pixel kelp density or biomass like MESMA.

		The model was trained and tested on data from the Falkland Islands; its performance on other geographic regions with different kelp species, water conditions, or image characteristics would require further validation and potential retraining.

		The current model processes each image independently, without leveraging temporal information from sequential satellite passes, which could potentially improve accuracy and consistency.

	Future Work

		Applying and evaluating the pipeline on Landsat 8/9 or Sentinel-2 imagery (different spectral characteristics, higher resolution for S2).

		Expanding the training dataset with more diverse geographic regions and temporal coverage from Floating Forests or other sources.

		Exploring modifications to output fractional cover or a confidence score rather than just binary.

		Investigating the integration of temporal information into the model.

		Direct comparison of model outputs with contemporaneous high-resolution UAV or field data for more granular validation.

		Operationalizing the pipeline for near real-time monitoring in specific regions of management interest.

	
	Broader Implications and Conclusion

		This research successfully demonstrates the development and viability of an automated, end-to-end deep learning pipeline for segmenting kelp canopy from satellite imagery, transforming raw image inputs into actionable map outputs.
	
		The study highlights the powerful synergy achieved by combining large-scale, human-generated labels from citizen science initiatives like Floating Forests with the sophisticated pattern recognition capabilities of modern deep learning techniques, creating a valuable resource from crowdsourced effort.

		Ultimately, this approach has the significant potential to enhance the efficiency, scalability, and cost-effectiveness of kelp forest monitoring, thereby supporting more timely and targeted conservation and restoration strategies by enabling the reallocation of resources towards direct, on-the-ground interventions.

Conclusion (~1 paragraph)

	todo

References

	todo

Appendices (Optional)


Table A1: Comprehensive Evaluation Metrics for All Experimental Configurations on the Test Set.
All models are UNet architectures with the specified ResNet backbone. "Cleaned" data refers to preprocessed and normalized imagery. "Augmented" refers to the application of random data augmentations during training. IoU = Intersection over Union.

Run Name	Backbone	Data Preprocessing	Augmentation	Land Mask Applied	Optimal Threshold	IoU	Precision	Recall	F1-Score
18_og_noaug	ResNet18	Original	No	No	0.2929	0.3043	0.4192	0.5261	0.4666
18_og_noaug	ResNet18	Original	No	Yes	0.2929	0.3091	0.4286	0.5258	0.4723
18_clean_noaug	ResNet18	Cleaned	No	No	0.2404	0.4437	0.5874	0.6446	0.6147
18_clean_noaug	ResNet18	Cleaned	No	Yes	0.2404	0.4437	0.5874	0.6446	0.6147
18_clean_aug	ResNet18	Cleaned	Yes	No	0.2929	0.4983	0.6387	0.6939	0.6652
18_clean_aug	ResNet18	Cleaned	Yes	Yes	0.2929	0.4983	0.6387	0.6939	0.6652
34_og_noaug	ResNet34	Original	No	No	0.3091	0.3239	0.4645	0.5168	0.4893
34_og_noaug	ResNet34	Original	No	Yes	0.3091	0.3277	0.4727	0.5165	0.4936
34_clean_noaug	ResNet34	Cleaned	No	No	0.2768	0.4492	0.5827	0.6622	0.6200
34_clean_noaug	ResNet34	Cleaned	No	Yes	0.2768	0.4492	0.5828	0.6622	0.6200
34_clean_aug	ResNet34	Cleaned	Yes	No	0.3414	0.5028	0.6355	0.7066	0.6692
34_clean_aug	ResNet34	Cleaned	Yes	Yes	0.3414	0.5028	0.6355	0.7066	0.6692
50_og_noaug	ResNet50	Original	No	No	0.2485	0.3656	0.5178	0.5543	0.5354
50_og_noaug	ResNet50	Original	No	Yes	0.2485	0.3691	0.5252	0.5540	0.5392
50_clean_noaug	ResNet50	Cleaned	No	No	0.3293	0.4419	0.5782	0.6521	0.6130
50_clean_noaug	ResNet50	Cleaned	No	Yes	0.3293	0.4419	0.5782	0.6521	0.6130
50_clean_aug	ResNet50	Cleaned	Yes	No	0.3106	0.5010	0.6386	0.6993	0.6676
50_clean_aug	ResNet50	Cleaned	Yes	Yes	0.3106	0.5010	0.6386	0.6993	0.6676
