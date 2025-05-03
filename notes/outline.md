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
		
		- In short, its due to an explosion of pacific purple sea urchin populations

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

	- existing digital data for kelp maps
	
		- California Department of Fish and Wildlife shapefiles derived from aerial surveys (https://wildlife.ca.gov/Conservation/Marine/Kelp/Aerial-Kelp-Surveys)

			- Labor-Intensive & Time-Consuming to produce

		- Using unoccupied aerial vehicles to map and monitor changes in emergent kelp canopy

			- extremely small sites "The surveyed sites varied in size from 0.16 to 1.48 km²

		- MEMSA LAndsat data 
		
			- MESMA relies on spectral physics and pre-defined endmembers

			- In satellite imaging, endmembers represent pure materials, such as kelp or water. Because water itself doesn't always look the same – it might be clear, cloudy with mud, or have bright sun reflections – there are many types of "pure water" endmembers. The correct endmember must be selected in order for the MESMA process to be accurate. the paper uses an automatic selection process to choose the correct endmember per image. This fixed, automated process could lead to an incorrect endmember selection due to noise or the sampling process not perfectly represent all water variations in that specific scene. 




	- Bridging the Data Gap: Citizen Science:

		
		- Floating forests is difference as the entire dataset was labeled by humans. The study showed that accurate kelp labels could be constructed from the consensus of multiple untrained participants. 

		- based on landsat 7 sattalite images of coastlines of claifornia and tasmania 
			
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

	- Remote Sensing Principles for Kelp Detection
    	
		- spectral properties of kelp enable aid in detection.

        - Near-Infrared (NIR) 

			- Near Infared measures wave lengths to large for us to see. This again is super advantageous, chlorophyll in vegetation reflects this wavelength super well. So, kelp, which is high in clorophyll, will be extremely visible. This will help the model “see” kelp easier than just by color

		- Short-Wave Infrared (SWIR) 
		
			- First, Short Wave Infared measures wave lengths too small for us to see. Whata good about short wave infrared? water heavily absorbs these wavelengths and the land highly reflects this wavelength This Creates a high contrast between land and water

    -   Briefly introduce common approaches:
        -   *Vegetation Indices (VIs):* Mention indices like NDVI. Note their common use for terrestrial vegetation but limitations over water bodies due to water's strong influence on NIR/Red reflectance, often making them less reliable for floating algae without careful calibration or specific formulations.
-   Landsat 7 ETM+ Context
    -   Describe the relevant sensor characteristics: Specific bands used (mention SWIR, NIR, Red for the FF visualization), spatial resolution (30m), temporal revisit cycle.
    -   Address the Scan Line Corrector (SLC) failure (post-May 2003):
        -   Explain the resulting data gaps (stripes).
        -   Crucially, state how the Floating Forests data you used handled this. (e.g., "The Floating Forests project processed both pre- and post-SLC-off imagery; how gaps were handled by volunteers or in subsequent processing before generating the labels used in this study is relevant context..." *OR* "This study utilized only pre-SLC-off data from Floating Forests..." *OR* "Floating Forests presented images with gaps, and volunteers classified around them...". Be specific based on your dataset's origin).
-   Established Automated Method: Spectral Mixture Analysis (SMA/MESMA)
    -   Explain the core concept of SMA: Modeling a pixel's spectrum as a linear combination of pure reference spectra ('endmembers').
    -   Introduce Multiple Endmember SMA (MESMA) as an advancement.
        -   Explain the need for *multiple* endmembers, especially for variable components like water (clear, turbid, shallow, sun glint etc.), citing the Bell et al. (2020) approach.
    -   Detail the specific MESMA implementation from Bell et al. (2020) as the state-of-the-art automated baseline:
        -   Uses a single, constant kelp endmember.
        -   Uses a library of water endmembers *unique to each scene*.
        -   Clarify the process: Spectra are extracted *automatically* from 30 *pre-defined, fixed locations* within each scene known to be consistently water-covered. This set of 30 scene-specific spectra forms the library for that image's MESMA run.
    -   Discuss Strengths: Provides quantitative sub-pixel *fractional cover*, can be linked to biomass (citing Bell et al. 2020 validation), suitable for long-term, large-scale automated analysis.
    -   Discuss Limitations relevant to your framing:
        -   Accuracy depends heavily on the *representativeness* of the chosen endmembers (both the static kelp and the scene-specific water spectra extracted from fixed points).
        -   The automated extraction from fixed points might not always capture the *full range* of water variability or could be affected by localized noise/haze at those points in a specific image, potentially impacting unmixing accuracy.
        -   Known struggles with very low fractional cover (<~20%, citing Hamilton et al. 2020, Cavanaugh et al. 2023).
-   Alternative Label Source: The Floating Forests Project
    -   Introduce Floating Forests as a large-scale citizen science project on the Zooniverse platform, specifically designed to generate kelp labels for Landsat imagery.
    -   Describe the volunteer task: Visually inspecting Landsat 7 image subsets (displayed using SWIR/NIR/Red bands) and manually tracing perceived kelp canopy borders.
    -   Explain the Consensus Mechanism: Emphasize that data quality relies on agreement among multiple (up to 15) untrained volunteers. The final label for a pixel is derived from this consensus (e.g., requiring >=4 votes, as determined by Rosenthal et al.).
    -   Summarize Validation Results: Cite the Rosenthal et al. study (preprint/report) confirming that this consensus approach yields kelp classifications with accuracy (measured by MCC) comparable to expert-derived methods, establishing the dataset's utility.
    *   Frame its Distinct Nature: Highlight that it relies on human visual pattern recognition and consensus judgment, rather than spectral physics and automated endmember extraction. This makes it a fundamentally different type of label source, potentially capturing different information or having different sensitivities to noise/ambiguity compared to MESMA.
-   Deep Learning for Image Segmentation
    -   Define Semantic Segmentation: The task of assigning a class label (e.g., kelp or background) to every pixel in an image.
    -   Introduce the UNet Architecture: Briefly explain its encoder-decoder structure with skip connections. Mention its demonstrated success in biomedical imaging and increasingly in remote sensing for precise localization of features.
    -   Explain Transfer Learning and Fine-Tuning:
        -   Define transfer learning: Using models (like ResNet-18, -34, -50) pre-trained on large, general datasets (ImageNet).
        -   Explain the benefit: Leverages powerful, pre-learned visual features, drastically reducing the need for labeled data specific to the target task, speeding up training, and often improving performance.
        *   Clarify Fine-Tuning: Explain that in this context, it involves initializing the encoder with pre-trained weights and then *unfreezing* and updating *all* weights (encoder and decoder) during training on the target (kelp) dataset, allowing the model to adapt fully.
    -   Explain Data Augmentation: Describe its purpose – artificially expanding the training dataset by applying random transformations (flips, rotations, brightness changes, etc.) to input images. Explain how this improves model robustness, generalization, and helps prevent overfitting, especially with limited or imbalanced datasets.













Methods (~3-4 pages) - *Needs high detail for reproducibility*

- since we know the pixel resolution where 1 pixel is 900 square meters in the real world we can now calculate the kelp ecosystems area in square meters.

-	3.1. Data Source and Study Area:
-	-	Dataset: Specify "Landsat 7 ETM+ data processed and classified by the Floating Forests citizen science project."
-	-	Region(s) & Time Period: Clearly define the geographic area(s) (e.g., specific coastlines in California, Tasmania?) and the date range of the images used in your study.
-	-	Input Data Format: Describe the satellite image inputs: Were they the 3-band JPEGs (SWIR/NIR/Red) used in the FF interface? What was the spatial resolution effectively used?
-	-	Ground Truth Generation: CRITICAL: Explain *exactly* how you created the binary ground truth masks for training/evaluation from the Floating Forests output. Did you use the raw volunteer polygons? Did you rasterize them? Did you apply the consensus threshold (e.g., pixels marked by >= 4 volunteers = kelp)? Justify your choice.
-	3.2. Data Preprocessing:
-	-	Image Tiling: Describe how the initial (potentially large) image subsets from Floating Forests were tiled for input into the UNet (e.g., 256x256 pixels, 512x512 pixels? Any overlap?).
-	-	Normalization: Detail how pixel values were normalized (e.g., scaled to [0, 1], standardized using mean/std dev?).
-	-	Data Cleaning: Mention any steps taken to remove problematic images/tiles (e.g., based on cloud cover flags from FF, visual inspection?).
-	-	Addressing Data Imbalance (Attempt): Describe the experiment of removing tiles with zero kelp pixels. Explain the rationale (reduce class imbalance). State clearly whether this subsetting was used in the *final* successful model training or just an initial experiment.
-	-	Train/Validation/Test Split: Clearly state how the tiles were divided. Was it random split? Percentage for each set? Was care taken to ensure spatial or temporal independence if necessary (e.g., putting tiles from the same original large image entirely in one set)? Provide the number of tiles in each set.
-	3.3. Model Architecture:
-	-	Base Architecture: State you used the UNet architecture.
-	-	Backbones: Specify you replaced the standard UNet encoder with pre-trained ResNet-18, ResNet-34, and ResNet-50 backbones. Mention weights were initialized from ImageNet pre-training.
-	-	Modifications: Explain any necessary modifications (e.g., adjusting the final convolutional layer to output a single channel for binary segmentation).
-	-	Comparison Model: Briefly mention the baseline UNet trained from scratch that was attempted.
-	3.4. Model Training:
-	-	Framework: Specify the deep learning framework used (e.g., PyTorch, TensorFlow/Keras).
-	-	Loss Function: State the loss function (e.g., Binary Cross-Entropy (BCE), Dice Loss, Focal Loss, or a combination like BCE + Dice). Justify the choice, especially if related to data imbalance (Dice/Focal are often better for this than plain BCE).
-	-	Optimizer: Specify the optimizer (e.g., Adam, AdamW, SGD) and the learning rate used (mention if a learning rate scheduler was employed, e.g., ReduceLROnPlateau).
-	-	Data Augmentations: List *all* augmentations applied during training (e.g., horizontal/vertical flips, rotations, scaling, brightness/contrast adjustments, elastic distortions). Specify the library used (e.g., Albumentations) and the probability of each augmentation being applied.
-	-	Training Setup: Mention batch size, number of epochs trained for. Briefly describe the hardware used (e.g., "trained on an NVIDIA RTX 3090 GPU").
-	3.5. Evaluation:
-	-	Metrics: Define the metrics used:
-	-	-	Intersection over Union (IoU) / Jaccard Index (Explain calculation: TP / (TP + FP + FN)). Standard for segmentation.
-	-	-	Dice Coefficient (Explain calculation: 2*TP / (2*TP + FP + FN)). Also standard, sensitive to positive class.
-	-	-	Pixel Accuracy, Precision, Recall (Optional but useful context).
-	-	Procedure: Explain that models were evaluated on the held-out test set after training converged based on validation set performance.
-	3.6. Testing Pipeline Implementation:
-	-	Describe the script developed for inference. Explain its inputs (directory of images, model weights file) and outputs (predicted binary masks saved as image files). Briefly mention key steps (loading model, image preprocessing identical to training validation, running inference, applying threshold if necessary, saving output).

Results (~2-3 pages) - *Focus on objective findings*
-	4.1. Dataset Characteristics: Provide summary statistics of your final training, validation, and test sets (number of tiles, approximate percentage of kelp pixels overall – highlighting the imbalance).
-	4.2. Model Performance Comparison:
-	-	Present a table comparing the performance metrics (IoU, Dice, etc.) of the UNets with ResNet-18, -34, and -50 backbones on the test set. Clearly identify the best-performing model.
-	-	Mention the performance (or failure to converge) of the UNet trained from scratch for contrast.
-	4.3. Impact of Data Augmentation:
-	-	Present a table or graph comparing the best ResNet model trained *with* versus *without* data augmentations, showing the improvement in key metrics (IoU, Dice).
-	4.4. Impact of Data Imbalance Strategy (if tested on final model):
-	-	If you tested the impact of removing no-kelp tiles on the final trained ResNet model, report the results here. Did it help, hurt, or make no difference compared to training on all tiles (perhaps with a suitable loss function like Dice)?
-	4.5. Qualitative Results:
-	-	Include several figures. Each figure should show:
-	-	-	(a) The input Landsat 7 tile (SWIR/NIR/Red).
-	-	-	(b) The ground truth mask derived from Floating Forests.
-	-	-	(c) The predicted mask from your best-performing model.
-	-	Select examples that showcase:
-	-	-	Good agreement / successful segmentation.
-	-	-	Segmentation of complex or sparse kelp patches.
-	-	-	Typical failure modes (e.g., missed detections, false positives, edge inaccuracies, confusion with clouds/glint if applicable).
-	4.6. Testing Pipeline: Briefly state the pipeline was successfully implemented and show one visual example of its input and output.

Discussion (~2-3 pages) - *Interpret and contextualize*

- My proposed solution removes the need for planes, image collection, and human annotators.

		- The money saved from this automation could be used to fund more divers to combat urchin populations

-	5.1. Summary of Key Findings: Restate the main outcomes – successful training of ResNet-UNet, identification of the best backbone, quantified improvement from augmentations, confirmation of transfer learning necessity.
-	5.2. Interpretation of Results:
-	-	Why was transfer learning crucial? Discuss how features learned on ImageNet likely provide a useful starting point for identifying textures/patterns relevant to kelp detection, even in satellite imagery, overcoming the limitations of the specific training dataset size.
-	-	Why were augmentations effective? Discuss how they helped the model generalize better and become robust to variations in appearance, orientation, and illumination within the satellite data, especially given potential inconsistencies in citizen science labels.
-	-	Discuss the performance trade-offs between ResNet-18, 34, 50 (e.g., did the largest model always perform best, or was there a plateau?).
-	-	Discuss the data imbalance issue. How well did your chosen loss function and/or augmentations handle it? If removing empty tiles didn't help, speculate why (maybe context from surrounding water is important?).
-	5.3. Comparison with Existing Methods:
-	-	Compare your deep learning approach conceptually to MESMA (Bell et al.) and the direct Floating Forests consensus map (Rosenthal et al.).
-	-	Advantages: Automation, potential for consistent pixel-level output across large areas once trained, ability to learn complex spatial patterns.
-	-	Disadvantages: Requires curated training data (dependent on Floating Forests quality), computationally intensive training, potential for 'black box' behavior, output is binary (unlike MESMA's fraction unless you modify the output).
-	5.4. Limitations of the Study:
-	-	Ground Truth Quality: Acknowledge the reliance on citizen science data (Floating Forests consensus) as ground truth, which has inherent variability and isn't perfect field validation (cite Rosenthal's MCC scores as context for its quality).
-	-	Landsat 7 Data: Mention limitations of the input data (30m resolution might miss small/thin kelp, SLC-off issues if relevant and not fully handled).
-	-	Model Generalization: Note that the model was trained and tested on a specific geographic region/time period; performance might vary elsewhere without retraining or fine-tuning.
-	-	Temporal Aspect: Acknowledge that this approach treats each image independently and doesn't incorporate temporal information, which could potentially improve results.
-	5.5. Future Work:
-	-	Applying the model to Landsat 8/9 or Sentinel-2 data (different spectral bands, higher resolution for S2).
-	-	Expanding the training dataset (more regions, longer time series).
-	-	Testing different model architectures or loss functions.
-	-	Incorporating temporal context (e.g., using LSTMs or Transformers alongside the CNN).
-	-	Validating model outputs against independent, high-resolution data (e.g., UAV data like in Cavanaugh et al. 2023, if available) or recent field data.
-	-	Investigating specific failure cases identified in qualitative results.
-	5.6. Broader Implications: Conclude by emphasizing the potential of combining deep learning with large-scale citizen science datasets for efficient, automated monitoring of critical ecosystems like kelp forests, contributing to ecological research and conservation efforts.

Conclusion (~1 paragraph)
-	Summarize the core problem (large-scale kelp mapping), the approach taken (deep learning on Floating Forests/Landsat 7 data), the key finding (success via transfer learning and augmentation), and the main implication (automated mapping is feasible and promising).

References
-	List all cited works (Bell, Cavanaugh, Rosenthal, UNet paper, ResNet paper, etc.). Use a consistent citation style.

Appendices (Optional)
-	Detailed hyperparameters, more qualitative result figures, code snippets.