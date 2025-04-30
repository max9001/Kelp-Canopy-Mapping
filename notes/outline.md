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

results

	what did you find out / how did it work. very dry, numbers

discussion

	speculative -> what do the results mean. we think ____ happened because...

	discuss weaknesses / what could be improved
