1 Intro

    1.1 A Kelp Crisis
        
        California is currently experiencing a devastating die-off of its kelp forests, a crisis largely attributed to an explosion in pacific purple sea urchin populations. This urchin surge began after their natural predator, the sunflower sea star, suffered a catastrophic decline due to sea star wasting syndrome, which emerged around 2013. The disease pushed the sunflower sea star to the brink of extinction, leading to its current critically endangered status. With their primary predator virtually gone, purple sea urchin populations have exploded. These unchecked urchins, which feed on kelp, are now responsible for the destruction of vast kelp forests, with estimates indicating an 80% loss in Northern California since 2013 (Zuckerman, 2023).

    1.2 The Need for Intervention

        Intervention through manual urchin destruction by divers is possible, but the scarcity of these skilled individuals demands an efficient allocation strategy. The West Coast Region Kelp Team underscores this by calling for an efficient way to "conduct annual assessments of kelp forest ecosystem health" (Hohman et al., 2023, p. 5). The rationale is straightforward: areas identified with poor kelp health are likely to harbor high concentrations of urchins. These assessments would enable dive teams to strategically target urchin removal efforts in these compromised zones, thereby creating conditions more favorable for kelp recovery and regrowth. 

        A primary approach to determining kelp forest ecosystem health involves quantifying the amount of kelp visible on the ocean's surface. By measuring this surface kelp area and consistently tracking these measurements over successive years, managers can identify significant trends. For instance, a substantial reduction in the surface kelp area within a region that previously had extensive kelp beds is a strong indicator of a struggling ecosystem. Such temporal comparisons are therefore crucial for pinpointing areas most in need of intervention.

    1.3 Challenges with Existing Monitoring Methods and Why New Methods Are Needed

        Current kelp monitoring techniques, such as costly scuba diver surveys, struggle with scalability. This inherent limitation in spatial coverage per dive makes it impractical to assess vast coastlines comprehensively using only diver-based methods. Aerial surveys, while covering larger areas, share the issue of high cost and introduce the burden of repetitive human annotation. For example, the California Department of Fish and Wildlife annually conducts flights to photograph the ocean surface along the California coast. Scientists then manually analyze these images to perform kelp canopy mapping, a process that involves visually identifying kelp floating on the surface and meticulously calculating its total area, often expressed in square feet. This process is demonstrably expensive, time-consuming, and repetitive. The high costs associated with these traditional methods divert significant financial resources that could otherwise be allocated to direct intervention efforts, such as funding more diver teams for urchin removal. This financial constraint, coupled with the labor intensity, prompts an urgent need to explore whether this critical monitoring task can be automated using more cost-effective and readily available data sources.

    1.4 Existing Digital Data and Automation Techniques for Kelp Maps

        Several digital data sources and automation techniques exist for creating kelp maps, each with distinct advantages and limitations. The California Department of Fish and Wildlife, for instance, produces shapefiles derived from extensive aerial surveys. This process, detailed by MBC Applied Environmental Sciences (2017), involves capturing numerous photographs, manually creating photo-mosaics, and using GIS for georeferencing and area calculation, making it inherently labor-intensive and time-consuming to produce.

        More recently, Unoccupied Aerial Vehicles (UAVs) have been employed to map emergent kelp canopy at high resolution. Saccomanno et al. (2022) describe a workflow for creating kelp canopy maps from UAV imagery, valuable for local-scale restoration; however, a key limitation is the extremely small spatial scale, with surveyed sites noted as varying from only "0.16 to 1.48 km²."

        A more promising avenue for large-scale, cost-effective monitoring lies in utilizing Earth-observing satellite data, such as imagery from the Landsat program. This data is plentiful, regularly collected over vast areas by agencies like NASA, and is often available at low or no cost, offering a significant advantage over dedicated aerial or UAV campaigns. However, a primary drawback of satellite imagery is its inherent susceptibility to various forms of "noise" – such as atmospheric haze, sun glint on the water surface, and subtle water color variations – which can obscure or distort the appearance of kelp. For broader scale analysis, satellite data can be processed using techniques like Multiple Endmember Spectral Mixture Analysis (MESMA), as explored by Bell et al. (2020). MESMA leverages spectral physics and the use of pre-defined 'endmembers', which are pure materials like kelp or water, to estimate kelp cover. The idea is that any pixel is a linear combination of water and kelp. However, water itself doesn't always look the same – it might be clear, cloudy with mud, or have bright sun reflections – there are many types of water endmembers. The correct endmember must be selected in order for the MESMA process to be accurate. The paper uses an automatic selection process to choose the correct set of endmembers per image. The challenge with this fixed automation is its potential vulnerability to the aforementioned noise in satellite imagery; an incorrect endmember selection, possibly triggered by such noise or by the sampling process not perfectly representing all water variations in a specific scene, can lead to inaccuracies in the resulting kelp maps.

    1.5 Bridging the Data Gap: Citizen Science

        An alternative source of kelp data comes from the Floating Forests project, which offers a distinct method for generating kelp maps, addressing the challenges of interpreting noisy satellite imagery. 
        
        This citizen science initiative leverages historical data from Earth-observing satellite programs, specifically using Landsat 7 satellite imagery of the Falkland Islands. 
        
        The project demonstrated that accurate kelp maps could be constructed from the consensus of multiple untrained participants. 
        
        Studies by Rosenthal et al. (2018) have validated this consensus method, showing it produces kelp maps with significant accuracy. 
        
        The core strength of this approach lies in its inherent robustness to noise and visual ambiguities present in satellite data. 
        
        By aggregating the visual interpretations of many individuals, the consensus process effectively filters out isolated errors in judgment that arise from image noise, relying instead on shared human pattern recognition capabilities to identify kelp even under challenging visual conditions. 
        
        This human-in-the-loop approach offers distinct advantages in handling visual complexities—such as subtle water color variations, sun glint, or thin haze—compared to purely spectral techniques like MESMA, which may be more easily misled by such artifacts. 
        
        The digital kelp map data derived from Floating Forests, with its inherent noise resilience, is central to the present study. 
        
        It forms the basis that our deep learning models will be trained and evaluated on, to automate the process of generating kelp maps from widly available yet similarly noisy satellite imagery.

















Hohman, R., Bell, T., Cavanaugh, K., Contolini, G., Elsmore, K., FloresMiller, R., Garza, C.,
Hewerdine, W., Iampietro, P., Nickels, A., Saccomanno, V., & Tezak, S. (2023). Remote
sensing tools for mapping and monitoring kelp forests along the West Coast. National
Marine Sanctuaries Conservation Series ONMS-23-10. U.S. Department of Commerce,
National Oceanic and Atmospheric Administration, Office of National Marine
Sanctuaries.

    

Zuckerman, C. (2023, May 26). The vanishing kelp forest. The Nature Conservancy.
https://www.nature.org/en-us/magazine/magazine-articles/kelp-forest/




Saccomanno, V. R., Bell, T., Pawlak, C., Stanley, C. K., Cavanaugh, K. C., Hohman, R., Klausmeyer, K. R., Cavanaugh, K., Nickels, A., Hewerdine, W., Garza, C., Fleener, G., & Gleason, M. (2022). Using unoccupied aerial vehicles to map and monitor changes in emergent kelp canopy after an ecological regime shift. Remote Sensing in Ecology and Conservation, 9(1), 62–75. https://doi.org/10.1002/rse2.295


Source: STATUS OF THE KELP BEDS IN 2016:
	Ventura, Los Angeles,
	Orange, and San Diego Counties
	Prepared for:
	Central Region Kelp Survey Consortium and
	Region Nine Kelp Survey Consortium
	Prepared by:
	MBC Applied Environmental Sciences
	3000 Red Hill Avenue
	Costa Mesa, California 92626
	August 14, 2017




Bell et al. (2020) - Three decades of variability in California's giant kelp forests from the Landsat satellites (https://doi.org/10.1016/j.rse.2018.06.039)



Rosenthal, I. S., Byrnes, J. E. K., Cavanaugh, K. C., Bell, T. W., Harder, B., Haupt, A. J., Rassweiler, A. T. W., Pérez-Matus, A., Assis, J., Swanson, A., Boyer, A., McMaster, A., & Trouille, L. (2018). Floating Forests: Quantitative Validation of Citizen Science Data Generated From Consensus Classifications (Version 1). arXiv. https://doi.org/10.48550/ARXIV.1801.08522