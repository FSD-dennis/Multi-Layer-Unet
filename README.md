here is the structure of our project



## Introduction
- image data is a (350, 350, 7) discrete int set
- label data is a (350, 350, 1) binary set

here is the explanation for the 7 layers in our satellite of tiff file
1. Short-wave infrared (SWIR)
2. Near infrared (NIR)
3. Red
4. Green
5. Blue
6. Cloud Mask (binary – is there cloud or not)
7. Digital Elevation Model (meters above sea-level)


1. Short-wave Infrared (SWIR)
Utility: SWIR can penetrate through smoke and thin clouds, making it valuable for mapping and monitoring wildfires and for geological mapping. **It is sensitive to moisture content in soil and vegetation, aiding in plant stress analysis and water management.** SWIR is also used in mineralogy to distinguish between different rock and mineral types.

2. Near Infrared (NIR)
Utility: **NIR is highly reflective in healthy vegetation**, enabling the discrimination of vegetation types and conditions. It is used in calculating vegetation indices, such as the Normalized Difference Vegetation Index (NDVI), which measures vegetation health and biomass. NIR can also help in distinguishing between water bodies and land, as water absorbs NIR radiation.

<img src="https://static.sciencelearn.org.nz/images/images/000/004/209/full/SUS_SEAS_ART_11_MonitoringMarineEnvironmentsWithDrones_InfraRedKelp.jpeg" alt="Infrared Imaging of Kelp" width="400" height="300">

**infrared and short-wave infrared can only show kelp in the shallow water**

3. Red, Green, Blue (RGB)
Utility: These are the visible spectrum layers that correspond to the **primary colors perceived by the human eye.** Combining these layers creates true-color images that represent the world as humans see it. These layers are essential for:
Visual interpretation of satellite imagery.
Monitoring changes in land use and land cover.
Identifying features like rivers, forests, roads, and buildings.


Cloud Mask (binary – is there cloud or not)
Utility: **The cloud mask layer is crucial for identifying and filtering out clouds in satellite images**, which is essential for accurate analysis of ground conditions. Clouds can obscure the surface, affecting the analysis of vegetation, water bodies, and other features. Using cloud masks can improve the accuracy of environmental monitoring, agricultural assessments, and other analyses by excluding cloudy pixels.

**this is a binary**

Digital Elevation Model (DEM) (meters above sea-level)
Utility: DEM provides elevation data, which is vital for:
**Modeling water flow and hydrology, including flood risk assessment and watershed management.**
Planning infrastructure and construction projects by understanding the terrain.
Conducting geological and soil erosion studies.
Supporting navigation and line-of-sight analyses.
Enhancing the interpretation of other satellite layers by providing context about the terrain.

**[There is an easy analysis based for the visualization ipynb](https://github.com/y1u2a3n4g5/Multi-Layer-Unet/blob/main/visualization.ipynb)** 

## Model Structure
1. Unet model for short wave infrared + near infrared, input size : (350, 350, 4) with the output (350, 350, 2)
2. Unet model for RGB, input size : (350, 350, 3) with the output (350, 350, 2)

here is a good map for explaination:
<img src = "https://miro.medium.com/v2/resize:fit:1400/format:webp/1*f7YOaE4TWubwaFF7Z1fzNw.png">

- DoubleConv2d
    - nn.Module.Conv2d
    - nn.BatchNorm
    - nn.Module.Conv2d
    - nn.BatchNorm
- Down2d
    - nn.Maxpooling
    - DoubleConv2d()
- Up2d
    - nn.Upsampling + nn.functional.pad
    - DoubleConv2d()
- BinaryOut
    - DoubleConv2d()

3. combine that two output together


Short wave + near infrared - cloud mask - above sea level

**dice loss function**

<img src = "https://cdn-images-1.medium.com/v2/resize:fit:1600/0*CdS_Y_yl-rsem3-X">

dot multiply for the nominator and add for the denominator



## References
[Artificial intelligence convolutional neural networks map giant kelp forests from satellite imagery](https://rdcu.be/dyWWZ)
