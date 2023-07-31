# Identifying Houses in Areas without Addresses via Satelite Imagery and Mask-RCNN
#### Bob Adams - July 2023
<div style="text-align:center;">    
    <figure>
        <img src="images/iss030e254053~large.jpg" alt="Corfu Satelite" width="700"/><br>
        <figcaption>Photo Credit: [NASA](https://images.nasa.gov/details/iss030e254053) (ISS030-E-254053 (22 April 2012))</figcaption>
    </figure>
</div>


## Problem Statement
<div style="text-align:center;">
Send me a post card!<br>
...OK, what's the address?
</div>

---

Though common throughout most western countries and localities, unified building address systems are not universally available.  From remote communities in Africa and Ireland to [Carmel by the Sea in California]('https://ci.carmel.ca.us/post/addresses#:~:text=A%20unique%20characteristic%20of%20Carmel,houses%20south%20of%2012th%20Avenue%E2%80%9D'), _descriptive addresses_ are common and often require local knowledge to navigate.  Guiding deliveries and guests to unaddressed locations often requires providing detailed directions or providing local help, though providing a GPS location via a 'pin' in a mapping software is common in some areas.

Several attempts to address this gap have been undertaken by overlaying the planet with a grid and assigning unique identifiers to grid references.  [Google's PlusCodes]('https://maps.google.com/pluscodes/') assigns alphanumeric codes, organized logically to grid space.  [what3words]('https://what3words.com/clip.apples.leap') assigns a combination of three unique words in multiple languages along a similar schema.

_***But what if you don't know the identifier and can't really find someone to ask?***_

I have personally run into this on two occasions while trying to find the location of a house we were staying at in Donegal County, Ireland - and a similar experience on the Greek island of Corfu.  I had a picture of the house and a rough location, but no address.  It's not always easy to find "Tom's house, down the way from the Pub" in the dark!  To solve this issue, I found myself leveraging satelite images - identifying distinct features in images and attempting to locate the house by physical relationship.  The material (color) and shape of the roofs visible in images taken from the ground (or from inside the property) were the most consistently useful datapoints.  I took a few wrong turns, but eventually found the destination.

### Goal
This project seeks to leverage publicly available satelite and aerial imagery of unaddressed areas to identify buildings, roof types and materials and provide users with a map narrowing down likely destinations.  In cases where the town/city designation, paired with the roof description is unique, the goal is to create a geolocation of of the building they are looking for.  Otherwise, this project seeks to reduce the number of wrong turns taken in unfamiliar, unaddressed areas.

***How might this be leveraged in practice?***

Once an area is processed, users could take the following steps to help identify their destination:
1. Type in the name of their destination's village or town (ex. Agios Matheos)
2. Review available images of the building and select a roof color (ex. Red Tile)
3. Provide a best guess of the roof style, designated by the number of roof ridges (ex. one ridge)

The model will then assess the aerial imagery availabe (in this implementation, sourced from Google's Earth API) to identify buildings in the specified area and return building locations (latitude and longitude) of buildings matching the provided description.

### Methodology
This project requires the usage of a wide range of libraries and methods within the geopspatial analysis and data science disciplines.  A brief overview of the stages involved is listed below with details of the project provided in relevant sections later in this document.

#### 1. Define Geographic Boundaries
<div style="text-align:center;">    
    <figure>
        <img src="images/geo_boundaries.png" alt="Geographic Boundaries"/>
        <figcaption>Geographic Boundaries for the Island of Corfu and the Municipality of Agros</figcaption>
    </figure>
</div>
Before capturing satelite images covering the Earth, setting fixed boundaries over which to search is key to capture relevant and useful images.  For this project, the Greek island of Corfu is of interest.  As Greece's 7th largest island (>610 sq. km), a secondary search area is defined by one of the island's 16 municipalities.  


Island and relevant municipal boundaries are sourced from the Greek Government in the form of shapefiles.  These are re-projected into a common map projection and filtered to create relevant boundaries (ex. coastline) over which to search.
For more details, see [here](###-geographic-boundaries).

#### 2. Source Aerial Imagery
<div style="text-align:center;">    
    <figure>
        <img src="images/zoom-in.gif">
        <figcaption>... Enhance ... Enhance ... Enhance ... Enhance...<br>Variable Zoom Rates.  Image Credit: Google Satelite Imagery</figcaption>
    </figure>
</div>

Satelite Imagery is captured using Google's Earth Engine API and Google Satelite imagery.  Images are captured in two offset overlapping grids within the designated geographic boundary to maximize coverage and capture of complete buildings within individual images.  To aid in locating houses geographically, images are named with the latitude and longitude of the center of the image.

For more details, see [here](###-satelite-and-aerial-imagery).

#### 3. Model Selection
Matterport's Mask-RCNN model is selected to take advantage of a high quality, open source instance segmentation Convolutional Neural Network, and to minimize training time and resource requirements.  A version of the model upgraded for compatibility with Tensorflow versions 2+ is sourced from a branch created by Adam Kelly (aktwelve) via github and run in Google Colab.

For more details, see [here](###-model-selection)

#### 4. Source and Prepare Training Data
The Spacenet V2 [dataset](https://spacenet.ai/spacenet-buildings-dataset-v2/) is leveraged as the primary training data for this project.  Specifically, the smallest dataset covering portions of Khartoum, Sudan is leveraged due to data size constraints and model fitting time requirements.  This dataset is ideal in training a model to segment instances of buildings from satelite images, as the images are of high quality and each are provided with an accompanying mask for each building in a .geojson format.  Additional datasets were sourced for consideration in transfer learning opportunities including roofline and roof type detection use cases.

For more details, see [here](###-source-and-prepare-training-data)

#### 5. Instantiate a Pre-Trained Convolutional Neural Network
The Mask-RCNN model is adjusted to be retrained on the SpacenetV2 data and scoped for an accuracy level necessary for a proof of concept.

For more details, see [here](### Modeling)

#### 6. Evaluate Model Outputs and Identify Roof Colors 
Model performance (mean Average Precision) is hindered by a version compatibility issue between the Mask-RCNN model and versions of Tensorflow currently available in Google Colab.  As a result, model weights cannot be accurately imported to the model which then yields near-zero precision of predictions.

For more details, see [here](#### Model Performance)

Logic is constructed to extract predicted building geolocations and average roof color of predicted masks.  This ensures that if the model can be instantiated and trained in a compatible environment, the core use cases of the project (identify buildings by roof color from satelite imagery) is enabled.

For more details, see [here]()


---
## Data Sourcing and Preparation
### Geographic Boundaries
[Notebook](./code/01_Shoreline_Boundary.ipynb)
#### Sourcing
Though it is possible to capture arerial and satelite imagery covering the entirety of the planet, it would not be useful (oceans and seas), nor practical (data volume) to do so.  Two methods are available to establish boundaries, whithin which imagery can be captured:
1) ***Bounded Box*** - A simple box of Latitude and Longitude minimum and maximum values.  The result would cover a rectangle (map projectios notwithstanding).  This simpler approach would work well if a segment of a city is the area of interest, for example.
2) **Geometrically Defined Boundary*** - Leveraging geometric data (such as the border of a country or a coastline) protects against capturing irrelevant images.  In the case of an island, restricting capture to images on the island - and not the sea is preferable.

This project leverages a _blended_ approach.  Shapefiles (defining geographic boundaries) are isolated to the area of interest.  The minimum and maximum latitude and longitude values of the bounding geomertry is used to instantiate the outer limits of the search grid.  Images are considered gridwise across the search grid.  At each point, if the center of the image is outside the established geographic boundary, the image is not captured.

This project (focused on the Greek island of Corfu) leverages shapefiles available at Geodata.gov.gr for [Country Borders](https://geodata.gov.gr/en/dataset/aktogramme/resource/1ba9f74e-eb7a-4d0d-8858-864218806dbc) [Municipalities](https://geodata.gov.gr/en/dataset/oria-demon-kapodistriakoi), respectively.  For clarity, municipalities leveraged are defined by the Kapodistrias plan.

#### Preparation
Shapefiles are imported using the GeoPandas library.  Adjustments are made to the map projection (crs) to ensure compatibility with common latitude, longitude coordinate systems.

Line segments provided in the shapefile (on the order of a kilometer in length) are combined into closed shapes (beaches form an island).  The greek border shapefile covers the entirety of Greece, including the mainland and its many islands.  As the area of focus is Corfu, the correct geometry object must be isolated.  Names are not provided, so a sorted list of geometries is created by enclosing area.  Corfu is the seventh largest island in Greece, which enables identification and isolation.

Municipalities similarly cover the entirety of Greece, but names are provided for reference and filtering.  Here 'Ν. ΚΕΡΚΥΡΑΣ' (the Greek named for the island "Kerkyra") is used to filter to municipalities on the island.  The municipality of 'ΑΓΡΟΣ' (Agros) is selected to enable further segmentation of the island if needed.

---
### Satelite and Aerial Imagery
[Notebook](code/01_Image_Sourcing.ipynb)
#### Sourcing
The core of this project is the use of high quality satelite or (consistently sourced) aerial imagery covering an unaddressed area.  After assessing imagery available from NASA and Google's Earch Engine, Earth Engine is selected due to ease of use and exceptional documentation.  This includes a Google Colab Setup Guide provided in an .ipynb notebook [here](https://colab.research.google.com/github/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb).  Google Earth Engine operates via API, and requires an account to be created and API keys to be generated and used for each session.

#### [Imagery Selection](code/01_Image_Sourcing.ipynb##-select-projection-and-imagery-from-open-source-google-earth-engine)
Once a notebook is connected to Google's Earth Engine, HTML maps can be generated and displayed in a similar manner to those available on Google Maps, including zoom and drag functionality.  The baseline map does not provide satelite imagery, but does highlight road networks, towns, and other road-atlas relevant features.  The [folium](https://python-visualization.github.io/folium/modules.html) library can be leveraged to source and overlay a wide array of images, including multiple satelite sources.  Map layers (leaflets) can be explored and sourced from [leaflet providers on Github](http://leaflet-extras.github.io/leaflet-providers/preview/).

Each available leaflet is explored at high levels of zoom in an effort to select a leaflet which includes high quality images at a low zoom level (houses can be seen clearly when looking at a small geographic area.).  ESRI Imagery, provided via arcgis, enables a deep level of zoom, but imagery over the Island of Corfu is a bit blurry (image below).  This may make identifying individual buildings more difficult for a neural network, as the boundaries are clearly defined at high levels of zoom.
<div style="text-align:center;">
    <figure>
        <img src=images/Agios_Matheos_ESRI.png width=400 alt='Agios Matheos - ESRI Imagery'>
        <figcaption>Agios Matheos - ESRI Imagery</figcaption>
    </figure>
</div>

Another common library which commonly leverages the Google Earth Engine API is the geemap library.  The [documentation](https://github.com/gee-community/geemap/blob/master/geemap/basemaps.py) for this library contains a reference to the Google Satelite imagery.  Though this imagery is not available through the leaflet sources mentioned above (as ESRI imagery), it can be called via the Google Earth Engine if specified.  As shown below (with the same zoom level and map boundary), Google's Satelite imagery is much clearer for this area at high zoom levels.  As a result, this is the visualization layer selected for satelite image capture for this project.
<div style="text-align:center;">
    <figure>
        <img src=images/Agios_Matheos_Google_Satelite.png width=400 alt='Agios Matheos - Google Earth Imagery'>
        <figcaption>Agios Matheos - Google Earth Imagery</figcaption>
    </figure>
</div>

#### [Image Capture](code/01_Image_Sourcing.ipynb##-capture-small-scale-images)
> Disclaimer: This project is undertaken for exploratory and educational purposes.  Commercial use of images sourced through Google Earth Engine or other sources is generally prohibited.  Special limitations on use of specific leaflet layers may bear additional usage restrictions.

With geographic boundaries established and an imagery leaflet layer selected, the next step is to collect a gridwise sample of satelite images to serve as evaluation data for modeling (the area over which to identify and categorize buildings).  The first stage is defining the zoom level for the images to be captured.  Each leaflet image layer has a fixed maximum zoom limit.  This is generally the maximum useful zoom level to maintain resolution. 

> Though I am no expert in digital photography, one can think of this like older digital cameras - users can zoom in physically (the lenses on the camera move) without sacrificing the quality of the image captured.  Digital zoom can extend this range (the lenses no longer move), but the image resolution is compromised... the camera effectively "stretches" the pixels it reads to fill the desired frame.  This is also how 'pinch to zoom' works on smartphone cameras.

Here the maximum zoom level is selected (18 for Google Satelite Imagery) to maximize pixel density over a small area.

A frame size of 320x320 pixels is used for the images captured.  This may seem small, but this is chosen for three purposes:
1. Convolutional Neural networks are generally well suited for smaller images - this project also aims to identify individual buildings from the image.  Tuning the size of the images into digestable chunks is hypothesized to yield better accuracy and model performance (it is likely to find 10 houses from a small image than 100 houses in a larger image)
2. Geographic Area Covered.  At a zoom level of 18 and 300 pixels in height, each frame covers approximately 0.00128 degrees latitude... or 130 meters.  A lot of small houses can fit into an area roughly twice the size of a football pitch.
3. Mask-RCNN (the selected model for evaluation purposes) requires that each image be of dimensions which can be divided by two at least six times. (320 > 160 > 80 > 40 > 20 > 10 > 5)

At this image size, a single gridwise map covering the island of Corfu would result in 118000 images being considered or captured.  For the municipality of Agros, this is closer to 4000 images.  

Furthermore, images must be captured in two overlapping grids to maximize the instances of complete buildings being present in images.  A illustrative process flow of the image capture procedure is provided below.
<div style="text-align:center;">
    <figure>
        <img src=images/grid1.png alt='Initial Grid Capture'>
        <figcaption>1. Primary images are captured to create a grid covering the island boundaries within the maximun defined extents (cardinal directions) of the island.  Representative primary images boundaries are roughly identified by blue boxes.</figcaption>
    </figure>
    <figure>
        <img src=images/grid2.png width='400' height='400' alt='Secondary Grid Capture'>
        <figcaption>2. A secondary grid of images are captured within the same boundaries with a one-half width and height offset to the primary grid.  Any buildings which were divided into multiple images in the primary grid are captured entirely within one image on the secondary grid.  Representative primary images boundaries are roughly identified by orange boxes.</figcaption>
    </figure>
    <figure>
        <img src=images/grid_coastline.png width=400 alt='Coastline Control During Grid Capture'>
        <figcaption>3. Geographic Boundary Control - As each section of the grid is instantiated, the geographic boundary is considered prior to image capture.  Here the coastline is represented by a dashed green line.  If the center of the grid falls inside the geographic boundary (a closed shape), the image is captured.  Otherwise the image is skipped and the process moves to the next grid point.</figcaption>
    </figure>
</div>
    
Capturing each image requires five steps, accomplished in a single function:
1. Identify wither the center of the image is within the designated boundary geometry
2. Call Google Earth Engine API with the designated latitude/longitude, image size and zoom level - this returns an html window
3. Save the HTML window in a temporary folder - leveraging the latitude and longitude of the image in the file name
4. Call the imgkit library to read the html object and save the output as a .png file
5. Delete the temporary HTML file to save space.

Using the above process, thousands of high quality, relevant satelite and aerial images of consistent size and are captured quickly and available for modeling.  Some examples are below:
<div style="text-align:center;">
    <figure>
        <img src=images/39.70193999999998_19.71912999999999_sat.png width=400 alt='Hamlet, Agros'>
        <figcaption>Farming Hamlet - Agros, Corfu</figcaption>
    </figure>
    <figure>
        <img src=images/39.70433999999997_19.69832999999999_sat.png width=400 alt='Chorepiskopi Town'>
        <figcaption>Chorepiskopi</figcaption>
    </figure>
    <figure>
        <img src=images/39.70206892455841_19.718180550436898_sat.png width=400 alt='Agros, Agros'>
        <figcaption>Agros, Corfu</figcaption>
    </figure>
</div>

***Note*** During image extraction, rate limits resulting in 'unknown network errors' occurred after approximately 1000 image loads.  For larger areas, a larger window size or batching within the area of interest is recommended.

---
### Model Selection

The goal of this project is to leverage the images captured above to identify instances of buildings - and details about these buildings.  For this application, a Convolutional Neural Network is the best candidate machine learning model type.  Specifically, a Convolutional Neural Network which is designed to identify one or more instances of one or more classes of object within an image.  These model

<blockquote>
    <strong>A brief summary of Neural Networks and Convolutional Neural Networks</strong><br>  
    <p> - Machine Learning models consider input data and known answers (training data) to iteratively learn the rules which associate the inputs and resulting answers.  These rules can then be used to predict answers from unseen data.<br><br>
    - Neural Networks are a collection of relatively simple models (nodes) organized in successive layers to accomplish the same type of iterative learning using training data.  In most implemenetations, the nodes of each layer feed their output to successive layers in the model.  Nodes in deeper layers are not considering the input data directly, but <i>a representation of the input data generated by upstream layers</i>.  This enables Neural Networks to assess more complex relationships between the inputs and answers.<br><br>
        - Convolutional Neural Networks introduce <i>Convolution Layers</i> to a Neural Network.  These layers enable the model to assess arrays of numbers (often images) organized in multiple dimensions.  Boiled down, Convolution Layers 'look at' a small subset of the 'image' similar to the gridwise image capture approach above and calculate ouptputs to create a new representation of the input data.  One key feature is that these layers enable the network to identify features of an input without respect to their position.  Convolutional Neural Networks are the cornerstone of computer vision in applications like image classification (cat or dog?), object detection (lines on the road in self driving cars), and instance segmentation (identifying and reading a roadsign).<br><br>
</blockquote>

<a href="https://github.com/matterport/Mask_RCNN">Mask-RCNN</a> is a pre-trained Convolutional Neural Network used for Instance Segmentation.  One guiding principle of Neural Networks is that as the rules you are attempting to learn grow in complexity, the amount of data needed to train the model increases as well.  Mask-RCNN has been trained on a range of very large datasets, which enable it to break down input images efficiently and identify instances of objects.  It can also be retrained to find new 'classes' of objects (like buildings) reasonably quickly - as the internal rules of breaking down an image have already been learned.  This makes Mask-RCNN a great model to leverage when seeking to identify individual houses from satelite imagery! <br><br>
        Mask-RCNN's output when considering an image of a street below provides an illustration of both instance identification (individual boxes for instances of each type of object identified) and masking (the highlight overlay of the pixels that make up each object.)
<div style="text-align:center;">        
    <figure>
        <img src=https://github.com/matterport/Mask_RCNN/raw/master/assets/street.png width=600 alt = "Matterport Mask-RCNN Street View">
        <figcaption>Sample Mask-RCNN Output.  Credit: Matterport.  <a hreff = "https://github.com/matterport/Mask_RCNN"> Source Link</a></figcaption>
    </figure>
</div>
One note is that Mask-RCNN is not directly compatible with the current versions (2+) of the Tensorflow Library (which is used to run a wide range of neural networks, especially in Google's Colab environment).  For this project, a version of Mask-CNN upgraded to work with Tensorflow v2+ is sourced from Adam Kelly (aktwelve) <a href="https://github.com/akTwelve/Mask_RCNN">via Github</a>.  Many thanks to Adam and their collaborators for their work in ensuring this model is still usable!  Additional adjustments made to the model are listed in the Modeling section below.

---
### Source and Prepare Training Data

[Notebook](code/04_Training_Data.ipynb)

#### Sourcing
Machine Learning models make predictions on a like-for-like basis with the answers ("targets") they were trained on.  A model asked to predict a house price will need to be trained with house prices as a target.  A model trained to identify pictures of cats vs. dogs will need to be trained on labeled images of cats and dogs (and because the task is more complex, a lot more photos.).  This project seeks to generate masks (the highlighted sections) which comprise buildings from aerial photos.  Therefore, a dataset comprised of aerial images and <i>individual building masks</i> will be required for training.  Thankfully, this data is publicly available from Spacenet.ai.

Spacenet V2 is a publicly available dataset from [Spacenet.ai](https://spacenet.ai/about-us/).  This data is commonly used in Spacenet-operated geospatial machine learning challenges similar to those operated on kaggle.com.  The Spacenet V2 [dataset](https://spacenet.ai/spacenet-buildings-dataset-v2/) covers four cities and provides maps of category-masked images, including maps for buildings viewed from satelite photography.  This data is ideal to train a Convolutional Neural Network to identify, locate and mask individual buildings for this project.

Spacenet V2 data is available for download via Amazon S3 bucket or from [Radiant MLHUB](https://mlhub.earth/data/spacenet2).  For this project, I downloaded the Khartoum dataset from Radiant MLHUB.  It is the smallest of the four available datasets, which enables faster training times in line with the time constraints for this project.  Examples of the pre-processed images are below.

<blockquote>
    <strong>A note on extended use cases:</strong><br>
    Additional data is available from the Spacenet V2 dataset covering cities including sections of Las Vegas, Paris and Shanghai.  This data could be preprocessed and leveraged to train Mask-RCNN in combination with the Khartoum data leveraged for this project to improve accuracy of predictions across a wider range of use cases.  <br><br>
    In addition, data is sourced from <a href = "https://github.com/loosgagnet">loosgagnet via github</a> which could also be used to refine the model and open transfer learning opportunities.  Details on three such datasets imported but not utilized in the project directly are below:<br><br> 
    <strong>Building Detection and Roof Type Data</strong> from <a href="https://github.com/loosgagnet/Building-detection-and-roof-type-recognition">loosgagnet - Building-detection-and-roof-type-recognition</a>

This dataset was created by the research team leading: [A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image - Alidoost & Arefi, 2019](https://link.springer.com/article/10.1007/s41064-018-0060-5)

It consists of two training datasets for the above named paper:
1. **Building Detection** - includes three classes of urban objects such as buildings, roads and trees (4800 IR-R-G images per class, after data augmentation)
1. **Roof-Type Detection** - includes three classes of roofs such as flat, gable and hip roofs (4800 IR-R-G images per class, after data augmentation)

Citation: Fatemeh Alidoost, Hossein Arefi; “A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image”, PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science, December 2018, Volume 86, Issue 5–6, pp 235–248, https://doi.org/10.1007/s41064-018-0060-5 <br><br>
    <strong>Roofline Extraction Dataset</strong> from <a href='https://github.com/loosgagnet/Roofline-Extraction'>loosgagnet Roofline-Extraction via Github</a>

This dataset was created by the research team leading: [2D Image-To-3D Model: Knowledge-Based 3D Building Reconstruction (3DBR) Using Single Aerial Images and Convolutional Neural Networks (CNNs)]('https://doi.org/10.3390/rs11192219')

It consists of cropped images, nDSMs, and roof elements including three classes of eaves (red), ridges (green) and hips (blue)<br><br>
    
<div style="text-align:center;">        
    <figure>
        <img src=images/loosgagnet_roofline.png><img src = images/loosgagnet_roofline_mask.png>
        <figcaption>loosgagnet Roofline Extraction - images processed for this project</figcaption>
    </figure>
</div>
    
---
Citation: Fatemeh Alidoost, Hossein Arefi, Federico Tombari; “2D Image-To-3D Model: Knowledge-Based 3D Building Reconstruction (3DBR) Using Single Aerial Images and Convolutional Neural Networks (CNNs)”, Remote Sens. 2019, 11, 2219, [https://doi.org/10.3390/rs11192219]('https://doi.org/10.3390/rs11192219')
</blockquote>

#### Pre-Processing SpacenetV2 Data
SpacenetV2 Data is available in tar.gz files for each city.  Once extracted, the data requires significant pre-processing before being leveraged in modeling.  Details of the cleaning process are provided below and handled sequentially in the linked [notebook](code/04_Training_Data.ipynb).  Mustafa Aktas' <a href='https://github.com/Mstfakts/Building-Detection-MaskRCNN/tree/master'>Building Detection with MaskRCNN project on github</a> is leveraged as reference on starting approaches to some of the problems tacked during data preparation and model instantiation.  Many thanks to Mustafa for blazing this trail!

##### File Organization
Initial file directories are laid out such that images and image labels are stored in separate directories for each image. Ex. img1 (multiple images) and img1-labels (geojson) are stored individually in unique directories.  Within each image file directory, there are four distinct image bands or types (all in 16 bit .tif format) MS, PAN, PS-MS, PS-RGB
> PS-RGB format is the closest image band to the photos which will be assessed (captured via Earth Engine API above).  These are used moving forward.  PS-RGB files and associated .geojson files are moved to combined images, and labels directories, creating a single directory for images and an adjoining directory for their labels.

##### File Indexes
File names are unique, but include gaps in the indices within the source file (image and label 120-123 exist, but 124 does not.)  As these will be used to identify and select images and masks during training, a sequential and complete index list is preferable.
> The file indices of the image files are sorted (as strings) and a 1-indexed range is created to replace these indices.  The numerical file index is leveraged to ensure the images and their labels are converted to the same new index. Ex. 1000 > 1, 1001 > 2, etc.

Labels (building masks) are stored as .geojson objects - these will need to be reformatted as images for validation and leveraged to create individual building masks (arrays) for modeling (ex. Image 1, Building 1 is made up of these shaded pixels).
> The rioxarray and rasterio libraries are leveraged to create single building masks for each image as .tif files in a maps directory.  These are later imported via the OpenCV library and saved as .png files (viewable).  Example below.  It should be noted that individual buildings are not segmented in these mask images.  They are useful for inspection and validation only.  The model will require individual masks for each building for training purposes.  These are generated in line during modeling.

<div style="text-align:center;">
    <figure>
        <img src=data/training_data/spacenet_v2/images/img1002PS-RGB_.png width=400 alt="SpacenetV2 Image 1002"> <img src=data/training_data/spacenet_v2/maps/1002mask_.png width= 400 alt = 'SpacenetV2 Image 1002 Combined Mask'>
        <figcaption>SpacenetV2 Khartoum Image 1002 and matching mask for all combined buildings (sourced from geojson)</figcaption>
    </figure>
</div>

##### Image File Types
Image files are provided in a 16-bit .tif format.  These are not directly viewable by most systems.  A .png copy must be created for visualizaton and modeling (.png files can be converted to arrays for modeling, while .tif files cannot be converted directly).  .tif files must be retained, as the geographic metadata included in these files is key to matching the image with the geojson data in the labels when creating building-specific masks.
> The imageio and OpenCV libraries are leveraged to open each .tif image and save a .png copy.

---
### Modeling
[Notebook](code/05_Building_Detection.ipynb)

As stated above, this project leverages the Mask-RCNN pre-trained model, led by Waleed Abdulla and originally poblished on [github].  This model is adept at identifying instances of defined and trained object classes (ex. person, car, balloon, etc.).  Identified instances are fitted with a boundary box and a pixel-wise mask which can be overlaid on the original image to highlight the pixels found to make up the object instance.

The process of instantiating a pre-fit model is unique for each model.  As such, this notebook is based on the train_shapes ipynb instructional file available on Matterport's Mask R-CNN [Github Repository](https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb).  This provides the baseline for imports, configuration and data split.  The model in use has been made compatible for use with Tensorflow 2.0+ versions by Adam Kelly ([github repository](https://github.com/akTwelve/Mask_RCNN)).


##### Model Parameters

The model trained is based on the following parameters:
* <strong>Initial Loaded Weights = Imagenet</strong> - Model weights calculated based on training on the Imagenet dataset are used as a baseline for all layers and retained in all frozen layers of the model throughout training (here, 'heads' layers are retrained only.)
* <strong>BACKBONE = resnet50</strong> - resnet50 is a 50-layer deep pre-trained Convolutional Neural Network available through Tensorflow Keras.  This is a popular pre-trained model which balances model performance with training time requirements.
* <strong>GPU_COUNT = 1</strong> - Following the upgrade to Tensorflow version 2+, Mask-RCNN tends to encounter data loading and distribution issues when working with multiple GPUs.  This slows model training and prediction time, but should not have any negative performance impacts on prediction accuracy.
* <strong>IMAGES_PER_GPU = 1</strong> - Each step will consider images in batches of 8
* <strong>NUM_CLASSES = 2</strong> - For this project, the goal is to identify instances of buildings.  This requires a building class and a background class.
* <strong>LEARNING_RATE = 0.01</strong> - This is a high learning rate (which controls how aggressively the model adjusts learned weights to correct for errors).  This is selected due to the model training time requirements of the project.  Ideally, this would be a smaller value on the order of 0.001.
* <strong>STEPS_PER_EPOCH = 100</strong> - The model considers 100 batches of images in each epoch before saving weights and continuing training.  Training time varies for this model based on image complexity - at approximately 80 minutes of Google Colab GPU Runtime per 200 images.  Session timeout times necessitate a smaller step count per epoch to enable training to progress without losing significant progress.
* <strong>EPOCHS = 20</strong> (18 Completed) - This is a proof of concept model training and is not expected to exactly match buildings.  For a model going into production (not leveraged for a personal project), 200+ epochs with larger batch sizes and a more diverse range of augmented training data should be considered.

##### Building Masks
At model fitting time, images (.png files generated earlier) are loaded.  In parallel, .tif images are loaded in combination with the matching .geojson label files to generate a stacked numpy array of individual building masks.  Mustafa Aktas' approach is leveraged as a baseline for this portion of image preprocessing <a href="https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/SpaceNet_train.py">(reference)</a>. This approach requires the use of the geoio library, which has not been updated since 2016 and is out of date with current versions of Python.  

Taking inspiration from Mustafa Aktas' work, this project leverages some of the image rasterizing logic used to create single .png image masks from the same data (leveraging the rioxarray library).  All versions of the code are retained but commented out in the linked notebook for reference to those interested in similar approaches.

The result of this code is a stack of binary arrays with the following dimensions (image_width, image_height, number_of_buildings).  Each building has its own unique array mask passed into the model.  This enables the model to identify the rules that create multiple buildings per image, while also enabling the model to predict multiple buildings in a simlar stacked array output.
<div style="text-align:center;">
    <figure>
        <img src=images/multi-mask356k.png alt='Khartoum 356'>
        <img src=images/multi-mask911k.png alt='Khartoum 911'>
        <img src=images/multi-mask814k.png alt='Khartoum 814'>
        <figcaption>Images and Associated Masks for a sample of Khartoum Buildings</figcaption>
    </figure>
</div>
<blockquote>
    <strong>How to interpret these images?</strong><br>
    Here the masks (white sections) are generated for each building sequentially based on the order they are recorded in the geojson file.  These are stored in an array of the same size of the image (650x650).  Each array is then stacked on top of each other, creating a 3-dimensional array.  If all of the masks were 'added' together, they would form an image similar to the unified black and white mask shown in the data sourcing section.  The Mask-RCNN model includes a visualization library which converts the background to blue and overlays images similar to pages of film used on an overhead projector.  The buildings rendered in darker blue are just 'farther back' in the three-dimensional array.  Pretty cool!
</blockquote>


##### Adjustments to Mask R-CNN
In addition to the excellent work done by Adam Kelly and team to upgrade Mask-RCNN for Tensorflow 2+, several adjustments needed to be made to the model locally to enable the model to train on this particular dataset.  Details are listed below.
1. Update utils.py (294) prepare function to ensure generation of uniquely indexed images for both training and validation datasets (replaces 0-indexed sequential reference)
1. Update model.py to read a consistent class list (was previously dependent upon a numeric index of images, always 0 indexed.)
1. Update utils.py to handle an upgrade of the skimage library which interferes with mask resizing (where masks are stored as boolean arrays).  Cite: https://stackoverflow.com/a/73759783
1. Update model.py (2156) to change selected keras optimizer from 'optimizer = keras.optimizers.SGD(...' to 'optimizer = keras.optimizers.legacy.SGD(...' to handle upgrades made to tensorflow.  Cite: https://stackoverflow.com/a/75596562 in reference to [Tensorflow Release Notes](https://github.com/tensorflow/tensorflow/releases)
1. Update model.py at line 2362 to set workers=1 when calling keras.fit().  Multiprocessing with current Tensorflow versions and this model causes the fit process to hang (potentially indefinitely).  Cite: https://github.com/matterport/Mask_RCNN/issues/2696#issuecomment-1066224728

#### Model Performance
Following 18 epochs of of training using SpacenetV2 data (trainig the model 'heads' layers to recognize and identify instances of buildings), a mean average precision (mAP) of predictions was calculated (with an intersection over union ratio of 0.5) at 0.0%.  This indicates the model was not able to accurately identify any buildings when considering 10 images and maps from the validation dataset (of SpacenetV2).  A random sample image from the validation dataset was plotted, first considering the known building locations and masks and compared to the model outputs.
<div style="text-align:center;">
    <figure>
        <img src=images/validation_actual.png alt = 'SpacenetV2 Validation Image - Known Mask', width=300> <img src=images/validation_predicted.png alt='SpacenetV2 Validation Image - Predicted Building Masks' width = 300>
        <img src=images/validation_actual_2.png alt = 'SpacenetV2 Validation Image - Known Mask', width=300> <img src=images/validation_predicted_2.png alt='SpacenetV2 Validation Image - Predicted Building Masks' width = 300>
        <figcaption>Actual (left) and Predicted (right) Building Instances from SpaceNetV2 Images</figcaption>
    </figure>
</div>

***Why are the model results so inaccurate?***
Other recent users of the Mask-RCNN Model have reported similar model outputs, even on demonstrative model training and evaluation workflows available in the model itself ([example](https://github.com/matterport/Mask_RCNN/issues/2684)).  Users have in some cases been able to instantiate and train the Mask-RCNN model effectively by running the model in an environment with earlier versions of Tensorflow installed (namely <= Tensorflow 2.5).  Mask-RCNN relies on the Tensorflow and Keras libraries for a majority of the model components.  One user (Avinash) [identified](https://github.com/matterport/Mask_RCNN/issues/2684#issuecomment-1621635909) that the newer versions of Tensorflow cause the load_weights function to instantiate Mask-RCNN with incorrect weights throughout all layers.  Retraining the model 'heads' on new trainig data while leaving the hidden layers of the neural network frozen to the pre-trained weights results in highly inaccurate predictions (in this case, either no buildings or a large number of single-pixel or non-dimensional building instances are identified.).

Tensorflow requires access to GPUs to function properly.  As I do not have a GPU on my machine at the time of this project, Google's Colab environment is used for the modeling portion of this project.  Google Colab is built to run models such as this using runtimes which include GPUs.  As of July 2023, the earliest version of Tensorflow available for usage in Colab is [2.8](https://github.com/googlecolab/colabtools/issues/3663#issuecomment-1542884685) (with a nominal version at 2.12).  As a result, the model would need to be instantiated, trained and evaluated on a system which is set up specifically to run an earlier version of both Tensorflow and Keras.

### Model Predictions and Building Characteristics
#### Building Locations
Leveraging the prediction outputs from the Mask-RCNN model, it is possible to identify and map the center of each predicted building to a geographic location (in concert with data stored in the image file name.).  This is accomplished by identifying the center of the bounding box predicted by the model for each predicted object.  Effectively, the minimum and maximum x and y values are leveraged to calculate the center point of the detected object.  Note: this is not the centroid of the mask object, but the average x-y extent of the identified object.  This is expected to be more accurate, given the shape of the mask objects predicted may not exactly match the shape of the building structure.

As the x, y coordinates are zero-indexed at the bottom-left corner of each image and calculated as pixels, the coordinates are mapped to the image geolocation using the following logic (see [here](code/05_Building_Detection.ipynb###-model-results-examination) for details.):
1. The x and y coordinates are dividided by the image size (320), resulting in a 0-1 range for each coordinate
2. 0.5 is subtracted from each coordinate, shifting the plane to the center.  Now the bottom left corner (0,0) is cacluated as (-0.5, -0.5)
3. Coordinates are multiplied by the scale factor between the image and geography (0.0017, 0.00128).  This is calculated manually during the image capture process.
4. Finally, the Longitude, Latitude stored in the image filename at capture is added to the x and y coordinates, respectively to identify the geolocation of buildings in the same map projection as the one used to capture the image.

#### Building Roof Colors
Leveraging the mask object generated for each predicted building, it is possible to identify the average color of the image area which overlaps the building mask.  In effect, this returns a color family for the roof of identified buildings.  This is accomplished using the following logic flow in the extract_roof_color function (see [here](code/05_Building_Detection.ipynb###-model-results-examination) for details.):
1. The individual building masks are isolated and processed in a loop (multiple building masks are expected to be generated from each image).  Here, each mask is stored as a (W x H) array where pixels covered by the mask are 1, other pixels are 0.
2. A copy of the original image (stored as a (W x H x 3) array) is created.  Here the color bands for Red, Green, and Blue are separated into stacked arrays, all with the same height and width as the image.
3. For each color channel (red, green, blue), the image mask array is multiplied by the color channel array.  The result is 0 across red, green, and blue channels for all pixels where the mask does not overlap.  Where the mask does overlap with the image, the original image is retained.
4. For each color channel in the resulting array, the average numeric value (0-255) is calculated and stored in a running list of building mask instances (id, (R, G, B)).

## Conclusions and Recommendations
This project shows that it is feasible to leverage publicly available imagery and pre-trained machine learning models to identify and segment buildings in remote areas.  High quality aerial images can be sourced for areas of interest (large or small) - image capture can be limited to either a boundary box or another pre-defined designation, such as city, island or state.  This information can be assessed by a high quality pre-trained instance segmentation neural network (Mask-RCNN) - for which a high volume of building identification training data is available via Spacenet.  Model outputs can be leveraged on creation to extend the bounding boxes and masks generated for each predicted building to identify the building geolocation and roof color.

#### Recommendations
1. ***Modeling Environment*** - The Mask-RCNN model should be instantiated on a local development environment built to run Tensorflow version 2.2-2.5.  This requires the usage of a GPU.  A long window for training time should be anticipated, as the Tensorflow 2+ versions of the Mask-RCNN do not perform well when multi-threading between multiple GPUs.  Parallelization is not currently a viable option for time optimization.
2. ***Training Data*** - In addition to the Khartoum data processed for this proof of concept, the available SpacenetV2 data for Paris, Las Vegas and Singapore should also be processed and leveraged to train the Mask-RCNN model following the same workflow described in this project.  This would enable the model to identify a potentially larger range of building types with a higher level of accuracy.  Furthermore, training data augmentation should be considered to maximize the value extracted from each training image.
3. ***Transfer Learning*** - Training data pertaining to roof designs is sourced early in this project and is an excellent candidate for transfer learning with a secondary machine learning model (considering the information generated by Mask-RCNN as an input).  By capturing building images as the original image where it overalaps with a proposed building mask from Mask-RCNN, Buildings could be further classified by the number of roof peaks or by roof type (ex. Gable Roof, Flat Roof, or Hip-Valley).  This would bring increased value to end users who are likely to want to further refine their results by roof characteristics beyond color.  Two datasets are linked and sourced in this repository under data/training-data which are worth exploring for this purpose.
4. ***Visualization and User Interface*** - As building geolocations are an output from the primary model, an integration back into a mapping software, such as google maps could be leveraged to 'pin' buildings on a map, enabling users to explore a filtered list of buildings visually.

## Credits and Acknowledgements
 - General Assembly Data Science Immersive Instructional Team (Tim Book, Katie Sylvia, Rowan Schaefer) - Thank you for your guidance througout this project
 - [Google's Earth Engine Colab Setup Guide](https://colab.research.google.com/github/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb)
 - [leaflet providers on Github on available map layers](http://leaflet-extras.github.io/leaflet-providers/preview/)
 - [Geemap Library Community on Google Satelite Imagery](https://github.com/gee-community/geemap/blob/master/geemap/basemaps.py)
 - [Jarekk on imgkit image sizing](https://github.com/jarrekk/imgkit/issues/18#issuecomment-359372163)
 - [Geodata.gov.gr](https://geodata.gov.gr/en/dataset/aktogramme/resource/1ba9f74e-eb7a-4d0d-8858-864218806dbc)
 - [Geodata.gov.gr](https://geodata.gov.gr/en/dataset/oria-demon-kapodistriakoi)
 - [Geopandas.org - User Guide ](https://geopandas.org/en/stable/docs/user_guide/io.html)
 - [Martin Fleischmann on usage of Polygonize](https://stackoverflow.com/questions/68312843/how-to-create-polygons-from-linestring-unsorted-list)
 - [Geopandas.org on reprojecting maps and geo-objects between different CRS](https://geopandas.org/en/stable/docs/user_guide/projections.html#)
 - [Stackexchange user2856 on reporjecting crs systems](https://gis.stackexchange.com/a/412821)
 - [Spacenet.ai](https://spacenet.ai/about-us/)
 - [SpacenetV2](https://spacenet.ai/spacenet-buildings-dataset-v2/)
 - [Radiant MLHUB - Sourcing SpacenetV2 Data](https://mlhub.earth/data/spacenet2)
 - [Stackexchange User snowman2 on rioxarray usage to extract masks](https://gis.stackexchange.com/a/346899)
 - [Mustafa Aktas - Building-Detection-MaskRCNN](https://github.com/Mstfakts/Building-Detection-MaskRCNN/tree/master)
 - [Adam Van Etten Medium.com on pre-processing SpaceNet data](https://medium.com/the-downlinq/creating-training-datasets-for-the-spacenet-road-detection-and-routing-challenge-6f970d413e2f)
 - [john-hen on conversion of 16 bit .tif files to .png](https://stackoverflow.com/a/56761935)
 - [loosgagnet Roofline-Extraction via Github]('https://github.com/loosgagnet/Roofline-Extraction')
 -- [2D Image-To-3D Model: Knowledge-Based 3D Building Reconstruction (3DBR) Using Single Aerial Images and Convolutional Neural Networks (CNNs)]('https://doi.org/10.3390/rs11192219')
 -- Citation: Fatemeh Alidoost, Hossein Arefi, Federico Tombari; “2D Image-To-3D Model: Knowledge-Based 3D Building Reconstruction (3DBR) Using Single Aerial Images and Convolutional Neural Networks (CNNs)”, Remote Sens. 2019, 11, 2219, [https://doi.org/10.3390/rs11192219]('https://doi.org/10.3390/rs11192219')
 - [loosgagnet - Building-detection-and-roof-type-recognition](https://github.com/loosgagnet/Building-detection-and-roof-type-recognition) 
 -- This dataset was created by the research team leading: [A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image - Alidoost & Arefi, 2019](https://link.springer.com/article/10.1007/s41064-018-0060-5)
 -- Citation: Fatemeh Alidoost, Hossein Arefi; “A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image”, PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science, December 2018, Volume 86, Issue 5–6, pp 235–248, https://doi.org/10.1007/s41064-018-0060-5
 - [Anbarasan Thangapalam on initial conversion test from .tif to png (8bit to 8bit)](https://stackoverflow.com/a/28872806)
 -[Matterport's Mask R-CNN [Github Repository]](https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/train_shapes.ipynb)
 - [Adam Kelly's updated Mask-RCNN Repo for Tensorflow 2.0 (github repository)](https://github.com/akTwelve/Mask_RCNN)
 - [Stackoverflow user DouL on extracting image portions from RCNN masks](https://stackoverflow.com/q/60001378)
 - [Mask-RCNN Issue Board on mini-mask usage](https://github.com/matterport/Mask_RCNN/issues/1967#issuecomment-796687115)
 - [Github User Avinash-218 on likely causes of Mask-RCNN Modeling issues with Tensorflow 2.6+](https://github.com/matterport/Mask_RCNN/issues/2684#issuecomment-1621635909)
 - [Jonathan Hui via Medium on mAP Metrics](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)