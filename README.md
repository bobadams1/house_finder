# Identifying Houses in Areas without Addresses - House Finder
<p align="center">
    <figure>
        <img src="images/iss030e254053~large.jpg" alt="Corfu Satelite" width="700"/>
        <figcaption>Photo Credit: [NASA](https://images.nasa.gov/details/iss030e254053) (ISS030-E-254053 (22 April 2012))</figcaption>
    </figure>
</p>


## Problem Statement
Though common throughout most western countries and localities, unified building address systems are not universally available.  From remote communities in Africa and Ireland to [Carmel by the Sea in California]('https://ci.carmel.ca.us/post/addresses#:~:text=A%20unique%20characteristic%20of%20Carmel,houses%20south%20of%2012th%20Avenue%E2%80%9D'), _descriptive addresses_ are common and often require local knowledge to navigate.  Guiding deliveries and guests to unaddressed locations often requires providing detailed directions or providing local help, though providing a GPS location via a 'pin' in a mapping software is common in some areas.

Several attempts to address this gap have been undertaken by overlaying the planet with a grid and assigning unique identifiers to grid references.  [Google's PlusCodes]('https://maps.google.com/pluscodes/') assigns alphanumeric codes, organized logically to grid space.  [what3words]('https://what3words.com/clip.apples.leap') assigns a combination of three unique words in multiple languages along a similar schema.

_***But what if you don't know the identifier and can't really find someone to ask?***_

I have personally run into this on two occasions while trying to find the location of a house we were staying at in Donegal County, Ireland - and a similar experience on the Greek island of Corfu.  I had a picture of the house and a rough location, but no address.  It's not always easy to find "Tom's house, down the way from the Pub" in the dark!  To solve this issue, I found myself leveraging satelite images - identifying distinct features in images and attempting to locate the house by physical relationship.  The material (color) and shape of the roofs visible in images taken from the ground (or from inside the property) were the most consistently useful datapoints.  I took a few wrong turns, but eventually found the destination.

### Goal
This project seeks to leverage publicly available satelite and aerial imagery of unaddressed areas to identify buildings, roof types and materials and provide users with a map narrowing down likely destinations.  In cases where the town/city designation, paired with the roof description is unique, the goal is to create a geolocation of of the building they are looking for.  Otherwise, this project seeks to reduce the number of wrong turns taken in unfamiliar, unaddressed areas.

***How might this be leveraged in practice?***

Once an area is processed, users can take the following steps to help identify their destination:
1. Type in the name of their destination's village or town (ex. Agios Matheos)
2. Review available images of the building and select a roof color (ex. Red Tile)
3. Provide a best guess of the roof style, designated by the number of roof ridges (ex. one ridge)

The model will then assess the aerial imagery availabe (in this implementation, sourced from Google's Earth API) to identify buildings in the specified area and return building locations (latitude and longitude) of buildings matching the provided description.

### Methodology
This project requires the usage of a wide range of libraries and methods within the geopspatial analysis and data science disciplines.  A brief overview of the stages involved is listed below with details of the project provided in relevant sections later in this document.

##### 1. Define Geographic Boundaries
Before capturing satelite images covering the Earth, setting fixed boundaries over which to search is key to capture relevant and useful images.  For this project, the Greek island of Corfu is of interest.  As Greece's 7th largest island (>610 sq. km), a secondary search area is defined by one of the island's 16 municipalities

Island and relevant municipal boundaries are sourced from the Greek Government in the form of shapefiles.  These are re-projected into a common map projection and filtered to create relevant boundaries (ex. coastline) over which to search.
For more details, see [here](####_Geographic_Boundaries).

##### 2. Source Aerial Imagery
Satelite Imagery is captured using Google's Earth Engine API and Google Satelite imagery.  Images are captured in two offset overlapping grids within the designated geographic boundary to maximize coverage and capture of complete buildings within individual images.  To aid in locating houses geographically, images are named with the latitude and longitude of the center of the image.

For more details, see [here](####_Satelite_and_Aerial_Imagery).

##### 3. Model Selection

##### 4. Source and Prepare Training Data

##### 5. Instantiate a Pre-Trained Convolutional Neural Network

##### 6. Evaluate Model Outputs and Identify Roof Colors 

## Data Sources

### Data Dictionary

---
## Data Sourcing and Preparation
### Geographic Boundaries
[Notebook](./code/01_Shoreline_Boundary.ipynb)
#### Sourcing
Though it is possible to capture arerial and satelite imagery covering the entirety of the planet, it would not be useful (oceans and seas), nor practical (data volume) to do so.  Two methods are available to establish boundaries, whithin which imagery can be captured:
1) ***Bounded Box*** - A simple box of Latitude and Longitude minimum and maximum values.  The result would cover a rectangle (map projectios notwithstanding).  This simpler approach would work well if a segment of a city is the area of interest, for example.
2) **Geometrically Defined Boundary*** - Leveraging geometric data (such as the border of a country or a coastline) protects against capturing irrelevant images.  In the case of an island, restricting capture to images on the island - and not the sea is preferable.

This project leverages a _blended_ approach.  Shapefiles (defining geographic boundaries) are isolated to the area of interest.  The minimum and maximum latitude and longitude values of the bounding geomertry is used to instantiate the outer limits of the search grid.  Images are considered gridwise across the search grid.  At each point, if the center of the image is outside the established geographic boundary, the image is not captured.

This project (focused on the Greek island of Corfu) leverages shapefiles available at Geodata.gov.gr for [Country Borders](https://geodata.gov.gr/en/dataset/aktogramme/resource/1ba9f74e-eb7a-4d0d-8858-864218806dbc) and 
Source: [Municipalities](https://geodata.gov.gr/en/dataset/oria-demon-kapodistriakoi), respectively.  For clarity, municipalities leveraged are defined by the Kapodistrias plan.

#### Preparation
Shapefiles are imported using the GeoPandas library.  Adjustments are made to the map projection (crs) to ensure compatibility with common latitude, longitude coordinate systems.

Line segments provided in the shapefile (on the order of a kilometer in length) are combined into closed shapes (beaches form an island).  The greek border shapefile covers the entirety of Greece, including the mainland and its many islands.  As the area of focus is Corfu, the correct geometry object must be isolated.  Names are not provided, so a sorted list of geometries is created by enclosing area.  Corfu is the seventh largest island in Greece, which enables identification and isolation.

Municipalities similarly cover the entirety of Greece, but names are provided for reference and filtering.  Here 'Ν. ΚΕΡΚΥΡΑΣ' (the Greek named for the island "Kerkyra") is used to filter to municipalities on the island.  The municipality of 'ΑΓΡΟΣ' (Agros) is selected to enable further segmentation of the island if needed.

### Satelite and Aerial Imagery
[Notebook](code/01_Image_Sourcing.ipynb)
#### Sourcing
The core of this project is the use of high quality satelite or (consistently sourced) aerial imagery covering an unaddressed area.  After assessing imagery available from NASA and Google's Earch Engine, Earth Engine was selected due to ease of use and exceptional documentation.  This includes a Google Colab Setup Guide provided in an .ipynb notebook [here](https://colab.research.google.com/github/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb).  Google Earth Engine operates via API, and requires an account to be created and API keys to be generated and used for each session.

#### [Imagery Selection](code/01_Image_Sourcing.ipynb##-select-projection-and-imagery-from-open-source-google-earth-engine)
Once a notebook is connected to Google's Earth Engine, HTML maps can be generated and displayed in a similar manner to those available on Google Maps, including zoom and drag functionality.  The baseline map does not provide satelite imagery, but does highlight road networks, towns, and other road-atlas relevant features.  The [folium](https://python-visualization.github.io/folium/modules.html) library can be leveraged to source and overlay a wide array of images, including multiple satelite sources.  Map layers (leaflets) can be explored and sourced from [leaflet providers on Github](http://leaflet-extras.github.io/leaflet-providers/preview/).

Each available leaflet was explored at high levels of zoom in an effort to select a leaflet which includes high quality images at a low zoom level (houses can be seen clearly when looking at a small geographic area.).  ESRI Imagery, provided via arcgis, enables a deep level of zoom, but imagery over the Island of Corfu is a bit blurry (image below).  This may make identifying individual buildings more difficult for a neural network, as the boundaries are clearly defined at high levels of zoom.

<figure>
    <img src=images/Agios_Matheos_ESRI.png width=400 alt='Agios Matheos - ESRI Imagery'>
    <figcaption>Agios Matheos - ESRI Imagery</figcaption>
</figure>

Another common library which commonly leverages the Google Earth Engine API is the geemap library.  The [documentation](https://github.com/gee-community/geemap/blob/master/geemap/basemaps.py) for this library contains a reference to the Google Satelite imagery.  Though this imagery is not available through the leaflet sources mentioned above (as ESRI imagery), it can be called via the Google Earth Engine if specified.  As shown below (with the same zoom level and map boundary), Google's Satelite imagery is much clearer for this area at high zoom levels.  As a result, this is the visualization layer selected for satelite image capture for this project.

<figure>
    <img src=images/Agios_Matheos_Google_Satelite.png width=400 alt='Agios Matheos - Google Earth Imagery'>
    <figcaption>Agios Matheos - Google Earth Imagery</figcaption>
</figure>

#### [Image Capture](code/01_Image_Sourcing.ipynb##-capture-small-scale-images)
> Disclaimer: This project is undertaken for exploratory and educational purposes.  Commercial use of images sourced through Google Earth Engine or other sources is generally prohibited.  Special limitations on use of specific leaflet layers may bear additional usage restrictions.

With geographic boundaries established and an imagery leaflet layer selected, the next step is to collect a gridwise sample of satelite images to serve as evaluation data for modeling (the area over which to identify and categorize buildings).  The first stage is defining the zoom level for the images to be captured.  Each leaflet image layer has a fixed maximum zoom limit.  This is generally the maximum useful zoom level to maintain resolution. 

> Though I am no expert in digital photography, one can think of this like older digital cameras - users can zoom in physically (the lenses on the camera move) without sacrificing the quality of the image captured.  Digital zoom can extend this range (the lenses no longer move), but the image resolution is compromised... the camera effectively "stretches" the pixels it reads to fill the desired frame.  This is also how 'pinch to zoom' works on smartphone cameras.

Here the maximum zoom level is selected (18 for Google Satelite Imagery) to maximize pixel density over a small area.

A frame size of 300x310 pixels is used for the images captured.  This may seem small, but this is chosen for two purposes:
1. Convolutional Neural networks are generally well suited for smaller images - this project also aims to identify individual buildings from the image.  Tuning the size of the images into digestable chunks is hypothesized to yield better accuracy and model performance (it is likely to find 10 houses from a small image than 100 houses in a larger image)
2. Geographic Area Covered.  At a zoom level of 18 and 310 pixels in height, each frame covers approximately 0.0012 degrees latitude... or 130 meters.  A lot of small houses can fit into an area roughly twice the size of a football pitch.

At this image size, a single gridwise map covering the island of Corfu would result in 118000 images being considered or captured.  For the municipality of Agros, this is closer to 4000 images.  

Furthermore, images must be captured in two overlapping grids to maximize the instances of complete buildings being present in images.  A illustrative process flow of the image capture procedure is provided below.

<figure>
    <img src=images/grid1.png width=400 alt='Initial Grid Capture'>
    <figcaption>1. Primary images are captured to create a grid covering the island boundaries within the maximun defined extents (cardinal directions) of the island.  Representative primary images boundaries are roughly identified by blue boxes.</figcaption>
</figure>

<figure>
    <img src=images/grid2.png width=400 alt='Secondary Grid Capture'>
    <figcaption>2. A secondary grid of images are captured within the same boundaries with a one-half width and height offset to the primary grid.  Any buildings which were divided into multiple images in the primary grid are captured entirely within one image on the secondary grid.  Representative primary images boundaries are roughly identified by orange boxes.</figcaption>
</figure>

<figure>
    <img src=images/grid_coastline.png width=400 alt='Coastline Control During Grid Capture'>
    <figcaption>3. Geographic Boundary Control - As each section of the grid is instantiated, the geographic boundary is considered prior to image capture.  Here the coastline is represented by a dashed green line.  If the center of the grid falls inside the geographic boundary (a closed shape), the image is captured.  Otherwise the image is skipped and the process moves to the next grid point.</figcaption>
</figure>

Capturing each image requires five steps, accomplished in a single function:
1. Identify wither the center of the image is within the designated boundary geometry
2. Call Google Earth Engine API with the designated latitude/longitude, image size and zoom level - this returns an html window
3. Save the HTML window in a temporary folder - leveraging the latitude and longitude of the image in the file name
4. Call the imgkit library to read the html object and save the output as a .png file
5. Delete the temporary HTML file to save space.

Using the above process, thousands of high quality, relevant satelite and aerial images of consistent size and are captured quickly and available for modeling.  Some examples are below:

<figure>
    <img src=data/clean_data/satelite_images/39.712739999999954_19.746329999999986_sat.png width=400 alt=''>
    <figcaption>Fuel Station between Troumpetas and Chorepiskopi</figcaption>
</figure>
<figure>
    <img src=data/clean_data/satelite_images/39.72233999999993_19.741529999999987_sat.png width=400 alt=''>
    <figcaption>Chorepiskopi</figcaption>
</figure>
<figure>
    <img src=data/clean_data/satelite_images/39.70433999999997_19.70472999999999_sat.png width=400 alt=''>
    <figcaption>Olive Tree Grove - Agros, Corfu</figcaption>
</figure>

---
### Model Selection

The goal of this project is to leverage the images captured above to identify instances of buildings - and details about these buildings.  For this application, a Convolutional Neural Network is the best candidate machine learning model type.  Specifically, a Convolutional Neural Network which is designed to identify one or more instances of one or more classes of object within an image.  These model

> ***A brief summary of Neural and Convolutional Neural Networks***  Neural Networks are a class of machine learning model which leverages a series of layers 



---
### Source and Prepare Training Data



## Exploratory Data Analysis

## Modeling

A Neural Network (likely a Convolutional Neural Network, or CNN) is best suited to the task of identifying houses and their key characteristics as of interest in this project. There are two primary approaches available: Self-Trained Model, or a Pre-Trained Model.

Self Trained Models - are generally more flexible to specific tasks, but require a massive amount of training and validation data (as well as processing time) to fit model weights.
Pre-Trained Models - have been designed and trained separately and are available for public use. Here, the input and output layers are updated, while the key hidden layers of the Network are 'frozen'
Given the volume of training data which would be required to accurately identify houses, leveraging a pre-trained model is the likely best path for this use case. Widely available models are considered below.


### Adjustments to Mask R-CNN
1. Update utils.py (294) prepare function to ensure generation of uniquely indexed images for both training and validation datasets (replaces 0-indexed sequential reference)
1. Update model.py to read a consistent class list (was previously dependent upon a numeric index of images, always 0 indexed.)
1. Update utils.py to handle an upgrade of the skimage library which interferes with mask resizing (where masks are stored as boolean arrays).  Cite: https://stackoverflow.com/a/73759783
1. Update model.py (2156) to change selected keras optimizer from 'optimizer = keras.optimizers.SGD(...' to 'optimizer = keras.optimizers.legacy.SGD(...' to handle upgrades made to tensorflow.  Cite: https://stackoverflow.com/a/75596562 in reference to [Tensorflow Release Notes](https://github.com/tensorflow/tensorflow/releases)
1. Update model.py at line 2362 to set workers=1 when calling keras.fit().  Multiprocessing with current Tensorflow versions and this model causes the fit process to hang (potentially indefinitely).  Cite: https://github.com/matterport/Mask_RCNN/issues/2696#issuecomment-1066224728

downscale steps per epoch from generally recommended 2-500 range to 100 to create more checkpoints (slow training and colab timeouts)


## Visualizations and Accessibility

## Conclusions and Recommendations


## Credits and Acknowledgements

## Images
* ![Island of Corfu](images/iss030e254053~large.jpg) Credit: NASA Unpiloted Progress Resupply Vehicle NASA ID: iss030e254053 (22 April 2012). The city of Kerkyra (Corfu) is on seen the island at bottom-center.