# Identifying Houses in Areas without Addresses - House Finder
![](images/iss030e254053~large.jpg)
Photo Credit: [NASA](https://images.nasa.gov/details/iss030e254053) (ISS030-E-254053 (22 April 2012))
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

For more details, see [here](####_Satelite_and_Aerial_Imagery).
)
##### 3. Source and Prepare Training Data

##### 4. Instantiate a Pre-Trained Convolutional Neural Network

##### 5. Evaluate Model Outputs and Identify Roof Colors 

## Data Sources

### Data Dictionary

---
## Data Sourcing and Preparation
#### Geographic Boundaries
[Notebook](./code/01_Shoreline_Boundary.ipynb)
##### Sourcing
Though it is possible to capture arerial and satelite imagery covering the entirety of the planet, it would not be useful (oceans and seas), nor practical (data volume) to do so.  Two methods are available to establish boundaries, whithin which imagery can be captured:
1) ***Bounded Box*** - A simple box of Latitude and Longitude minimum and maximum values.  The result would cover a rectangle (map projectios notwithstanding).  This simpler approach would work well if a segment of a city is the area of interest, for example.
2) **Geometrically Defined Boundary*** - Leveraging geometric data (such as the border of a country or a coastline) protects against capturing irrelevant images.  In the case of an island, restricting capture to images on the island - and not the sea is preferable.

This project leverages a _blended_ approach.  Shapefiles (defining geographic boundaries) are isolated to the area of interest.  The minimum and maximum latitude and longitude values of the bounding geomertry is used to instantiate the outer limits of the search grid.  Images are considered gridwise across the search grid.  At each point, if the center of the image is outside the established geographic boundary, the image is not captured.

This project (focused on the Greek island of Corfu) leverages shapefiles available at Geodata.gov.gr for [Country Borders](https://geodata.gov.gr/en/dataset/aktogramme/resource/1ba9f74e-eb7a-4d0d-8858-864218806dbc) and 
Source: [Municipalities](https://geodata.gov.gr/en/dataset/oria-demon-kapodistriakoi), respectively.  For clarity, municipalities leveraged are defined by the Kapodistrias plan.

##### Preparation
Shapefiles are imported using the GeoPandas library.  Adjustments are made to the map projection (crs) to ensure compatibility with common latitude, longitude coordinate systems.

Line segments provided in the shapefile (on the order of a kilometer in length) are combined into closed shapes (beaches form an island).  The greek border shapefile covers the entirety of Greece, including the mainland and its many islands.  As the area of focus is Corfu, the correct geometry object must be isolated.  Names are not provided, so a sorted list of geometries is created by enclosing area.  Corfu is the seventh largest island in Greece, which enables identification and isolation.

Municipalities similarly cover the entirety of Greece, but names are provided for reference and filtering.  Here 'Ν. ΚΕΡΚΥΡΑΣ' (the Greek named for the island "Kerkyra") is used to filter to municipalities on the island.  The municipality of 'ΑΓΡΟΣ' (Agros) is selected to enable further segmentation of the island if needed.

#### Satelite and Aerial Imagery
[Notebook](code/01_Image_Sourcing.ipynb)
##### Sourcing
The core of this project is the use of high quality satelite or (consistently sourced) aerial imagery covering an unaddressed area.  After assessing imagery available from NASA and Google's Earch Engine, Earth Engine was selected due to ease of use and exceptional documentation.  This includes a Google Colab Setup Guide provided in an .ipynb notebook [here](https://colab.research.google.com/github/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb).  Google Earth Engine operates via API, and requires an account to be created and API keys to be generated and used for each session.

Once a notebook is connected to Google's Earth Engine, HTML maps can be generated and displayed in a similar manner to those available on Google Maps, including zoom and drag functionality.  The baseline map does not provide satelite imagery, but does highlight road networks, towns, and other road-atlas relevant features.  The [folium](https://python-visualization.github.io/folium/modules.html) library can be leveraged to source and overlay a wide array of images, including multiple satelite sources.  Map layers (leaflets) can be explored and sourced from [leaflet providers on Github](http://leaflet-extras.github.io/leaflet-providers/preview/).

Each available leaflet was explored at high levels of zoom in an effort to select a leaflet which includes high quality images at a low zoom level (houses can be seen clearly when looking at a small geographic area.).  ESRI Imagery, provided via arcgis, enables a deep level of zoom, but imagery over the Island of Corfu is a bit blurry.
[ESRI Image]()


##### Preparation

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
### Satelite Imagery

### Land Imagery

### Combined Predictions

## Visualizations and Accessibility

## Conclusions and Recommendations


## Credits and Acknowledgements

## Images
* ![Island of Corfu](images/iss030e254053~large.jpg) Credit: NASA Unpiloted Progress Resupply Vehicle NASA ID: iss030e254053 (22 April 2012). The city of Kerkyra (Corfu) is on seen the island at bottom-center.