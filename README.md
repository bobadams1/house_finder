# Identifying Houses in Areas without Addresses - House Finder

## Problem Statement
Though common throughout most western countries and localities, unified building address systems are not universally accepted.  From remote communities in Africa and Ireland to [Carmel by the Sea in California]('https://ci.carmel.ca.us/post/addresses#:~:text=A%20unique%20characteristic%20of%20Carmel,houses%20south%20of%2012th%20Avenue%E2%80%9D'), descriptive addresses are common and often require local knowledge to navigate.  Guiding deliveries and guests to unaddressed locations often requires providing detailed directions or providing local help, though providing a GPS location via a 'pin' in a mapping software is common in some areas.

Several attempts to address this gap have been undertaken by overlaying the planet with a grid and assigning unique identifiers to grid references.  [Google's PlusCodes]('https://maps.google.com/pluscodes/') assigns alphanumeric codes, organized logically to grid space.  [what3words]('https://what3words.com/clip.apples.leap') assigns a combination of three unique words in multiple languages along a similar schema.

_***But what if you don't know the identifier and can't really find someone to ask?***_

I have personally run into this on two occasions while trying to find the location of a house we were staying at in Donegal County, Ireland - and a similar experience on the Greek island of Corfu.  I had a picture of the house and a rough location, but no address.  It's not always easy to find "Tom's house, down the way from the Pub" in the dark!  To solve this issue, I found myself leveraging satelite images - identifying distinct features in images and attempting to 

### Goal

### Methodology



## Data Sources

### Data Dictionary

## Data Sourcing

## Data Preparation and Transformation

## Exploratory Data Analysis

## Modeling

A Neural Network (likely a Convolutional Neural Network, or CNN) is best suited to the task of identifying houses and their key characteristics as of interest in this project. There are two primary approaches available: Self-Trained Model, or a Pre-Trained Model.

Self Trained Models - are generally more flexible to specific tasks, but require a massive amount of training and validation data (as well as processing time) to fit model weights.
Pre-Trained Models - have been designed and trained separately and are available for public use. Here, the input and output layers are updated, while the key hidden layers of the Network are 'frozen'
Given the volume of training data which would be required to accurately identify houses, leveraging a pre-trained model is the likely best path for this use case. Widely available models are considered below.

### Satelite Imagery

### Land Imagery

### Combined Predictions

## Visualizations and Accessibility

## Conclusions and Recommendations


## Credits and Acknowledgements

## Images
* ![Island of Corfu](images/iss030e254053~large.jpg) Credit: NASA Unpiloted Progress Resupply Vehicle NASA ID: iss030e254053 (22 April 2012). The city of Kerkyra (Corfu) is on seen the island at bottom-center.