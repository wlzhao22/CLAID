# Class Agnostic Instance-level Descriptor via Hierarchical Semantic Region Decomposition

This repository contains the implementation code for the Class Agnostic Instance-level Descriptor via Hierarchical Semantic Region Decomposition project. The main components of the project are the detector and descriptor.

## Overview
The Class Agnostic Instance-level Descriptor via Hierarchical Semantic Region Decomposition is a computer vision project that aims to generate class-agnostic descriptors for instances in an image. The project utilizes a hierarchical semantic region decomposition approach to extract fine-grained descriptors that encode rich visual information for instance search.

## Components

### Detector
The detector component is responsible for localizing hierarchical semantic instances in an image.

### Descriptor
The descriptor component is responsible for generating class-agnostic descriptors for the detected instances. 

## Installation
To use this project, follow these steps:

1. Download the repository:

2. Install the required packages
~~~
pip install -r requirements.txt
~~~

## Usage
To use the detector components, follow the instructions below:
1. Prepare your input image(s) and ensure they are accessible within the project directory.
Prepare your input dataset txt file which contains the pathes of dataset images.

2. Run the detector to find hierarchical semantic regions in one image:
~~~
python detector.py --mode one --image_path /path/to/image.jpg --save_path /path/to/save/results/
~~~

3. Run the detector to find hierarchical semantic regions in a reference dataset:
~~~
python detector.py --mode dataset --dataset_path /path/to/dataset_path.txt --save_path /path/to/save/masks_and_boxes/
~~~

After finding the instances using detector, to use the descriptor components, follow the instructions below:
1. Run the descriptorto get the feature and boxes of reference.
~~~
python descriptor.py --is_qry False --dataset_path /path/to/dataset_path.txt --mask_path /path/to/save/masks/ --save_path /path/to/save/results/ --box_path /path/to/box_path.txt
~~~

2. Run the descriptorto get the feature and boxes of query.
~~~
python descriptor.py --is_qry True --dataset_path /path/to/dataset_path.txt --save_path /path/to/save/results/ --box_path /path/to/box_path.txt
~~~