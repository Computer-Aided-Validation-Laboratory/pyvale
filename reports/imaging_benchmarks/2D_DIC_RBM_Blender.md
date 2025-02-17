# Rigid Body Motion Images for 2D DIC using Blender

## What is Blender and how is it used
Blender is an open source computer graphics software.  
It can be used to render images through both rasterisation and ray-tracing.  
Blender can be automated using the blender-python API `bpy`.  
It should be noted that even the latest version of `bpy` requires Python ==3.11.*  


## How is the mesh imported
Data from the SimData object can be extracted and imported as a mesh - and subsequently used to deform the object.  
Blender only reads surface meshes, so any 3D meshes must be skinned prior to importing.  
The SimData mesh must be converted to .obj format to be imported into Blender - as it supports both quad and triangle meshes.   

## Defining camera/lighting parameters
The intrinsic lighting and camera parameters can be precisely defined.  
The camera currently being modelled is an AV Alvium 1800 U-507:  
- Pixel dimensions: 2452 x 2056
- Pixel size: 3.45 um

It should be noted that when calculating the Field Of View, Blender uses a slightly simplified pinhole camera model, so the FOV given by Blender is slightly different than expected. However, this difference is accounted for when defining the camera parameters, so the outputted FOV of the coded Blender should be accurate.   

Four types of lighting can be used:  
- Point 
- Sun
- Spot 
- Area

A point source light is currently being used for these sets of rendered images.   
These parameters can be altered to accurately reflect the experimental setup being used.  
 
## Rigid body motion images  
In order to quantify the accuracy of the Blender-produced images, a set of rigid body motion images were produced (of an object with a speckle pattern applied).    
In-plane rigid body motion between 0 and 1 pixel was applied to the object, and it was imaged.  

This displacement was applied to the object in two different ways:  
- Applying the displacement directly to the object as a whole through Blender
- Applying the displacement to every node within the object via the SimData object (the same way in which deformation displacements are applied)  

Applying the displacement in these two different ways was to not only determine the error from Blender-produced images, but also to test the method in which deformation displacements are applied with a simpler example case.  


These images were run through MatchID (with the same DIC parameters) to compare the MatchID calculated displacement with the imposed displacement  

|Imposed displacement | MatchID displacement when applied through Blender |  MatchID displacement when applied via SimData object  |
| :-----------------: | :-----------------------------------------------: | :---------------------------------------------------:  |
| 0.1                 |  0.0999962                                        |  0.0999961                                             |
| 0.2                 |  0.199994                                         |  0.199994                                              |
| 0.3                 |  0.299992                                         |  0.29999                                               |
| 0.4                 |  0.39999                                          |  0.399992                                              |
| 0.5                 |  0.499989                                         |  0.499996                                              |
| 0.6                 |  0.599991                                         |  0.600001                                              |
| 0.7                 |  0.699994                                         |  0.700011                                              |
| 0.8                 |  0.799997                                         |  0.800019                                              |
| 0.9                 |  0.900002                                         |  0.900002                                              |
| Average error (%)   | 1.90E-03                                          | 2.25E-03                                               |


This shows that the displacement that MatchID calculates is very close to the 'true' imposed displacement values.  
The average errors between the imposed displacement and MatchID calculated displacements are very low, and the deviation of the MatchID displacement from the imposed displacement is lower than the noise-floor, so can be considered just noise.  
The visualisation of the data confirms this, as no displacement fields are seen, only noise (see below).  

![MatchID rigid body motion data view](./images/RBM_matchid.png)  





