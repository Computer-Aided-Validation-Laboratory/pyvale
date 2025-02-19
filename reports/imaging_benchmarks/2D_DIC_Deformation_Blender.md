# Deformation images for 2D DIC using Blender

After verifying the accuracy of Blender-produced images using rigid body motion images, images in which the object within was deforming were rendered.  

## How deformation is set to the object in Blender  
The SimData object was used to store not only the mesh data, but also the deformation data, as this meant that Blender code could be integrated into `pyvale`.  
In order to deform the object, the displacements stored in the SimData object were applied to each individual node.  
- Firstly the positions of the deformed nodes were extracted from the SimData object
- A new frame for each timestep was created using shape keys
- The node coordinates for the new frame were set to the previously calculated deformed node positions
- An image of this frame is taken and the process is repeated for all the timesteps within the displacement data  

## How the deformation images were verified  
In order to accurately verify that the Blender images were accurately representing the displacements applied, they were compared to images produced via image deformation.  
This was done so that any errors are attributed to the setting of the displacement within the images, as opposed to the DIC processing.  
The same simulation exodus file was used for both the Blender and image deformation images, and the reference image for the image deformation was set as the initial (non-deformed) image produced by Blender, to ensure that the FOV, ROI and image resolution were the same.  
The image deformation tool that was used was the one within `pyvale`. This tool has already been benchmarked against MatchID's own image deformation module, so benchmarking the Blender image rendering against it is deemed to be sufficient.  
It should be noted that in order to accurately compare the data between the Blender images and image deformation images, the ROI had to be exactly the same, otherwise interpolation errors between the subsets caused false errors between the datasets.

#### How the datasets were compared  
The quantity of interest compared is the horizontal displacement, as this was directly imposed to the object in Blender.   
The image deformation and Blender datasets were compared within MatchID (the image deformation data was subtracted from the Blender data) and this full-field data was extracted to python for further analysis.  
The full-field data was plotted, to visualise the error across the whole ROI.    
When analysing the error maps, the magnitude of error was related to a 'noisefloor' value. This noisefloor for all cases is taken to be 0.01 pixels.  
For the cases with larger errors, the error data was made positive, and plotted with a maximum value of 4.2 times the noisefloor.   

## Chosen simulation cases  
Four initial simulation cases were chosen as benchmarks:
- A positive linear displacement in the x-direction with the displacement in the left side of the object set to 0 
	- This was chosen as it is a very simple benchmark, with a highly recognisable solution
	- This case was applied with displacement functions at two different scales to see the impact of displacement size on the error
- A shearing case in which a linear diplacement in the x-direction was applied with displacement at the bottom of the object set to 0
- Simulation case 18, which is a simulation case within `pyvale`, so data from the Blender images can be compared to other methods within `pyvale`  

All of these simulation cases were for 2D meshes, to ensure that no out-of-plane motion occured.  

### Linear displacement simulation case
The linear displacement simulation case had a displacement field with the pattern below:

![Linear function displacement field](./images/Small_linear_paraview.png)   

Both cases had the same linear displacement applied, with the larger displacement case having a displacement of 10 times the magnitude of the smaller case.  
The final deformation image for the larger linear displacement case can be seen below:  
![Final deformation image](./images/linear_disp_large.tiff)    

When comparing the error from the larger and small deformation cases, the pattern in the error maps is largely the same, however the magnitude of error for the larger deformation is an order of magnitude larger.  
The error maps produced are the difference between the blender and image deformation data, divided by the threshold noise-floor (0.01 px), to see how much the error deviates from the noise-floor.  
Both sets of error maps are largely points around a single value, with areas of higher error in the top-right and bottom-left corners.   
These larger errors manifest themselves as squares, and this is due to the fact the mesh used was very coarse.  

![Small case error map](./images/linear_small_initial_mesh.svg)   
_Error map for the small linear displacement case at the final timestep_  

![Large case error map](./images/linear_large_initial_mesh.png)   
_Error map for the large linear displacement case at the final timestep_  

#### Effect of mesh fineness on error maps
The effect of using a finer mesh was investigated for both simulation cases.   
Meshes with increasing elements were tested. The meshes sizes that were tested were:
- 16 x 8 (the original mesh)
- 50 x 25
- 100 x 50
- 200 x 50  

_Larger displacement function_: 
The different size meshes were all tested with the same environment in Blender, and with the same ROI in MatchID, so any differences in the results can be seen as due to the difference in mesh coarseness.   

![Large case 50x25](./images/large_case_50x25.png)   
_Error map at the final timestep for larger displacement case with mesh size of 50x25_   

![Large case 100x50](./images/large_case_100x50.png)   
_Error map at the final timestep for larger displacement case with mesh size of 100x50_   

![Large case 200x100](./images/large_case_200x100.png)    
_Error map at the final timestep for larger displacement case with mesh size of 200x100_     

When using finer meshes, it can be seen that any artefacts from the mesh size are gone, and the only full-field pattern left is noise.  
- Some artefacts of the mesh remain in the error maps with a mesh size of 50x25. 
- The error maps between the finer meshes are largely similar, so as long as the mesh size is finer than 100x50, it is deemed to be reasonable to be used.  

Using a finer mesh also reveals the true nature of the error map, as it can be seen to be noise, instead of having areas of relatively larger error
- It can also be seen that the maximum error is also a function of this mesh size, so error can be more accurately determined with a smaller mesh  

Even though the average error is only around 0.677 times the noisefloor, the error map is not just noise, it has higher error at the right hand side of the image. Therefore, care should be taken when using this method for large deformations.  

_Smaller displacement function_:   
The same relationship between mesh size and error was also seen for the smaller displacement function.  

![Small case 50x25](./images/small_case_50x25.png)   
_Error map at the final timestep for smaller displacement case with mesh size of 50x25_   

![Small case 100x50](./images/small_case_100x50.png)   
_Error map at the final timestep for smaller displacement case with mesh size of 100x50_   

![Small case 200x100](./images/small_case_200x100.png)   
_Error map at the final timestep for smaller displacement case with mesh size of 200x100_

Notably, for a mesh size of larger than 100x50, the magnitude of error sits below the value of the noisefloor. Therefore, even the more extreme values of the error can be explained as noise.  
The average value of error at the last timestep is around 0.228 times the noisefloor, well below the threshold for noise.  

|            | 16x8 | 50x25 | 100x50 | 200x100 |
| -----------|------|-------|--------|---------|
|Larger disp | 2.02 | 0.737 | 0.684  | 0.677   |
| Smaller disp | 0.331| 0.230 | 0.228  | 0.228  |   

_The average error for the final timestep normalised by the noisefloor, for different mesh sizes_   


#### Comparison to reference solution  
The results from processing these images were also compared to the reference solution.   
This solution was created by applying the displacement function across a matrix.   
However, this method may not be accurate as it is possible that the ROI is not the one assumed, and so there are interpolation errors present (as each subset is not be compared to its counterpart).    
For more accurate benchmarking, comparison to image deformation was used, as they both contain errors from the DIC processing, so any error present between them is due to the image rendering process.    

Error maps between blender results and reference solution (for small displacement function, with mesh errors removed):   
- All error maps between blender results and the refernce solution showed the same pattern.  
- Comparison to image deformation displacements also showed this same pattern, so it is likely that this error is due to the DIC processing.   

![Comparison between blender and refernce](./images/blender_reference_comparison.svg)   
_Comparison between Blender displacements and the reference solution_    

![Comparison between imagedef and reference](./images/imagedef_reference_comparison.svg)    
_Comparison between image deformation displacements and the reference solution_     


### Shearing simulation case  
The shearing simulation case had a displacement field with the pattern below: 

![Shearing disp field](./images/Shear_case_paraview.png)   
 
The initially tested images had the same issue as the linear displacement images, in that areas of larger displacements were found (relating to the coarse mesh).  
However, the extremes of these errors were lower as compared to the  linear displacement cases. The error map for the final timestep can be seen below.  

![Shear error map final timestep](./images/shear_error_map_16x8.png)   
_Error map of shear case at the final timestep with the original mesh_  

Since multiple artefacts of the mesh can be seen in this case's error maps, more images were generated using a finer mesh of size 100x50, since this was deemed to be adequate.  

![Finer mesh error map](./images/shear_error_map_100x50.png)   
_Error map of shear case at the final timestep with a mesh of 100x50 elements_  

It can be seen that the error map for this simulation case is much typical of noise when using a finer mesh. The extremes of the error are again below the noisefloor, meaning this error map can be regarded as noise.  
The average error (at the final timestep) when using this mesh is around 0.19 times the noisefloor.   


### Case 18 simulation case  
Simulation case 18 had a displacement field with the pattern below:  

![Case 18 disp field](./images/case18_paraview.png)   


The error map for simulation case 18 can be seen to just be noise, with no clear deformation patterns.  
It can be seen that there is an area of higher error at the top-left corner of the image, however this is likely due to the mesh coarseness, and could be altered by making the mesh finer.  
From the error map, you can see that most of the error is lower than the noise-floor, showing that the existing error is due to noise.  
It should be noted that the error maps for each timestep are very similar, with the final timestep being chosen as an example, as it has the largest level of error.  

![Case 18 error map](./images/case18_error_map.png)   
_Error map for the final timestep of case 18_  


### Impact of DIC parameters 
The impact of DIC parameters was tested for the first three simulation cases.  
The images were processed with two different sets of parameters:
- Subset size of 17 and step size of 10
- Subset size of 15 and step size of 8  

The error maps for each set of parameters were very similar, the only difference being that the error maps processed with the larger DIC parameters were more smoothed.  
It is expected that this would be more prevelant if even larger DIC parameters were used.  

## Conculusion from comparison between Blender and image deformation
From the comparison between the Blender and image deformation produced images, the error in displacement between them is very low (smaller than the threshold noisefloor of 0.01 pixels).  
However, it was seen that the accuracy of the Blender images is dependent on the mesh size. Therefore, a mesh of sufficient size should be used with this image rendering technique.  


## Future work for Blender rendering capabilities  
The clear next step for utilising Blender's rendering capabilities is to develop a stereo camera system.  
This would allow 3D objects deforming both in and out of plane to be imaged.  
It would also be ideal to utilise a DIC processing engine withing `python` to automatically process the images once they are generated.  

