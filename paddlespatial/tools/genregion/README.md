
# Generating Urban Areas of Interest Using the Road Network

## 1.Introduction

Urban area of interest (AOI) is broadly defined as the area within an urban environment that attracts People’s attention, and it is the prerequisite for many spatio-temporal applications such as urban planning and traffic analysis. However, constructing closed polygons to form numerous AOIs in a city is time-consuming and perplexing. Traditional grid-based methods offer the simple solution as they cut the city into equal-sized grids, but information about the urban structure will disappear.

To relieve this problem, we report a vector-based approach to divide urban space into proper regions using road networks. Regions are generated through following steps:

1. The algorithm tries to simplify the road network by conducting hierarchical clustering on two endpoints of each road segment.
2. Every segment is broken into pieces by its intersections with other segments to guarantee that each segment only connects to others at the start and end nodes.
3. We generate each region by recursively finding the leftmost link of a segment until a link has travelled back to itself.
4. We merge tiny blocks and remove sub-regions to have a more meaningful output.

To ensure the robustness of our model, we use road data from various sources and generate regions for different cities, and we also compare our model to other urban segmentation techniques to demonstrate the superiority of our method. A coherent application public interface is public available, and the code is open sourced (<https://github.com/PaddlePaddle/PaddleSpatial/tree/main/paddlespatial/tools/genregion>).

## 2. Examples

### Regions generation examples from New York and Beijing road networks

New York:
<p align="center">
    <img align="center" src="result/newyork_links.png" width="350" height="350" alt="New York Road Network" style="margin:0 auto"/>
    <img align="center" src="./result/newyork_polygons.png" width="350" height="350" alt="New York Urban Regions" style="margin:0 auto"/>  
</p>

Beijing:
<p align="center">
    <img align="center" src="result/beijing_links.png" width="350" height="350" alt="Beijing Road Network" style="margin:0 auto"/>
    <img align="center" src="./result/beijing_polygons.png" width="350" height="350" alt="Beijing Urban Regions" style="margin:0 auto"/>  
</p>

## 3. Usage

### Installation

The `genregion` has been published to The Python Package Index (PyPI) to help users quickly install this package. Users can simply run the command below to access our application:

```shell
pip install genregion
```

### Instruction

When programmers and researchers delve into projects about spatial analysis, the Python `shapely` package is always unavoidable. It provides exhaustive and powerful spatial-related objects as well as functions which make geometric problems more manageable. Therefore, our module interface is constructed highly associated with the `shapely` package.  

The main function `generate_regions()` takes a list of road segments and generate a list of `Shapely.geometry.Polygon` objects as the output. To make our application flexible, the interface accepts 4 types of input:

1. The file path that stores every segment of the road network, and each line in this file should only contain point coordinates of a single road segment.
    * Ex: _255834.51327326 3323376.71868603,260889.23516149 3321991.45674967_.
        Note that there is a comma between two points.

2. A list of tuples of shapely points:
    * Ex: _[(Point_1, Point_2), (Point_3, Point_4), ...]_.

3. A list of tuples of point coordinates.
    * Ex: _[((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...]_.

4. A list of shapely LineString objects.
    * Ex: _[LineString_1, LineString_2]_.

Specifially, a concise example to use our module is shown below:

```python
>>> from genregion import generate_regions
>>> urban_regions = generate_regions(segments, grid_size=1024, \
                     area_thres=10000, width_thres=20, \
                     clust_width=25, point_precision=0)
```

Note that there are 5 extra parameters here. They can help you adjust the cluster and filter regions.

* grid_size: It denotes the size of the grid while we hash those segments and regions in order to facilitate searching.

* area_thres: It represents the threshold to filter out regions, and any region whose area is smaller than this value will be merged.

* width_thres: The width here means the ratio of the area to the perimeter of a region.
It defines whether a region is so narrow that it can be treated like a road. Any region has the ratio less than this value will be merged.

* clust_width: The threshold while merging clusters. Only two clusters have distance less than this value can be merged to a new cluster.

* point_precision: The precision of the coordinate.
  * If we are using projection coordinates like BD09MC or EPSG3587, we suggest Point.precision = 2.
  * If we are using the geodetic system like WGS84 and BD09, we suggest Point.precision = 5.

### Potential Resources

The ***figshare*** website (<https://figshare.com/articles/dataset/Urban_Road_Network_Data/2061897>) provides road networks data for 80 of the most populated urban areas in the world. The data consist of a graph edge list for each city and two corresponding GIS shapefiles (i.e., links and nodes). We use some of their data to run the experiment. If you cannot find a good road network data source, this can be your choice.

## 4. Main advantages

* Succinct Interface: Users only have to input a list of segments and can get a list of polygons easily.

* Compatibility: The input and output of our function can both be `shapely` objects, which can help users continue their geo-spatial projects smoothly.

* Consistency: Unlike other region generation algorithms, our output contain polygons puerly, without any segment left. Meanwhile, our algorithm can be deployed in either Python 2 or Python 3.

* Effectiveness: The largest urban road network in the world is in New York, and our algorithm can process it in less than 9 minutes.
