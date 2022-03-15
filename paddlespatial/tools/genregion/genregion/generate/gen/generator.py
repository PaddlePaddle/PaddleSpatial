# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 

import math
from collections import OrderedDict

from genregion.region import error
from genregion.region import geometry as mygeo
from genregion.generate.gen import cluster

def cos(invec, outvec):
    """Calculate the cosine value for two vectors.

    For each node, an invec means the vector that points to it, whereas an outvec means the vector that starts from it.
    This fuction is mainly used to calculate the cosine value of the invec and outvec of a node.

    Args:
        invec (tuple): A tuple of coordinates that represents the vector.
        outvec (tuple): A tuple of coordinates that represents the vector.

    Raises:
        error.RegionError: Zero vector.

    Returns:
        float: The cosine value.
    """
    (inx, iny) = invec
    (outx, outy) = outvec
    if (inx == 0 and iny == 0) or (outx == 0 and outy == 0):
        raise error.RegionError("Zero vector.")
    inlen = math.sqrt(inx * inx + iny * iny)
    outlen = math.sqrt(outx * outx + outy * outy)
    return (1.0 * inx * outx + 1.0 * iny * outy) / (inlen * outlen)

    
def side(invec, outvec):
    """Calculate the relative position of the outvec with respect to the invec.

    Here we use the cross product of two vectors to calculate the pseudo-sine value.
    We call it 'pseudo' since we don't divide those two norms, because we only have to know whether it is greater than 0 or not.
    If this value is less than 0, we define the outvec is on the right side of the invec.
    If this value is larger than 0, we define the outvec is on the left side of the invec.
    However, if the invec and the outvec are having the same or the opposite directions, the pseudo-sine value will be 0 iindistinctively.
    So we need the cosine angle of those two vectors to come in and help.
    When the pseudo-sine value is 0, if the cosine value is greater than 0, then the invec and the outvec must have the same direction;s
    on the contrary, if the cosine value is less than 0, those two vectors must have the opposite directions.


    Args:
        invec (tuple): A tuple of coordinates that represents the vector.
        outvec (tuple): A tuple of coordinates that represents the vector.

    Raises:
        error.RegionError: Zero vector.

    Returns:
        int: 1 represents the outvec is on the right. -1 represents the other way.
    """
    (inx, iny) = invec
    (outx, outy) = outvec
    if (inx == 0 and iny == 0) or (outx == 0 and outy == 0):
        raise error.RegionError("Zero vector.")
    side1 = inx * outy - outx * iny
    cos1 = inx * outx + iny * outy
    if side1 < 0:   
        return -1
    elif side1 > 0:  
        return 1
    elif cos1 > 0:   
        return 1
    else:   
        return -1


class Node(mygeo.Point):
    """ The point object for region generation.

    The difference between the node and the mygeo.Point is that we record the invecs and the outvecs.

    Attributes:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        in_links (list): Store all links that end at this node.
        out_links (list): Store all links that start at this node.

    """
    def __init__(self, x, y):
        """Initialization of the Node.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
        """
        mygeo.Point.__init__(self, x, y)
        self.in_links = []
        self.out_links = []


class Link(mygeo.Segment):
    """The segment object for region generation.

    Attributes:    
        start (node): The starting node of a link.
        end (node): The ending node of a link.
        used (bool): Check whether this linke is used during the region generation process.

    """
    def __init__(self, start, end):
        """The segment consists of two nodes.

        Args:
            start (node): The starting node of a link.
            end (node): The ending node of a link.
        """
        mygeo.Segment.__init__(self, start, end)
        start.out_links.append(self)
        end.in_links.append(self)
        self.used = False
        
    def leftest(self):
        """Get the leftest outlink of the current link.

        Raises:
            error.RegionError: No leftest link.

        Returns:
            Link: The leftest Link.
        """
        min_cs = 5.0
        min_link = None
        for link in self.end.out_links:
            out_vec = link.vec()
            in_vec = self.vec()
            side1 = side(in_vec, out_vec)
            cos1 = cos(in_vec, out_vec)
            cs = side1 * cos1 + (1 - side1)
            if cs < min_cs:
                min_cs = cs
                min_link = link
        if min_link is None:
            raise error.RegionError("There is no leftest link.")
        return min_link
    
    def vec(self):
        """Vectorize the current link based on its start node and end node.

        Raises:
            error.RegionError: Empty link.

        Returns:
            tuple: A tuple of coordinates that represents the vector.
        """
        if self.is_empty():
            raise error.RegionError("Empty link cannot be vectorized.")
        return (self.end.x - self.start.x, self.end.y - self.start.y)

    def reverse(self):
        """Reverse the direction of a link.

        Returns:
            Link: The link that has the opposite direction of the current link.
        """
        return Link(self.end, self.start)


class ValueDict(object):
    """Check whether certain values are in this dictionary. If not, it will add them automatically.

    Important note. You will find almost all dictionaries in this package using OrderedDict instead of the common dict.
    This is because we want to reach out to the same result when we use either Python 2 environment or Python 3 environment.
    In Python 2, the default dict object does not record the insertion order, whereas in Python 3 the default dict does.
    This will influence the iteration process when we cluster points or generate regions.
    In order to keep the consistent final result, we choose to use the OrderedDict object as out default dictionary.

    Attributes:
        value_dict(dict): The dictionary to store everything.
    """
    def __init__(self):
        """Construct an empty dictionary.
        """
        self.value_dict = OrderedDict()

    def find(self, key, value=None):
        """Add a key to the dictionary. 

        If the key exists in this dictionary, we return the value of this key.
        Else we return the insertion value of the key.
        Args:
            key (optional): The key of the dictionary.
            value (optional, optional): Defaults to None.

        Returns:
            Optional: The corresponding value.
        """
        key_str = key
        if key_str in self.value_dict:
            return self.value_dict[key_str]
        else:
            if value is None:
                val = key
            else:
                val = value
            self.value_dict[key_str] = val
            return val

    def is_in(self, key, value=None):
        """Check whether the key is in the dictionary.

        If not, add the value of the key to this dictionary.
        If the value is also None, we add the value as the key.

        Args:
            key (optional): The key of the dictionary.
            value (optional, optional): Defaults to None.

        Returns:
            bool: True if the key exists in the dictioanry.
        """
        key_str = key
        if key_str in self.value_dict:
            return True
        else:
            if value is None:
                val = key
            else:
                val = value
            self.value_dict[key_str] = val
            return False
        

class RegionGenerator(object):
    """The main region generator. 

    Attributes:
        links (list): A list that stores all links we need to use during region generation.
        node_dict (ValueDict): Keep track of all nodes being used.
        link_dict (dict): Keep track of all links being used.
    """
    def __init__(self, segs):
        """Initialize the region generator with segments.

        Args:
            segs (list): A list of mygeo.Segment objects.
        """
        self.links = []
        self.node_dict = ValueDict()
        # self.link_dict = {}
        self.link_dict = OrderedDict()
        for seg in segs:
            start = self.node_dict.find(seg.start.askey(), Node(seg.start.x, seg.start.y))
            end = self.node_dict.find(seg.end.askey(), Node(seg.end.x, seg.end.y))
            if start == end:     # When the start node and the end node of the segment are very close to each other.
                continue
            link = self.__get_link(start, end)
            if link is not None:
                self.links.append(link)
                self.links.append(link.reverse())

    def __get_link(self, start, end):
        """Generate a link using the start node and the end node.

        Args:
            start (node): The start node.
            end (node): The end node.

        Returns:
            Link: The directional link.
        """
        ret = None
        segstr = mygeo.Segment(start, end).askey()
        if segstr not in self.link_dict:
            self.link_dict[segstr] = 1
            ret = Link(start, end)
        return ret
        
    def run(self):
        """Generate the region.

        The basic idea for us to generate the region is to recursively search for the leftest link.
        When the outlink reachs a inlink, a ring is formed.

        Returns:
            List: A list of regions.
        """
        ret = []
        for link in self.links:
            if not link.used:
                reg = self.get_a_region(link) 
                if reg is not None:
                    ret.append(reg)
        return ret
    
    def get_a_region(self, link):
        """Get a region starting with the input link.

        Args:
            link (Link): The starting link.

        Raises:
            error.RegionError: The start node and the end node should be the same.
            error.RegionError: Too many rings are counter clockwise.

        Returns:
            mygeo.Region: The region formed by finding the leftest outlink recursively.
        """
        nodes = [link.start]
        next = link
        while not next.used:
            nodes.append(next.end)
            next.used = True
            next = next.leftest()
        if not nodes[0] == nodes[-1]:
            raise error.RegionError("The first node is not the same as the end node.")
        points = None
        holes = []
        node_dict = {}
        temp_holes = []
        i = 0
        while i < len(nodes):
            nd = nodes[i]
            ndstr = nd.askey()
            if ndstr in node_dict:
                start_idx = node_dict[ndstr]
                sub_nodes = nodes[start_idx: i]
                if len(sub_nodes) >= 3:
                    region = mygeo.Region(sub_nodes)
                    if region.area() > 0:
                        if mygeo.is_counter_clockwise(sub_nodes):
                            if points is None:
                                points = sub_nodes
                            else:
                                raise error.RegionError("Too many rings are counter clockwise")
                        else:
                            holes.append(sub_nodes)
                    else:
                        temp_holes.append(sub_nodes)
                nodes = nodes[:start_idx] + nodes[i:]
                i = start_idx
            else:
                node_dict[ndstr] = i
            i += 1
        if points is not None:
            return mygeo.Region(points, holes)
        else:
            return None

                    
def segments_to_points(segments):
    """Get all distinct points from a list of segments.

    Args:
        segments (list): A list of mygeo.Segment objects.

    Returns:
        List: A list of mygeo.Point object.
    """
    ret = []
    point_dict = ValueDict()
    for seg in segments:
        if not point_dict.is_in(seg.start.askey(), seg.start):
            ret.append(seg.start)
        if not point_dict.is_in(seg.end.askey(), seg.end):
            ret.append(seg.end)
    return ret


def segments_to_cluster_points(segments):
    """Get all distinct cluster points from a list of segments.

    Note that the difference between this function and the previous one is that
    it generates a list of Cluster.Point objects.

    Args:
        segments (list): A list of mygeo.Segment objects.

    Returns:
        list: A list of Cluster.point objects.
    """
    ret = []
    point_dict = ValueDict()
    for seg in segments:
        if not point_dict.is_in(seg.start.askey(), seg.start):
            ret.append(cluster.Point(seg.start.x, seg.start.y))
        if not point_dict.is_in(seg.end.askey(), seg.end):
            ret.append(cluster.Point(seg.end.x, seg.end.y))
    return ret


def regions_to_points(regions):
    """Get all distinct vluster points from a list of regions.

    Args:
        regions (list): A list of mygeo.Point objects.

    Returns:
        list: A list of mygeo.Point objects.
    """
    ret = []
    point_dict = ValueDict()
    for region in regions:
        for pt in region.points:
            if not point_dict.is_in(pt.askey(), pt):
                ret.append(pt)
        for hole in region.holes:
            for pt in hole:
                if not point_dict.is_in(pt.askey(), pt):
                    ret.append(pt)
    return ret


def regions_to_cluster_points(regions):
    """Get all distinct cluster points from a list of regions.

    Note that the difference between this function and the previous one is that
    it generates a list of Cluster.Point objects.

    Args:
        segments (list): A list of mygeo.Segment objects.

    Returns:
        list: A list of Cluster.point objects.
    """
    ret = []
    id = 0
    point_dict = ValueDict()
    for region in regions:
        id += 1
        for pt in region.points:
            if not point_dict.is_in(pt.askey(), pt):
                ret.append(cluster.Point(pt.x, pt.y, id))
        for hole in region.holes:
            for pt in hole:
                if not point_dict.is_in(pt.askey(), pt):
                    ret.append(cluster.Point(pt.x, pt.y, id))
    return ret


def clusters_to_pointmap(clusters):
    """Generate a point dictionary from a list of clusters.

    Args:
        clusters (list): A list of clusters.

    Returns:
        ValueDict: The point dictionary.
    """
    point_dict = ValueDict()
    for cluster in clusters:
        center = cluster.center()
        for ptstr in cluster.points:
            point_dict.is_in(ptstr, center)
    return point_dict


def __segment_2_newseg(segment, point_dict, seg_dict):
    """Revise the precision of the segment and remove if it is in the segment dictionary.

    Args:
        segment (mygeo.Segment): The target segment.
        point_dict (ValueDict): The point dictionary.
        seg_dict (ValueDict): The segment dictionary.

    Returns:
        mygeo.Segment: The precision-revised segment.
    """
    start = point_dict.find(segment.start.askey(), segment.start)
    end = point_dict.find(segment.end.askey(), segment.end)
    start.trunc()
    end.trunc()
    if start == end:
        return None
    newseg = mygeo.Segment(start, end)
    if seg_dict.is_in(newseg.askey(), newseg):
        return None
    return newseg


def simplify_by_pointmap(segments, point_dict):
    """ Get all precision-revised segments from a list of segments and a list of regions.

    Args:
        segments (list): A list of segments.
        point_dict (ValueDict): A point dictionary.

    Returns:
        list: A list of all precision-revised segments.
    """
    ret = []
    seg_dict = ValueDict()
    for seg in segments:
        newseg = __segment_2_newseg(seg, point_dict, seg_dict)
        if newseg is not None:
            ret.append(newseg)
    return ret


class RegionAttract(object):
    """Attract and merge regions.

    Attributes:
        grid_dict (dict): Grid segmentations.
        regions (list): The list of regions that needs to attract and merge.
        point_map (ValueDict): Store the revise all points that being used.
        width (int): The merging threshold.

    """
    def __init__(self, regions, width):
        """Initialization.

        Args:
            regions (list): A list of regions that needs to attract and merge.
            width (int): The merging threshold.
        """
        grid_dict = {}
        points = regions_to_points(regions)
        point_map = ValueDict()
        for pt in points:
            grid = pt.grid(width)
            item = [pt, None, width]
            if point_map.is_in(pt.askey(), item):
                continue
            for grid_x in range(grid[0] - 1, grid[0] + 2):
                for grid_y in range(grid[1] - 1, grid[1] + 2):
                    key_grid = (grid_x, grid_y)
                    if key_grid in grid_dict:
                        grid_dict[key_grid].append(item)
                    else:
                        grid_dict[key_grid] = [item]
        self.grid_dict = grid_dict
        self.regions = regions
        self.point_map = point_map
        self.width = width

    def __modify_ring(self, points):
        """Revise the precision of every point on the ring.

        Args:
            points (list): A list of points.

        Returns:
            list: A list of revised points.
        """
        new_points = []
        for pt in points:
            item = self.point_map.find(pt.askey(), None)
            if item is None or item[1] is None:
                new_points.append(pt)
            else:
                new_points.append(item[1])
        return new_points
    
    def __project(self, seg, point):
        """Find the location of the projection point on a segment.

        The project() function first finds a point on the LineString which has the shortest distance to the point.
        Then, it calculate how much it goes from the begining of the LineString.
        The interpolate() function returns the point that travels a certain distance from the begining of the line.

        Args:
            seg (mygeo.Segment): The given undirectional segment.
            point (mygeo.Point): The given point.

        Returns:
            mygeo.Point: The projection point.
        """
        l1 = seg.linestring()
        d1 = l1.project(point.point())
        pt = l1.interpolate(d1)
        return mygeo.Point(pt.x, pt.y)

    def run(self, segs):
        """Merge the existing regions to the segments.

        Args:
            segs (list): A list of segments.

        Returns:
            List: A list of new regions.
        """
        # Calculate the point projection on the segment and the distance.
        for seg in segs:
            for grid in seg.grids(self.width):
                if grid not in self.grid_dict:
                    continue
                for item in self.grid_dict[grid]:
                    (pt, spt, min_dist) = item
                    dist = mygeo.pt_2_seg_dist(pt, seg)
                    if dist < min_dist:
                        item[1] = self.__project(seg, pt)
                        item[2] = dist
        # Construct based on the point_map.
        ret = []
        for reg in self.regions:
            points = self.__modify_ring(reg.points)
            holes = []
            for hole in reg.holes:
                holes.append(self.__modify_ring(hole))
            newreg = None
            try:
                newreg = mygeo.Region(points, holes)
            except error.RegionError:
                error.debug("The region is not valid: %s" % (str(reg)))
            if newreg is not None and newreg.polygon().is_valid:
                ret.append(newreg)
            else:
                ret.append(reg)
        return ret

