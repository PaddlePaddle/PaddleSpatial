# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 

import sys
import threading
from collections import OrderedDict

from genregion.region import geometry as mygeo
from genregion.region import error

class Link(mygeo.Segment):
    """The link object designed for some region operations.

    Attributes:
        twin_link (link): The link at the same position but with opposite direction.
        region (region): The region should always be at the left side of the link.

    """
    #__slots__ = ("twin_link", "region")

    def __init__(self, start, end):
        """Initialize the link object.

        Args:
            start (point): The start point.
            end (point): The end point.
        """
        mygeo.Segment.__init__(self, start, end)
        self.twin_link = None
        self.region = None
    
    def twin_link_region(self):
        """Get the region of the twin link.

        Returns:
            region: The left region of the twin link.
        """
        ret = None
        if self.twin_link is not None:
            ret = self.twin_link.region
        return ret


class Neighbor(object):
    """The neighbor region object.

    Attributes:
        region (region): A region in the neighborhood.
        points (list): A list of points that represents the contiguous border of two regions.
        length (float): The length of the contiguous border of two regions.
    """
    #__slots__ = ("region", "points", "length")

    def __init__(self, region, points, length):
        """Initialize the neighbor object.

        Args:
            region (region): One region in the neighborhood.
            points (list): A list of points that represent the contiguous border of two regions.
            length (float): The length of the contiguous border of two regions.
        """
        self.region = region
        self.points = points
        self.length = length

    def merge(self, points, length):
        """Merge a list of points to this neighbor.

        Note that the merging process can only happen when the last point of the neighbor is the last
        point of the new list of points.

        Args:
            points (list): A list of points that waited to be merged.
            length (float): The original length of the contiguous border.

        Raises:
            error.RegionError: Two lists of points are not connected.
        """
        if not points[-1] == self.points[0]:
            raise error.RegionError("Two lists of points are not connected.")
        newpoints = []
        newpoints.extend(points)
        newpoints.extend(self.points[1:])
        self.points = newpoints
        self.length += length

    def display(self):
        """Print the information of this neighbor.
        """
        print("neighbor")
        if self.region is not None:
            print("\tregion: ", str(self.region))
        else:
            print("\tregion: None")
        print("\tPoint: ")
        for pt in self.points:
            print(str(pt), ", ")
        print("\tlength: ", self.length)


class LinkDict(object):
    """The dictionary for links.

    Attributes:
        link_dict (dict): A dictionary contains all links.

    """
    #__slots__ = ("link_dict", )

    def __init__(self):
        """Initialize the LinkDict with an empty dictionary.
        """
        # self.link_dict = {}
        self.link_dict = OrderedDict()

    def clear(self):
        """Clear the dictionary.
        """
        # self.link_dict = {}
        self.link_dict = OrderedDict()

    def get_link(self, start, end):
        """Get or add a link using the start and the end point.

        Note that we first try to find the link in the dictionary and return it.
        If it is not existed in the dictionary, we add it to the dictionary and return it.

        Args:
            start (mygeo.Point): The start point.
            end (mygeo.Point): The end point.

        Returns:
            link: A directional link.
        """
        ret = None
        link = Link(start, end)
        linkstr = link.askey()
        if linkstr in self.link_dict:
            ret = self.link_dict[linkstr]
            if not start == ret.start:
                ret = ret.twin_link
        else:
            twin_link = Link(end, start)
            link.twin_link = twin_link
            twin_link.twin_link = link
            self.link_dict[linkstr] = link
            ret = link
        return ret

    def del_link(self, start, end):
        """Delete a link from the link dictionary.

        Args:
            start (mygeo.Point): The start point.
            end (mygeo.Point): The end point.
        """
        linkstr = Link(start, end).askey()
        if linkstr in self.link_dict:
            del self.link_dict[linkstr]

    def segments(self):
        """Get all undirectional segments from the link dictionary.

        Returns:
            list: A list of segments.
        """
        ret = []
        for link in self.link_dict.values():
            ret.append(mygeo.Segment(link.start, link.end))
        return ret


def points_2_segs(points):
    """Convert a list of points into a list of segments.

    Args:
        points (list): A list of mygeo.Point objects.

    Returns:
        list: A list of segments.
    """
    ret = []
    lastpt = points[0]
    for pt in points[1:]:
        ret.append(mygeo.Segment(lastpt, pt))
        lastpt = pt
    return ret


class Region(mygeo.Region):
    """The region object designed to facilitate some region operations.

    Attributes:
        points (list): A list of points that indicates the outer loop of the regions.
        holes (list): A list of lists of points that indicates holes.
        link_dict (LinkDict): A dictionary of links that stores all links we need.
        parent_region (region): The original region that contains this region.
        links (list): A list of lists of links.
        area (float): The area of this region.
        length_ (float): The perimeter of this region.
        width_ (float): A width that trying to indicate whether the region is too narrow.
        neighbors (list): A list of neighbors.
    """
    #__slots__ = ("link_dict", "parent_region", "links", "area_", "length_", "width_", \
                 #"neighbors")

    def __init__(self, region, parent_region, link_dict):
        """Initialize the region.

        Args:
            region (mygeo.Region): The mygeo.Region object used to initialize the region.
            parent_region (region): The region that originally contains this region.
            link_dict (LinkDict): A dictionary that stores all needed links.
        """
        mygeo.Region.__init__(self, region.points, region.holes)
        self.link_dict = link_dict
        self.parent_region = parent_region
        self.links = self.__get_links()
        self.area_ = region.area()
        self.length_ = region.length()
        self.width_ = self.area_ / self.length_
        self.neighbors = None

    def __trans_points_to_links(self, points):
        """Convert a list of points to a list of links.

        Args:
            points (list): A list of links.

        Returns:
            list: A list of links.
        """
        ret = []
        lastpt = points[-1]
        for pt in points:
            link = self.link_dict.get_link(lastpt, pt)
            link.region = self
            ret.append(link)
            lastpt = pt
        return ret

    def __get_links(self):
        """Convert a region in to links.

        Returns:
            list: A list of lists of links that represents the region and it's holes.
        """
        links = self.__trans_points_to_links(self.points)
        ret = [links]
        for hole in self.holes:
            links = self.__trans_points_to_links(hole)
            ret.append(links)
        return ret
        
    def __compute_ring_neighbors(self, links):
        """Get all neighbors of a list of links acquired from a ring.

        Args:
            links (list): A list of links acquired from a ring.

        Returns:
            list: A list of neighbors. 
        """
        ret = []
        if len(links) == 0:
            return ret
        points = []
        length = 0
        last_region = -1
        for link in links:
            linklen = link.length()
            region = link.twin_link_region()
            if region == last_region:
                points.append(link.end)
                length += linklen
            else:
                if last_region != -1:
                    ret.append(Neighbor(last_region, points, length))
                points = [link.start, link.end]
                length = linklen
                last_region = region
        if last_region != -1:
            if len(ret) > 1 and last_region == ret[0].region:
                ret[0].merge(points, length)
            else:
                ret.append(Neighbor(last_region, points, length))
        return ret
        
    def refresh_neighbors(self):
        """Get all neighbors of this region.
        """
        ret = []
        for links in self.links:
            neighbors = self.__compute_ring_neighbors(links)
            ret.extend(neighbors)
        self.neighbors = ret

    def is_small_region(self, area_thres, width_thres):
        """Check whether the current region is a small region.

        There are two types of small regions.
        The first type is the one that has a very small area.
        The second type is the one that is so narrow that can be considered as a road in reality.

        Args:
            area_thres (int): The threshold of a small area.
            width_thres (int): The threshold of a narrow width

        Returns:
            bool: True if it is a small region.
        """
        return self.area_ <= area_thres or self.width_ <= width_thres
            
    def merge_small_region(self, area_thres, width_thres):
        """Get all segments that need to be removed from this region.

        The main purpose of this function is to merge a small region to a larger one, 
        or merge a small region to another small region to get a larger one.
        A very simple way of merging two regions is to remove their contiguous border.
        So, we first try to find a non-small neighbor which has the largest contiguous border of his region.
        If the region does not have a non-small neighbor, we find the neighbor that has the largest contiguous border.

        Args:
            area_thres (int): The threshold of the area.
            width_thres (int): The threshold of the width.

        Returns:
            list: A list of segments that denotes the contiguous border.
        """
        small = True   # The current region is a small region.
        max_length = 0
        max_region = None
        for neighbor in self.neighbors:
            if neighbor.region is None:
                continue
            if not neighbor.region.is_small_region(area_thres, width_thres):
                if small or neighbor.length > max_length: # Find the region that has the largest length.
                    small = False
                    max_length = neighbor.length
                    max_region = neighbor.region
            elif small and neighbor.length > max_length: 
                max_length = neighbor.length
                max_region = neighbor.region
        if max_region is not None:
            for nb in self.neighbors:
                if nb.region == max_region:
                    return points_2_segs(nb.points)
        return None
    
    def merge_sibling_region(self):
        """Get all segments of all sibling regions of this region.

        Returns:
            list: A list of segmetns.
        """
        if self.parent_region is None:
            return None
        ret = []
        for nb in self.neighbors:
            if nb.region is not None and nb.region.parent_region == self.parent_region:
                ret.extend(points_2_segs(nb.points))
        return ret



def build_region_dict(regions, grid_size, key=None):
    """Construct the dictionary of grids and regions.

    The key is the grid, and the corresponding value is a list of regions that appears on that grid.

    Args:
        regions (list): A list of regions for us to construct the dictionary.
        grid_size (int): The size of the grid.
        key (tuple, optional): The integer coordinates of the grid. Defaults to None.

    Returns:
        dict: A dictionary that helps us find regions according to the grid information.
    """
    if key is None:
        key = lambda x: x
    grid_dict = OrderedDict()
    for reg in regions:
        for grid in key(reg).grids(grid_size):
            if grid in grid_dict:
                grid_dict[grid].append(reg)
            else:
                grid_dict[grid] = [reg]
    return grid_dict


class RegionMerger(object):
    """Merge small regions to larger ones.

    Attributes:
        regions (list): A list of regions that needed to be merged.
        raw_regions (list): A list of raw regions. However, we temporarily shut down this function.
        raw_region_dict (dict): A dictionary that stores all raw regions, and we do not need to use it.
        link_dict (LinkDict): A dictioanry which stores all links.

    """
    def __init__(self, regions, raw_regions):
        """Initialize the region merger.

        Args:
            regions (list): A list of regions waiting to be merged.
            raw_regions (list): We temporarily do not need to consider this.
        """
        self.regions = regions
        self.raw_regions = raw_regions
        self.raw_region_dict = None
        self.link_dict = LinkDict()

    def __get_raw_region(self, region, grid_size):
        """We temporarily do not need to consider this function.
        """
        if self.raw_region_dict is None:
            return None
        center = region.center()
        grid = center.grid(grid_size)
        if grid in self.raw_region_dict:
            raw_regions = self.raw_region_dict[grid]
            for rawreg, mbr in raw_regions:
                if mbr[1].x >= center.x and mbr[1].y >= center.y \
                   and mbr[0].x <= center.x and mbr[0].y <= center.y:
                    if rawreg.polygon().covers(center.point()):
                        return rawreg
        return None

    def __trans_regions(self, grid_size):
        """Transform the region type.

        Args:
            grid_size (int): The grid size.

        Returns:
            list: A list of transformed regions.
        """
        ret = []
        if self.raw_regions is not None:
            raw_regions = [(r, r.mbr()) for r in self.raw_regions]
            self.raw_region_dict = build_region_dict(raw_regions, grid_size, lambda x:x[0])
        for region in self.regions:
            raw_region = self.__get_raw_region(region, grid_size)
            newreg = Region(region, raw_region, self.link_dict)
            ret.append(newreg)
        for reg in ret:
            reg.refresh_neighbors()
        return ret

    def run(self, grid_size, area_thres, width_thres):
        """Merge regions based on given thresholds.

        Args:
            grid_size (int): The size of the grid.
            area_thres (int): Any region that has a smaller area than this will be merged.
            width_thres (int): Any region that has a smaller width than this will be merged.

        Returns:
            list: A list of segments that constructs those new regions.
        """
        regions = self.__trans_regions(grid_size) # mygeo.Region -> Region
        for region in regions:
            segs = region.merge_sibling_region()
            if segs is not None:
                for seg in segs:
                    self.link_dict.del_link(seg.start, seg.end)
            elif region.is_small_region(area_thres, width_thres):
                segs = region.merge_small_region(area_thres, width_thres)
                if segs is not None:
                    for seg in segs:
                        self.link_dict.del_link(seg.start, seg.end)
        return self.link_dict.segments()


class RegionFilter(object):
    """Filter out subregions among other regions.

    Attributes:
        regions (list): A list of regions.
    """
    def __init__(self, regions):
        """Initialize the filter.

        Args:
            regions (list): A list of regions that waited to be filtered.
        """
        self.regions = regions

    def __delete_contain(self, grid_dict):
        """Delete subregions.

        If more than half of the region is contained in the other region,
        we also consider it as the subregion.

        Args:
            grid_dict (dict): The grid dictionary that helps locate the region.
        """
        total = len(grid_dict)
        for dict_regions in grid_dict.values():
            regions = [r for r in dict_regions if not r[0].is_empty()]
            regions.sort(key=lambda x: x[1])
            for i in range(len(regions) - 1):
                child_reg, child_area, child_mbr = regions[i]
                for (parent_reg, parent_area, parent_mbr) in regions[i + 1:]:
                    if parent_mbr[1].x >= child_mbr[1].x and parent_mbr[1].y >= child_mbr[1].y \
                      and parent_mbr[0].x <= child_mbr[0].x and parent_mbr[0].y <= child_mbr[0].y:
                        try:
                            if (parent_reg.polygon().covers(child_reg.polygon())):
                                child_reg.destroy()
                        except error.RegionError:
                            continue
            
    def run(self, grid_size):
        """Filter the region.

        Args:
            grid_size (int): The size of the grid.

        Returns:
            list: A list of filtered regions.
        """
        # grid_dict = {}
        grid_dict = OrderedDict()
        for reg in self.regions:
            area = reg.area()
            mbr = reg.mbr()
            for grid in reg.grids(grid_size):
                if grid in grid_dict:
                    grid_dict[grid].append((reg, area, mbr))
                else:
                    grid_dict[grid] = [(reg, area, mbr)]
        self.__delete_contain(grid_dict)
        return [r for r in self.regions if not r.is_empty()]

    def __thread_start(self, func, grid_dict_list):
        """Start a thread for a function.

        Args:
            func (function): The function we want to run.
            grid_dict_list (list): A list of grids we want to input into the function.
        """
        thds = []
        for gd in grid_dict_list:
            t = threading.Thread(target=func, args=(gd,))
            t.start()
            thds.append(t)
        for t in thds:
            t.join()

    def multi_run(self, grid_size, thread_num=4):
        """Run the region filter through multiple threads.

        Args:
            grid_size (int): The size of the grid.
            thread_num (int, optional): Number of threads we want to use. Defaults to 4.

        Returns:
            list: A list of filtered regions.
        """
        grid_dict_list = []
        for i in range(thread_num):
            # grid_dict_list.append({})
            grid_dict_list.append(OrderedDict())
        for reg in self.regions:
            area = reg.area()
            mbr = reg.mbr()
            for grid in reg.grids(grid_size):
                idx = (grid[0] + grid[1]) % thread_num
                grid_dict = grid_dict_list[idx]
                if grid in grid_dict:
                    grid_dict[grid].append((reg, area, mbr))
                else:
                    grid_dict[grid] = [(reg, area, mbr)]
        self.__thread_start(self.__delete_contain, grid_dict_list)
        return [r for r in self.regions if not r.is_empty()]


class RegionFinder(object):
    """Find the id of the region among a list of regions.

    Attributes:
        grid_size (int): The size of the grid.
        region_dict (dict): The region dictioanry.
    """
    def __init__(self, regions, grid_size):
        """
        Desc: 初始化

        Args:
            self : self

            regions : list of (id, region)

            grid_size : 网格大小
        Return: 
            None 
        Raises: 
            None
        """
        self.grid_size = grid_size
        self.region_dict = build_region_dict(regions, grid_size, key=lambda x: x[1])

    def find_region(self, point):
        """Find the id of the region that the given point is located at.

        Args:
            point (point): The given point.

        Returns:
            optional: The id of the region is determined by the user.
        """
        grid = point.grid(self.grid_size)
        if grid not in self.region_dict:
            return None
        for id, region in self.region_dict[grid]:
            if region.polygon().covers(point.point()):
                return id
        return None


