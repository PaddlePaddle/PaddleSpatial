# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

import sys
import gc
from collections import OrderedDict

from genregion.region import geometry as mygeo
from genregion.region import error

class SegSpliter(object):
    """The segments breaker.

    The main function of this object is to break segments into smaller pieces so that
    a segment only connects to others at its two end points. 

    Attributes:
        segs (list): A list of original segments that need to be processed.
        valids (list): A list of indicators that records the validity of each segment.
        grid_size (int): The size of the grid.
        grid_dict (dict): The dictionary that records grid information.
    """
    def __init__(self, segs, grid_size):
        """Initilize the spliter object.

        Args:
            segs (list): A list of original segments.
            grid_size (int): The size of the grid.
        """
        self.segs = segs
        self.valids = [ True for seg in segs ]
        self.grid_size = grid_size
        self.grid_dict = None
        
    def __proc_seg_by_seg(self, i1, i2):
        """Break two segments based on their intersection.
        
        After we find the intersection of two segments, we break both of them.
        Then we set the indicator for orginal segments to false, and add new pieces to the list.

        Args:
            i1 (int): The index of the first segment.
            i2 (int): The index of the second segment.

        Returns:
            list: New segments.
        """
        seg1 = self.segs[i1]
        seg2 = self.segs[i2]
        ret = []
        points = seg1.intersect(seg2)
        if len(points) > 0:
            for pt in points:
                pt.trunc()
            segs = self.__split_seg_by_points(seg1, points)
            if len(segs) > 1:
                ret.extend(segs)
                self.valids[i1] = False
            segs = self.__split_seg_by_points(seg2, points)
            if len(segs) > 1:
                ret.extend(segs)
                self.valids[i2] = False
        return ret

    def __split_seg_by_points(self, seg, points):
        """Break a segment into pieces based on a list of points.

        Args:
            seg (segment): The segment that needs to be broken.
            points (list): A list of points that should be on the segment.

        Returns:
            list: A list of broken segments.
        """
        pts = []
        line = seg.linestring() # Make the line directional
        for pt in points:
            if pt == seg.start or pt == seg.end:
                continue
            sppt = pt.point()
            dist = line.project(sppt) # Calculate the distance from the begining.
            pts.append((pt, dist))
        pts.sort(key=lambda x: x[1])
        pts.append((seg.end, 0))
        last_point = seg.start
        ret = []
        for pt, dist in pts:
            if last_point == pt:
                continue
            newseg = mygeo.Segment(last_point, pt)
            ret.append(newseg)
            last_point = pt
        return ret

    def __add_dict(self, indexes):
        """Add new segments to the grid dictionary.

        Args:
            indexes (list): A list of indexes indicating positions of segments in segs.
        """
        for i in indexes:
            seg = self.segs[i]
            for grid in seg.grids(self.grid_size):
                if grid in self.grid_dict:
                    self.grid_dict[grid].append(i)
                else:
                    self.grid_dict[grid] = [i]

    def __get_intersect_segs(self, i1):
        """Find all segments that intersect with a segment, and break them.

        Args:
            i1 (int): The index of the segment.

        Returns:
            list: A list of new segments.
        """
        ret = []
        seg = self.segs[i1]
        for grid in seg.grids(self.grid_size):
            if grid in self.grid_dict:
                for i2 in self.grid_dict[grid]:
                    seg2 = self.segs[i2]
                    if self.valids[i2]:
                        segs = self.__proc_seg_by_seg(i1, i2)
                        if len(segs) > 0:
                            ret.extend(segs)
                            if not self.valids[i1]:
                                return ret
        return ret

    def run(self):
        """Break all segments and filter out repeated ones.

        Returns:
            list: A list of new segments that only connect to others at their two end nodes.
        """
        # self.grid_dict = {}
        self.grid_dict = OrderedDict()
        i = 0
        while i < len(self.segs):
            if self.valids[i]:
                res_segs = self.__get_intersect_segs(i)
                if len(res_segs) > 0:
                    self.segs.extend(res_segs)
                    self.valids.extend([ True for rs in res_segs])
                if self.valids[i]:
                    self.__add_dict([i])
            i += 1
        ret = []
        seg_set = set([])
        for i in range(len(self.segs)):
            if self.valids[i]:
                seg = self.segs[i]
                seg_key = seg.askey()
                if seg_key not in seg_set:
                    seg_set.add(seg_key)
                    ret.append(seg)
        return ret
