# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################

import sys
from collections import OrderedDict
from ordered_set import OrderedSet

from genregion.region import error
from genregion.region import geometry as mygeo

class Point(mygeo.Point):
    """Point object for clustering.
    
    Attributes:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        id: The point id.
    """
    #__slots__ = ("id", )

    def __init__(self, x, y, id=None):
        mygeo.Point.__init__(self, x, y)
        self.id = id

class Cluster(object):
    """object that records clustering information.

    Attributes:
        minx (float): The smallest x-coordinate among all points in this cluster.
        maxx (float): The largest x-coordinate among all points in this cluster.
        miny (float): The smallest y-coordinate among all points in this cluster.
        maxy (float): The largest x-coordinate among all points in this cluster.
        points (set): Contains all points in this cluster.
        ids (set): Contains all point ids in this cluster.
    """
    #__slots__ = ("minx", "maxx", "miny", "maxy", "points", "ids")

    def __init__(self):
        """A cluster object contains information about MBR and points.
        """
        self.minx = None
        self.maxx = None
        self.miny = None
        self.maxy = None
        # self.points = set([])
        # self.ids = set([])
        self.points = OrderedSet([])
        self.ids = OrderedSet([])

    def count(self):
        """Count the number of the points in this cluster.

        Returns:
            int: The number of the points.
        """
        return len(self.points)

    def destroy(self):
        """Destroy the cluster.
        """
        self.points.clear()
        self.ids.clear()
        self.minx = self.maxx = self.miny = self.maxy = None

    def is_empty(self):
        """Check whether the cluster is empty.

        Returns:
            bool: True if empty.
        """
        return self.minx is None

    def add_point(self, point, is_vip=True):
        """Add a new point to this cluster.

        Args:
            point (Point): The new point.
            is_vip (bool, optional): It is used to calculate the center of the cluster. Defaults to True.
        """
        self.points.add(point.askey())
        if is_vip:
            if self.minx is None or point.x < self.minx:
                self.minx = point.x
            if self.maxx is None or point.x > self.maxx:
                self.maxx = point.x
            if self.miny is None or point.y < self.miny:
                self.miny = point.y
            if self.maxy is None or point.y > self.maxy:
                self.maxy = point.y
        if point.id is not None:
            self.ids.add(point.id)

    def merge(self, cl):
        """Merge a new cluster to the current one.

        Args:
            cl (Cluster): The other cluster.

        Raises:
            error.RegionError: Id conflict.
        """
        if cl.is_empty():
            return
        if not self.can_merge(cl):
            raise error.RegionError("Cluster id conflict.")
        if self.is_empty():
            self.minx, self.maxx, self.miny, self.maxy = cl.minx, cl.maxx, cl.miny, cl.maxy
        else:
            self.minx = min(self.minx, cl.minx)
            self.maxx = max(self.maxx, cl.maxx)
            self.miny = min(self.miny, cl.miny)
            self.maxy = max(self.maxy, cl.maxy)
        for pt in cl.points:
            self.points.add(pt)
        for id in cl.ids:
            self.ids.add(id)

    def can_merge(self, cl):
        """Check whether one cluster can be merged with the current one. 

        Args:
            cl (Cluster): The other cluster.

        Returns:
            bool: True if possible.
        """
        for id in cl.ids:
            if id in self.ids:
                return False
        return True

    def in_cluster(self, point):
        """Check whether a point is in the current cluster.

        Args:
            point (Point): The checking point.

        Returns:
            bool: True if contained.
        """
        return (point.askey() in self.points)

    def center(self):
        """Get the MBR center of all vip points as the cluster center.

        Raises:
            error.RegionError: Empty cluster.

        Returns:
            mygeo.Point: A point contains the 2D coordinates.
        """
        if self.is_empty():
            raise error.RegionError("Empty cluster has no center.")
        return mygeo.Point((self.minx + self.maxx) / 2, (self.miny + self.maxy) / 2)

    def width_x(self):
        """Calculate the horizontal width of the cluster.

        Raises:
            error.RegionError: Empty cluster.

        Returns:
           float: Width. None if empty.
        """
        if self.is_empty():
            raise error.RegionError("Empty cluster has no width_x.")
        return self.maxx - self.minx

    def width_y(self):
        """Calculate the vertical width of the cluster.

        Raises:
            error.RegionError: Empty cluster.

        Returns:
            float: Width. None if empty.
        """
        if self.is_empty():
            raise error.RegionError("Empty cluster has no width_y.")
        return self.maxy - self.miny


def is_friend_dist(dist, width):
    """Check whether the width is larger than the threshold.

    Args:
        dist (tuple): (width_x, width_y)
        width (int): The threshold.

    Returns:
        bool: True if possible.
    """
    return dist[0] <= width and dist[1] <= width


class HCCluster(Cluster):
    """Hierarchical clustering object.

    Attributes:
    minx (float): The smallest x-coordinate among all points in this cluster.
    maxx (float): The largest x-coordinate among all points in this cluster.
    miny (float): The smallest y-coordinate among all points in this cluster.
    maxy (float): The largest x-coordinate among all points in this cluster.
    points (set): Contains all points in this cluster.
    ids (set): Contains all point ids in this cluster.
    friends (list): Contains all friend clusters.

    """
    def __init__(self):
        """Initialization of a hierarchical cluster.

        In this object, we add a new variable called 'friend'. The friends of a cluster are its neighbors.
        Two clusters are friends if they merge into one cluster, of which each side is less than the threshold.

        """
        Cluster.__init__(self)
        # (cluster, dist)
        # The best friend is self.friends[0]
        self.friends = []

    def collect_best_friend(self):
        """Move the best friends to the begining of the friends list. 

        Note that there might be more than one best friends for each hccluster.

        """
        min_weight = sys.float_info.max
        for friend in self.friends:
            weight = friend[1]
            if weight < min_weight:
                min_weight = weight
        swap_count = 0
        i = 0
        while i < len(self.friends):
            if self.friends[i][1] == min_weight:
                self.friends[swap_count], self.friends[i] = self.friends[i], self.friends[swap_count]
                swap_count += 1
            i += 1

    def del_friends(self, del_frds):
        """Remove a list of friends from the current friends list.

        If the first item of the friend list get removed, we need to elect best friends again.

        Args:
            del_frds (list of friends): Unwanted friends.

        Returns:
            bool: True if best friends are reelected.
        """
        i = 0
        first_del = False
        while i < len(self.friends): 
            hc, weight = self.friends[i]
            if hc in del_frds:
                self.friends.pop(i)
                i -= 1
                if i == 0:
                    first_del = True
            i += 1
        if first_del and len(self.friends) > 1:
            self.collect_best_friend()
        return first_del

    def bfs_merge(self, width):
        """Check the condition and merge the cluster with its best friends.

        Two clusters will be merged if they are best friends of each other.

        Args:
            width (int): Merging threshold.

        Returns:
            bool: True if the merging process happens.
        """
        best_frds = self.best_friends()
        for hc, weight in best_frds:
            frds = hc.best_friends()
            for (hc2, weight) in frds:
                if self == hc2:
                    self.merge(hc, width)
                    return True
        return False

    def merge(self, hc, width):
        """Merge two hccluster objects together.
        
        Merge the current cluster with the other one. 
        Update the friends list of the combined cluster.

        Args:
            hc (hccluster): The merging hccluster.
            width (int): The merging threshold.
        """
        Cluster.merge(self, hc)
        new_friends = OrderedSet([])
        for fhc, weight in self.friends:
            new_friends.add(fhc)
        for fhc, weight in hc.friends:
            new_friends.add(fhc)
        hc.destroy()
        self.friends = []
        for fhc in new_friends:
            if fhc == self or fhc == hc:
                continue
            fhc.del_friends([self, hc])
            dist = self.distance(fhc, width)
            if is_friend_dist(dist, width) and self.can_merge(fhc):
                self.add_friend(fhc, dist)
                fhc.add_friend(self, dist)

    def best_friends(self):
        """Find all best friends of the current cluster.

        Returns:
            List: Best friends.
        """
        ret = []
        if len(self.friends) > 0:
            min_weight = self.friends[0][1]
            for friend in self.friends:
                if min_weight == friend[1]:
                    ret.append(friend)
        return ret

    def distance(self, hc, width):
        """Calculate the distance between two HCCluster objects.

        Definition of the distance between two clusters:
            If the widths of cluster A are (xa, ya), and the widths of cluster B are (xb, yb), 
            then the widths of the merged cluster C are (xc, yc), and they should both less than the width.
            So, the distance should be defined as:
                if xa + ya > xb + yb: 
                    delta_x = xc - xa
                    delta_y = yc - ya 
                else
                    delta_x = xc - xb
                    delta_y = yc - yb 

        Args:
            hc (hccluster): The merging hccluster.
            width (int): The merging threshold.

        Raises:
            error.RegionError: Empty object.

        Returns:
            tuple: (width_x, width_y)
        """
        if hc.is_empty() or self.is_empty():
            raise error.RegionError("Empty object has no distance.")
        minx = min(self.minx, hc.minx)
        maxx = max(self.maxx, hc.maxx)
        miny = min(self.miny, hc.miny)
        maxy = max(self.maxy, hc.maxy)
        width_x = maxx - minx
        width_y = maxy - miny
        if width_x <= width and width_y <= width:
            if self.width_x() + self.width_y() > hc.width_x() + hc.width_y():
                return (width_x - self.width_x(), width_y - self.width_y())
            else:
                return (width_x - hc.width_x(), width_y - hc.width_y()) 
        return (width + 1, width + 1)

    def add_friend(self, hc, dist):
        """Add a new cluster to the friends list.

        Args:
            hc (hccluster): The new cluster.
            dist (Tuple): Distance.
        """
        (dist_x, dist_y) = dist
        weight = dist_x + dist_y
        if len(self.friends) > 0 and weight <= self.friends[0][1]:
            self.friends.insert(0, (hc, weight))
        else:
            self.friends.append((hc, weight))

    def display(self):
        """Print detailed information about the current cluster.
        """
        print("hc cluster: ")
        for pt in self.points:
            print(pt, ",")
        for fhc, weight in self.friends:
            print("friend weight: ", weight, " : ")
            for pt in fhc.points:
                print(pt, ",")
        print("mbr: (%f, %f), (%f, %f)" % (self.minx, self.miny, self.maxx, self.maxy))


class HCAlgorithm(object):
    """The main object to run the hierarchical clustering algorithm.

    Attributes:
        width (int): Clustering threshold.
        hcclusters (list): A list of hcclusters that perform the clustering algorithm.
        grids (dict): Record all grids that being used for this algorithm.

    """
    def __init__(self, width):
        """The implementation of the clustering algorithm.

        Args:
            width (int): Clustering threshold.
        """
        self.width = width
        # Intermediate results of the clustering algorithm.
        self.hcclusters = []
        # dict of (grid_id, HCCluster list)
        # self.grids = {}
        self.grids = OrderedDict()

    # Build for debugging
    def output_points(self, points, path="seg_points.txt"):
        """Output all points of a given list of points.

        Args:
            points (list): segment points
            path (str, optional): . Defaults to "seg_points.txt".

        """
        with open(path, "w") as fw:
            for pt in points:
                fw.write(str(pt.askey())+'\n')

    # Build for debugging
    def output_grids(self, path="grid_id.txt"):
        """Output all grid ids.

        Args:
            path (str, optional): Defaults to "grid_id.txt".
        """
        with open(path, "w") as fw:
            for grid_id in self.grids:
                fw.write(str(grid_id) + "\n")

    # Build for debugging
    def output_clusters(self, ret, path="cluster.txt"):
        """Output all clusters.

        Args:
            ret (list): A list of hcc clusters.
            path (str, optional): Defaults to "cluster.txt".
        """
        with open(path, "w") as fw:
            for hc in ret:
                fw.write("%f, %f, %f, %f, %d\n" % (hc.minx, hc.maxx, hc.miny, hc.maxy, len(hc.points)))

    def run(self, points):
        """The implementation of the hierarchical clustering algorithm.
        Args:
            points (list): A list of mygeo.Point ready to be clustered.

        Returns:
            list: A list of clusters.
        """
        # Make each point as the initial cluster.
        self.build_clusters(points)
        # Construct the friendship between each pair of clusters.
        self.build_friends()
        # Deep first merge.
        merge_count = 1
        while merge_count > 0:
            merge_count = 0
            for hc in self.hcclusters:
                if not hc.is_empty():
                    if hc.bfs_merge(self.width):
                        merge_count += 1
        # Append each cluster to the list.
        ret = []
        for hc in self.hcclusters:
            if not hc.is_empty():
                ret.append(hc)
        # error.debug("Number of grids: %d" % len(self.grids)) 
        # wrting grid_id for debugging
        # self.output_grids()
        # self.output_clusters(ret)
        # self.output_points(points)
        return ret
        
    def build_clusters(self, points):
        """Construct initial clusters based on each point and assign them to each grid.

        Args:
            points (list): A list of mygeo.Point.
        """
        for pt in points:
            hc = HCCluster()
            hc.add_point(pt)
            self.hcclusters.append(hc)
            grid = pt.grid(self.width)    # grid is a tuple (grid_x, grid_y) 
            if grid in self.grids:
                self.grids[grid].append(hc)
            else:
                self.grids[grid] = [hc]
    
    def build_friends(self):
        """Construct the friendship among adjacent grids.
        """
        for grid in self.grids:
            grid_x, grid_y = grid
            self.build_friend_one_grid(grid)
            self.build_friend_two_grid(grid, (grid_x - 1, grid_y + 1))
            self.build_friend_two_grid(grid, (grid_x + 1, grid_y))
            self.build_friend_two_grid(grid, (grid_x, grid_y + 1))
            self.build_friend_two_grid(grid, (grid_x + 1, grid_y + 1))

    def build_friend_one_grid(self, grid):
        """Two clusters in the same grid must be friends.

        Args:
            grid (tuple): (grid_x, grid_y). Horizontal and vertical ids that identify a grid.
        """
        hcs = self.grids[grid]
        count = len(hcs)
        for i in range(count - 1):
            hc1 = hcs[i]
            for j in range(i + 1, count):
                hc2 = hcs[j]
                dist = hc1.distance(hc2, self.width)
                if is_friend_dist(dist, self.width) and hc1.can_merge(hc2):
                    hc1.add_friend(hc2, dist)
                    hc2.add_friend(hc1, dist)

    def build_friend_two_grid(self, grid1, grid2):
        """Build friendship between all clusters in each grid.

        Args:
            grid1 (tuple): (grid_x, grid_y).
            grid2 (tuple): (grid_x, grid_y).
        """
        if grid1 not in self.grids or grid2 not in self.grids:
            return
        hcs1 = self.grids[grid1]
        hcs2 = self.grids[grid2]
        for hc1 in hcs1:
            for hc2 in hcs2:
                dist = hc1.distance(hc2, self.width)
                if is_friend_dist(dist, self.width) and hc1.can_merge(hc2):
                    hc1.add_friend(hc2, dist)
                    hc2.add_friend(hc1, dist)
        

class Classifier(object):
    """ Put the point into a cluster according to the distance.

    Attributes:
        width (int): The clustering threshold.
        grids (int): All grids that being used.

    """
    def __init__(self, clusters, width):
        """Initialization.

        Args:
            clusters (list): A list of clusters.
            width (int): Threshold.
        """
        self.width = width
        self.grids = {}
        for cluster in clusters:
            center = cluster.center()
            lb = mygeo.Point(center.x - width / 2.0, center.y - width / 2.0)
            rt = mygeo.Point(center.x + width / 2.0, center.y + width / 2.0)
            lb_grid = lb.grid(width)
            rt_grid = rt.grid(width)
            for x in range(lb_grid[0], rt_grid[0] + 1):
                for y in range(lb_grid[1], rt_grid[1] + 1):
                    grid = (x, y)
                    if grid in self.grids:
                        self.grids[grid].append(cluster)
                    else:
                        self.grids[grid] = [cluster]
    
    def run(self, points):
        """Implementation of the distribution process.

        Args:
            points (list): A list of mygeo.Point objects.

        Returns:
            list: A list of mygeo.Point objects which cannot be classified.
        """
        left_points = []
        for pt in points:
            if not self.classify_point(pt):
                left_points.append(pt)
        return left_points

    def classify_point(self, point):
        """Classify the point to a cluster. 

        Args:
            point (mygeo.Point): The target point.

        Returns:
            bool: True if the point can be classified.
        """
        grid = point.grid(self.width)
        ret = False
        if grid in self.grids:
            clusters = self.grids[grid]
            max_count = 0
            min_cluster = None
            for cluster in clusters:
                center = cluster.center()
                dist_x = abs(center.x - point.x)
                dist_y = abs(center.y - point.y)
                if dist_x <= self.width / 2.0 and dist_y <= self.width / 2.0 \
                        and point.id not in cluster.ids:
                    count = cluster.count()
                    if count > max_count:
                        min_cluster = cluster
                        max_count = count
            if min_cluster is not None:
                min_cluster.add_point(point, False)
                ret = True
        return ret


