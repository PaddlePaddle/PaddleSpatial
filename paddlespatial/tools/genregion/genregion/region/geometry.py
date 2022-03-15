# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
import math

from shapely import wkt
from shapely import geometry as spgeo
from shapely import ops as spops
from ordered_set import OrderedSet

from genregion.region import error

class Point(object):
    """ The point object.
    
    Attributes:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
    
    """
    __slots__ = ("x", "y")
    # The precision of the coordinate.
    # If we are using projection coordinates like BD09MC or EPSG3587, we suggest Point.precision = 2.
    # If we are using the geodetic system like WGS84 and BD09, we suggest Point.precision = 5.
    precision = 2
    base = math.pow(10, precision)
    
    @classmethod
    def set_precision(cls, precision):
        """Set the precision of the point.

        Args:
            precision (int): The level of precision.
        """
        cls.precision = int(precision)
        cls.base = math.pow(10, cls.precision)

    def __init__(self, x, y):
        """Construct a point object from either the number or the string.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
        """
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, pt):
        """Check whether both coordiates of two points are the same.

        Args:
            pt (Point): The other point.

        Returns:
            bool: True if they are the same.
        """
        return pt is not None and self.x == pt.x and self.y == pt.y

    def __str__(self):
        """Convert the point to a string format.

        Raises:
            error.RegionError: Empty object.

        Returns:
            str: The string format of the point.
        """
        if self.is_empty():
            raise error.RegionError("Cannot transfer an empty object to a string.")
        format = "%%.%df %%.%df" % (Point.precision, Point.precision)
        return format % (self.x, self.y)

    def trunc(self):
        """Truncate the point by precision.

        Raises:
            error.RegionError: Empty object.

        Returns:
            Point: Return the object itself.
        """
        if self.is_empty():
            raise error.RegionError("Cannot truncate an empty object.")
        self.x = int(self.x * Point.base + 0.5) * 1.0 / Point.base
        self.y = int(self.y * Point.base + 0.5) * 1.0 / Point.base
        return self

    def askey(self):
        """Convert the point object to a tuple of float as the dictionary key.

        Raises:
            error.RegionError: Empty object.

        Returns:
            tuple: A tuple of float represents the key.
        """
        if self.is_empty():
            raise error.RegionError("Empty object has no key.")
        return (self.x, self.y)

    def point(self):
        """Convert the point to the shapely point.

        Sometimes we need to use some functions in the shapely package such as interpolate() or project().

        Raises:
            error.RegionError: Empty object.

        Returns:
            spego.Point: The shapely format point.
        """
        if self.is_empty():
            raise error.RegionError("Cannot convert an empty object.")
        return spgeo.Point(self.x, self.y)

    def grid(self, grid_size=1024):
        """Find the grid which the point locates.

        To identify a grid, we use the integer coordinates for its position.

        Args:
            grid_size (int, optional): The width of the grid. Defaults to 1024.

        Raises:
            error.RegionError: Empty object.

        Returns:
            tuple: The horizontal and the vertical number which identify the order.
        """
        if self.is_empty():
            raise error.RegionError("Cannot find the grid for an empty object.")
        x_grid = int(self.x) // int(grid_size)
        y_grid = int(self.y) // int(grid_size)
        return (x_grid, y_grid)

    def destroy(self):
        """Destroy the point by setting its coordinates to None.
        """
        self.x = None
        self.y = None

    def is_empty(self):
        """Check whether the point object is empty.

        Returns:
            bool: True if not None.
        """
        return self.x is None


class Segment(object):
    """ The segment object, which is undirectional.

    Attributes:
        start (point): The start point.
        end (point): The end point.

    """
    __slots__ = ("start", "end")

    def __init__(self, start, end):
        """Initialize the segment using two points.

        Args:
            start (point): The start point.
            end (point): The end point.

        Raises:
            error.RegionError: Same point.
        """
        if start == end:
            raise error.RegionError("The start and the end point are the same.")
        self.start = start
        self.end = end

    def __eq__(self, seg):
        """Check whether two segments are the same.

        Args:
            seg (segment): The other segment.

        Returns:
            bool: True if the same.
        """
        if seg is not None and ((self.start == seg.start and self.end == seg.end) \
           or (self.end == seg.start and self.start == seg.end)):
            return True
        return False

    def __str__(self):
        """Convert a segment into a string.

        Raises:
            error.RegionError: Empty object.

        Returns:
            str: The string format like "x1 y1,x2 y2".
        """
        if self.is_empty():
            raise error.RegionError("Cannot convert an empty object.")
        start = str(self.start)
        end = str(self.end)
        if start < end:
            retstr = "%s,%s" % (start, end)
        else:
            retstr = "%s,%s" % (end, start)
        return retstr
    
    def askey(self):
        """Convert the segment object to a tuple of float as the dictionary key.

        Raises:
            error.RegionError: Empty object.

        Returns:
            tuple: A tuple of 4 float numbers represents the key.
        """
        if self.is_empty():
            raise error.RegionError("Empty object has no key.")
        if self.start.x < self.end.x \
                or (self.start.x == self.end.x and self.start.y < self.end.y):
            ret = (self.start.x, self.start.y, self.end.x, self.end.y)
        else:
            ret = (self.end.x, self.end.y, self.start.x, self.start.y)
        return ret

    def destroy(self):
        """Destroy the segment object by changing the start and end to None.
        """
        self.start = self.end = None

    def is_empty(self):
        """Check whether the segment is empty.

        Returns:
            bool: True if empty.
        """
        return self.start is None

    def length(self):
        """Calculate the length of the segment.

        Raises:
            error.RegionError: Empty object.

        Returns:
            float: The length of the segment.
        """
        if self.is_empty():
            raise error.RegionError("Empty object has no length.")
        return pt_2_pt_dist(self.start, self.end)

    def linestring(self):
        """Convert the segment into the directional shapely linestring.

        Raises:
            error.RegionError: Empty object.

        Returns:
            spgeo.LineString: The directional linestring.
        """
        if self.is_empty():
            raise error.RegionError("Cannot convert an empty segment.")
        points = []
        points.append((self.start.x, self.start.y))
        points.append((self.end.x, self.end.y))
        return spgeo.LineString(points)

    def __grids(self, x, y, grid_size, grid_set):
        """Return the grid that the given point locates.

        Note that there might be more than one grid being updated.
        If the point locates inside a grid, then we return one grid.
        If the point locates on the side of one grid, we return two grids.
        If the point locates on the vertex of one gird, we return four grids.

        Args:
            x (float): The x-coordinate of a point.
            y (float): The y-coordinate of a point.
            grid_size (int): The size of the grid.
            grid_set (set): The given set of the segments.
        """
        grid_x = int(x) // grid_size # changed
        grid_y = int(y) // grid_size # changed
        gs = [(grid_x, grid_y)]
        if grid_x * grid_size == x:
            gs.append((grid_x - 1, grid_y))
        if grid_y * grid_size == y:
            for i in range(len(gs)):
                gs.append((gs[i][0], grid_y - 1))
        for gd in gs:
            grid_set.add(gd)

    def grids(self, grid_size=1024):
        """Return all grids that a segment travels through.

        Args:
            grid_size (int, optional): The size of the grid. Defaults to 1024.

        Raises:
            error.RegionError: Empty object.

        Returns:
            list: A list of grid ids.
        """
        if self.is_empty():
            raise error.RegionError("Empty object is not in any grid.")
        grid_set = OrderedSet([])
        self.__grids(self.start.x, self.start.y, grid_size, grid_set)
        self.__grids(self.end.x, self.end.y, grid_size, grid_set)
        min_gx = int(min(self.start.x, self.end.x)) // grid_size # changed
        max_gx = int(max(self.start.x, self.end.x)) // grid_size # changed
        min_gy = int(min(self.start.y, self.end.y)) // grid_size # changed
        max_gy = int(max(self.start.y, self.end.y)) // grid_size # changed
        for grid_x in range(int(min_gx) + 1, int(max_gx) + 1):
            x = grid_x * grid_size
            y = self.start.y + (self.end.y - self.start.y) * (x - self.start.x) \
                    / (self.end.x - self.start.x)
            self.__grids(x, y, grid_size, grid_set)
        for grid_y in range(int(min_gy) + 1, int(max_gy) + 1):
            y = grid_y * grid_size
            x = self.start.x + (self.end.x - self.start.x) * (y - self.start.y) \
                    / (self.end.y - self.start.y)
            self.__grids(x, y, grid_size, grid_set)
        return list(grid_set)

    def intersect(self, seg):
        """Calculate the intersection with another segment.

        Note that there might be more or less than one intersection points of two segments.
        If two segments do not cross with each other, there is no intersection.
        If two segments overlap with each other, we return the starting and the ending overlapping point.
        If two segments merely intersect with each other, we return the intersection point.


        Args:
            seg (segment): The other segment.

        Returns:
            list: A list of intersection points.
        """
        thres = 0.001 / Point.base   # if t < thres, then we consider t = 0
        s01_x = self.end.x - self.start.x
        s01_y = self.end.y - self.start.y
        s23_x = seg.end.x - seg.start.x
        s23_y = seg.end.y - seg.start.y
        s20_x = self.start.x - seg.start.x
        s20_y = self.start.y - seg.start.y
        d01_len = math.sqrt(s01_x * s01_x + s01_y * s01_y)
        d23_len = math.sqrt(s23_x * s23_x + s23_y * s23_y)
        denom = s01_x * s23_y - s23_x * s01_y
        if denom == 0:  # Parallel or collinear
            d20_len = math.sqrt(s20_x * s20_x + s20_y * s20_y)
            cos201 = 0
            if d20_len > thres:
                cos201 = (s01_x * s20_x + s01_y * s20_y) / (d01_len * d20_len)
            sin201_2 = 1.0 - cos201 * cos201
            if sin201_2 < 0:
                sin201_2 = 0
            epson = d20_len * math.sqrt(sin201_2)
            if epson > 0.01 / Point.base:  # Parallel but not overlapped
                return []
            d1 = - d20_len * cos201
            if s01_x * s23_x + s01_y * s23_y > 0:
                d2 = d1 + d23_len
            else:
                d2 = d1 - d23_len
            d3 = d01_len
            if d2 < -thres and d1 < -thres or d2 > d3 + thres and d1 > d3 + thres:
                return []
            points = [(0, Point(self.start.x, self.start.y)), \
                      (d1, Point(seg.start.x, seg.start.y)), \
                      (d2, Point(seg.end.x, seg.end.y)), \
                      (d3, Point(self.end.x, self.end.y))]
            points.sort(key=lambda x: x[0])
            if points[1][1] == points[2][1]:
                return [points[1][1]]
            else:
                return [points[1][1], points[2][1]]
        s_numer = s01_x * s20_y - s01_y * s20_x
        t_numer = s23_x * s20_y - s23_y * s20_x
        s = s_numer / denom
        t = t_numer / denom
        if -s * d23_len > thres or (s - 1.0) * d23_len > thres:
            return []
        if -t * d01_len > thres or (t - 1.0) * d01_len > thres:
            return []
        if t >= 0 and t <= 1:
            if s < 0:
                return [Point(seg.start.x, seg.start.y)]
            if s > 1:
                return [Point(seg.end.x, seg.end.y)]
            else:
                return [Point(self.start.x + t * s01_x, self.start.y + t * s01_y)]
        if s >= 0 and s <= 1:
            if t < 0:
                return [Point(self.start.x, self.start.y)] 
            if t > 1:
                return [Point(self.end.x, self.end.y)]
        return []


class Region(object):
    """The region object.

    Attributes:
        points (list): A list of points that indicates the outer loop of polygon.
        holes (list): A list of lists of points that indicates holes.

    """
    __slots__ = ("points", "holes")

    def __filter_points(self, points):
        """Filter out the point which is the same as the previous one.

        Args:
            points (list): A list of points.

        Raises:
            error.RegionError: Less than 3 points.

        Returns:
            list: A list of filtered points.
        """
        if len(points) < 3:
            raise error.RegionError("Less than 3 points.")
        pts = []
        lastpt = None
        for pt in points:
            if not pt == lastpt:
                pts.append(pt)
                lastpt = pt
        if pts[0] == pts[-1]:
            pts.pop()
        return pts

    def __init_points(self, points, holes=None):
        """Initialize the points for constructing the polygon.

        The order of the points matters here.
        The order of points in the outer loop must be counterclockwise.
        The order of points in inner loops must be clockwise.
        Otherwise, we need to rearrange them in the correct way.

        Args:
            points (list): A list of points.
            holes (list, optional): A list of point lists representing holes. Defaults to None.

        Raises:
            error.RegionError: Less than 3 points.
            error.RegionError: Less than 3 points.
        """
        points = self.__filter_points(points)  # Filter out repeat points.
        if len(points) < 3:
            raise error.RegionError("A polygon cannot have less than 3 points.")
        if not is_counter_clockwise(points):    # The outer loop must be counterclockwise.
            self.points = points[::-1]
        else:
            self.points = points
        newholes = []
        if holes is not None and len(holes) > 0:
            for hole in holes:
                if len(hole) < 3:
                    raise error.RegionError("A polygon cannot have less than 3 points.")
                hole = self.__filter_points(hole)
                if is_counter_clockwise(hole):   # Inner loops must be clockwise.
                    newholes.append(hole[::-1])
                else:
                    newholes.append(hole)
        self.holes = newholes

    def __init_segments(self, segments):
        """Converting a list of segments into a list of points to construct the polygon.

        Note that we need to first convert those segments to a list of points and run the previous function.
        However, before doing that, we have to check a few conditions:
            1. The number of segments must be larger than 3.
            2. The starting point of each segment must be the same as the ending point of the previous one.
            3. The last segment and the first segment must be connected.
        After that, we will convert those segments into the point list for further polygon construction.

        Args:
            segments (list): A list of segments.

        Raises:
            error.RegionError: Less than 3 segments.
            error.RegionError: Consequtive segments are not connected.
            error.RegionError: The starting and the ending segment are not connected.
        """
        if len(segments) < 3:
            raise error.RegionError("The number of segments is less than 3.")
        points = [segments[0].start, segments[0].end]
        for seg in segments[1:]:
            if not seg.start == points[-1]:
                raise error.RegionError("Consequtive segments are not connected.")
            points.append(seg.end)
        if not points[0] == points[-1]:
            raise error.RegionError("The starting and the ending segment are not connected.")
        self.__init_points(points)
    
    def __init_polygon(self, polygon):
        """Convert a polygon to a list of points to construct the polygon.

        Args:
            polygon (spgeo.Polygon): The shapely format polygon.
        """
        points = []
        for pt in polygon.exterior.coords[:-1]:
            points.append(Point(pt[0], pt[1]))
        holes = []
        for inner in polygon.interiors:
            hole_points = []
            for pt in inner.coords[:-1]:
                hole_points.append(Point(pt[0], pt[1]))
            holes.append(hole_points)
        self.__init_points(points, holes)

    def __init_wkt(self, wktstr):
        """Convert a wkt string to a list of points to construct the polygon.

        Args:
            wktstr (str): The shapely wkt string format.

        Raises:
            error.RegionError: Not a shapely wkt string.
            error.RegionError: Loading failed.
        """
        try:
            geo = wkt.loads(wktstr)
            if geo.geom_type == "Polygon":
                self.__init_polygon(geo)
            else:
                raise error.RegionError("Not a polygon wkt string.")
        except Exception as e: 
            raise error.RegionError("Loading failed:" + e.message)

    def __init__(self, first, second=None):
        """Construct region points based on the given type of input.

        Args:
            first (list): Can be a list of points, segments, wkt strings or polygons 
            second (list, optional): Used for holes. Defaults to None.

        Raises:
            error.RegionError: Fail to initialize the region.
            error.RegionError: Fail to initialize the region.
        """
        if isinstance(first, list):
            if len(first) > 0:
                if isinstance(first[0], Point):
                    self.__init_points(first, second)
                elif isinstance(first[0], Segment):
                    self.__init_segments(first)
                else:
                    raise error.RegionError("Cannot initialize the region.")
        elif isinstance(first, spgeo.Polygon):
            self.__init_polygon(first)
        elif isinstance(first, str):
            self.__init_wkt(first)
        else:
            raise error.RegionError("Cannot initialize the region.")

    def __str__(self):
        """Convert the region to a wkt format string.

        Raises:
            error.RegionError: Empty object.

        Returns:
            str: The wkt format string.
        """
        if self.is_empty():
            raise error.RegionError("Cannot convert an empty object to a string.")
        ss = []
        for pt in self.points:
            ss.append(str(pt))
        ss.append(ss[0])
        wktstr = "POLYGON((" + ",".join(ss) + ")"
        if self.holes is not None:
            for hole in self.holes:
                ss = []
                for pt in hole:
                    ss.append(str(pt))
                ss.append(ss[0])
                wktstr += ",(" + ",".join(ss) + ")"
        wktstr += ")"
        return wktstr

    def assign(self, region):
        """Assign a new value to the region based on another region.

        Args:
            region (region): The target region.
        """
        self.points = region.points
        self.holes = region.holes
    
    def __ring_2_segments(self, points):
        """Generate a list of segments based on a list of points.

        Args:
            points (list): A list of points.

        Returns:
            list: A list of segments.
        """
        ret = []
        if points is None or len(points) < 2:
            return ret
        lastpt = points[-1]
        for pt in points:
            ret.append(Segment(lastpt, pt))
            lastpt = pt
        return ret

    def segments(self):
        """Convert the region into a list of segments.

        Raises:
            error.RegionError: Empty object.

        Returns:
            list: A list of segments.
        """
        if self.is_empty():
            raise error.RegionError("Cannot generate segments from an empty object.")
        ret = self.__ring_2_segments(self.points)
        for hole in self.holes:
            segs = self.__ring_2_segments(hole)
            ret.extend(segs)
        return ret

    def destroy(self):
        """Destroy a region by making everything empty.
        """
        self.points = self.holes = None

    def is_empty(self):
        """Check whether a region is empty.

        Returns:
            bool: True if empty.
        """
        return self.points is None

    def center(self):
        """ Return the center of the polygon.

        Note that this center might not be the centroid of the polygon.
        We basically use the representative point of the polygon.
        It is a cheaply computed point that is guaranteed to be within the geometric object.
        This function is mainly used to represent a polygon.

        Raises:
            error.RegionError: Empty object.

        Returns:
            point: The center of the object.
        """
        if self.is_empty():
            raise error.RegionError("Empty object does not have the center.")
        pt = self.polygon().representative_point()
        return Point(pt.x, pt.y)

    def mbr(self):
        """Calculate the MBR of the region.

        Raises:
            error.RegionError: Empty object.

        Returns:
            tuple: (minx, miny), (maxx, maxy)
        """
        if self.is_empty():
            raise error.RegionError("Cannot calculate the MBR for an empty region.")
        maxx = minx = self.points[0].x
        maxy = miny = self.points[0].y
        for pt in self.points[1:]:
            if pt.x < minx:
                minx = pt.x
            elif pt.x > maxx:
                maxx = pt.x
            if pt.y < miny:
                miny = pt.y
            elif pt.y > maxy:
                maxy = pt.y
        return (Point(minx, miny), Point(maxx, maxy))

    def grids(self, grid_size=1024):
        """Locate grids where the polygon lies in.

        Args:
            grid_size (int, optional): The size of the grid. Defaults to 1024.

        Raises:
            error.RegionError: Empty object.

        Returns:
            list: A list of grid ids (grid_x, grid_y).
        """
        if self.is_empty():
            raise error.RegionError("Cannot find the grid for an empty object.")
        bounds = self.mbr()
        start_grid = bounds[0].grid(grid_size)
        end_grid = bounds[1].grid(grid_size)
        ret = []
        for x_grid in range(start_grid[0], int(end_grid[0])):
            for y_grid in range(start_grid[1], int(end_grid[1])):
                ret.append((x_grid, y_grid))
        return ret

    def polygon(self):
        """Convert the region into shapely polygon.

        Raises:
            error.RegionError: Empty object.

        Returns:
            spgeo.Polygon: The shapely polygon.
        """
        if self.is_empty():
            raise error.RegionError("Cannot convert an empty region.")
        points = []
        for pt in self.points:
            points.append((pt.x, pt.y))
        holes = []
        if self.holes is not None:
            for hole in self.holes:
                hole_points = []
                for pt in hole:
                    hole_points.append((pt.x, pt.y))
                holes.append(hole_points)
        return spgeo.Polygon(points, holes)
        
    def __operate(self, op, region):
        """Run an operation on two polygons.

        Args:
            op (str): Can only be 'intersect', 'subtract' or union.
            region (region): The other region.

        Raises:
            error.RegionError: Empty object.
            error.RegionError: Unsupported operation type.

        Returns:
            list: A list of result regions. It only contains Polygon, Multipolygon or GeometryCollection.
        """
        if self.is_empty():
            raise error.RegionError("Cannot run a operation on an empty object.")
        ret = []
        polygon1 = self.polygon()
        polygon2 = region.polygon()
        if op == "intersect":
            inter = polygon1.intersection(polygon2)
        elif op == "subtract":
            inter = polygon1.difference(polygon2)
        elif op == "union":
            inter = polygon1.union(polygon2)
        else:
            raise error.RegionError("Unsupported operation type.")
        if inter.geom_type == "Polygon":
            ret.append(Region(inter))
        elif inter.geom_type == "MultiPolygon" or inter.geom_type == "GeometryCollection":
            for poly in inter:
                if poly.geom_type == "Polygon":
                    ret.append(Region(poly))
        return ret

    def intersect(self, region):
        """The intersection of two regions.

        Args:
            region (region): Another region.

        Returns:
            list: A list of regions.
        """
        return self.__operate("intersect", region)

    def subtract(self, region):
        """The difference of two regions.

        Args:
            region (region): Another region.

        Returns:
            list: A list of regions.
        """
        return self.__operate("subtract", region)

    def union(self, region):
        """The union of two regions.

        Args:
            region (region): Another region.

        Returns:
            list: A list of regions.
        """
        return self.__operate("union", region)

    def area(self):
        """Calculate the area of this region.

        Raises:
            error.RegionError: Empty object.

        Returns:
            float: The area value.
        """
        if self.is_empty():
            raise error.RegionError("Empty object has no area.")
        ret = []
        polygon = self.polygon()
        return polygon.area

    def length(self):
        """Calculate the perimeter of the region.

        Raises:
            error.RegionError: Empty object.

        Returns:
            float: The perimeter value.
        """
        if self.is_empty():
            raise error.RegionError("Empty object has no perimeter.")
        polygon = self.polygon()
        return polygon.length

    def gridize(self, grid_size=1024):
        """Cut a region into several pieces according to the grid.

        Args:
            grid_size (int, optional): Defaults to 1024.

        Raises:
            error.RegionError: Empty object.

        Returns:
            list: A list of result polygons.
        """
        if self.is_empty():
            raise error.RegionError("Empty object cannot be cut.")
        points = []
        bounds = self.mbr()
        start_grid = bounds[0].grid(grid_size)
        end_grid = bounds[1].grid(grid_size)
        minx = (start_grid[0] - 1) * int(grid_size)
        maxx = (end_grid[0] + 2) * int(grid_size)
        miny = (start_grid[1] - 1) * int(grid_size) 
        maxy = (end_grid[1] + 2) * int(grid_size)
        use_min = True
        for x_grid in range(int(start_grid[0]) + 1, int(end_grid[0]) + 1):
            x = x_grid * int(grid_size)
            if use_min:
                points.append((x, miny))
                points.append((x, maxy))
            else:
                points.append((x, maxy))
                points.append((x, miny))
            use_min = not use_min
        if use_min:
            y_list = range(start_grid[1] + 1, end_grid[1] + 1)
        else:
            y_list = range(end_grid[1], start_grid[1], -1)
        use_max = True
        for y_grid in y_list:
            y = y_grid * int(grid_size)
            if use_max:
                points.append((maxx, y))
                points.append((minx, y))
            else:
                points.append((minx, y))
                points.append((maxx, y))
            use_max = not use_max
        ret = []
        if len(points) > 0:
            line = spgeo.LineString(points)
            polygons = spops.split(self.polygon(), line)
            for poly in polygons:
                if poly.geom_type == "Polygon":
                    ret.append(Region(poly))
        else:
            ret.append(self)
        return ret

    def contains(self, point):
        """Check whether a point is in this region.

        Args:
            point (point): The point object.

        Raises:
            error.RegionError: Empty object.

        Returns:
            bool: True if the point is in this polygon.
        """
        if self.is_empty():
            raise error.RegionError("Empty object cannot have a point.")
        return self.polygon().covers(point.point())


def is_counter_clockwise(points):
    """Check whether a list of points are counterclockwise.

    Args:
        points (list): A list of points.

    Returns:
        bool: True if the arrangement of those points is counterclockwise.
    """
    pts = []
    for pt in points:
        pts.append((pt.x, pt.y))
    ring = spgeo.LinearRing(pts)
    return ring.is_ccw


def pt_2_pt_dist(pt1, pt2):
    """Calculate the distance between two points.

    Args:
        pt1 (point): The point object.
        pt2 (point): The point object.

    Returns:
        float: The distance of those two points.
    """
    return pt1.point().distance(pt2.point())


def pt_2_seg_dist(pt, seg):
    """Calculate the distance between a point and a segment.

    Args:
        pt (point): The point object.
        seg (segment): The segment object.

    Returns:
        float: The distance.
    """
    return pt.point().distance(seg.linestring())


def pt_2_reg_dist(pt, reg):
    """Calculate the distance of a point to a region.

    If the point is on the edge of the region, we return 0.
    If the point locates inside the region, we return a negative distance value.
    If the point locates outside the region, we return a positive distance value.

    Args:
        pt (point): The point object.
        reg (region): The region object.

    Returns:
        float: Distance that can be negative, positive or 0.
    """
    polygon = reg.polygon()
    point = pt.point()
    is_cover = polygon.covers(point)
    ring = spgeo.LinearRing(polygon.exterior.coords)
    min_dist = ring.distance(point)
    for hole in polygon.interiors:
        ring = spgeo.LinearRing(hole.coords)
        dist = ring.distance(point)
        if dist < min_dist:
            min_dist = dist
    if is_cover:
        return -min_dist
    else:
        return min_dist


def line_simp(points, threshold):
    """Perform Douglas simplification on a list of points.

    Args:
        points (list): A list of point objects that need to be simplified.
        threshold (float): Tolerance of the algorithm. 

    Returns:
        list: A list of simplified points.
    """
    pts = []
    for pt in points:
        pts.append((pt.x, pt.y))
    line = spgeo.LineString(pts)
    newline = line.simplify(threshold, preserve_topology=False)
    newpoints = []
    for npt in newline.coords[:]:
        newpoints.append(Point(npt[0], npt[1]))
    return newpoints


def dump_segments(segments, seg_file):
    """Save all segments to an local file.

    Args:
        segments (list): A list of segments.
        seg_file (str): The file path.
    """
    with open(seg_file, "w") as fo:
        for seg in segments:
            fo.write("%s\n" % (str(seg)))


def gen_segments(seg_file):
    """Segment generator.

    Args:
        seg_file (str): The file path to read those segments.

    Yields:
        Segment: Segment objects.
    """
    with open(seg_file, "rb") as fi:
        line = fi.readline()
        while line.decode() != "":
            p1, p2 = line.decode().strip().split(",")
            x1, y1 = p1.split(" ")
            x2, y2 = p2.split(" ")
            yield Segment(Point(float(x1), float(y1)), Point(float(x2), float(y2)))
            line = fi.readline()


def load_segments(seg_file):
    """Load a list of segments from a file.

    Args:
        seg_file (str): The file path to load those segments.

    Returns:
        list: A list of segments.
    """
    segs = []
    for seg in gen_segments(seg_file):
        segs.append(seg)
    return segs


def dump_regions(regions, reg_file):
    """Save all regions to a local file.

    Args:
        regions (list): A list of regions.
        reg_file (str): The file path to save all regions.
    """
    with open(reg_file, "w") as fo:
        for reg in regions:
            fo.write("%s\n" % (str(reg)))


def gen_regions(reg_file):
    """Region generator.

    Args:
        reg_file (str): The file path that stores those regions.

    Yields:
        Region: Region object.
    """
    with open(reg_file, "rb") as fi:
        line = fi.readline()
        while line != "":
            yield Region(line.strip())
            line = fi.readline()


def load_regions(reg_file):
    """Load all regions from a file.

    Args:
        reg_file (str): The file path.

    Returns:
        list: A list of regions.
    """
    regions = []
    for reg in gen_regions(reg_file):
        regions.append(reg)
    return regions

