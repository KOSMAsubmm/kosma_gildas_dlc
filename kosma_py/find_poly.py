
from sicparse import OptionParser
import numpy
import linedb
import pyclass
from math import pi
import numpy as np



rad_to_arcsecond = 1/(pi/(180*60*60.0))

# determine if a point is inside a given polygon or not
# Polygon is a list of (x,y) pairs.

def point_inside_polygon(x,y,poly):
   n = len(poly)
   inside =False
   p1x,p1y = poly[0]
   for i in range(n+1):
     p2x,p2y = poly[i % n]
     if y > min(p1y,p2y):
         if y <= max(p1y,p2y):
             if x <= max(p1x,p2x):
                 if p1y != p2y:
                     xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                 if p1x == p2x or x <= xinters:
                     inside = not inside
     p1x,p1y = p2x,p2y
   return inside


def main():
   """
   find_poly/ find spectra based on polygon from lmv
              default finds just the extremes of the polygons
              -f options finds all spectra inside the polygon box  warning **SLOW

              -p save plot of polygon with range numbers stamped into file name

   """
   parser = OptionParser()
   parser.add_option("-f", "--find_full_poly", dest="find_full_poly", action="store_true", default=False)
   parser.add_option("-p", "--save_plot", dest="save_plot", action="store_true", default=False)
   parser.add_option("-o", "--output_path", dest="output_path", default="./")
   #    parser.add_option( "-e", "--energy", dest="energy", nargs=1, type="float", default=-1)
   #    parser.add_option( "-a", "--aij", dest="aij", nargs=1, type="float", default=-1)
   try:
      (options, args) = parser.parse_args()
   except:
      pyclass.message(pyclass.seve.e, "find_poly", "Invalid option")
      pyclass.sicerror()
      return
   #
   #
   if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
   # check is the poly array there
   try:
      x_coord = pyclass.gdict.poly.x*rad_to_arcsecond
      y_coord = pyclass.gdict.poly.y*rad_to_arcsecond
      poly_array = np.array(zip(x_coord,y_coord))
      area = pyclass.gdict.poly.area
      n_points = pyclass.gdict.poly.nxy
   except:
      pyclass.message(pyclass.seve.e, "find_poly", "Poly structure empty")
      pyclass.sicerror()
      return
   # set range based on polygon maximum
   print "set range {0} {1} {2} {3}".format(min(x_coord),max(x_coord),min(y_coord),max(y_coord))
   Sic.comm("set range {0} {1} {2} {3}".format(min(x_coord),max(x_coord),min(y_coord),max(y_coord)))
   # find all data available, check if a file is opened
   #
   if options.save_plot:
        filename = "{4}/poly_find_map_coord_{0:3.0f}_{1:3.0f}_{2:3.0f}_{3:3.0f}.png".format(min(x_coord),max(x_coord),min(y_coord),max(y_coord),options.output_path)
        pyclass.message(pyclass.seve.e, "find_poly", "saving {0}".format(filename))
        Sic.comm("ha {0} /device png /overwrite".format(filename))
   #
   if pyclass.gdict.found==0:
      pyclass.message(pyclass.seve.e, "find_poly", "nothing found")
      return
   #
   try:
      Sic.comm("find")
   except pgutils.PygildasError:
      pass
   if not options.find_full_poly:
        return
   x_off = pyclass.gdict.idx.loff*rad_to_arcsecond
   y_off = pyclass.gdict.idx.boff*rad_to_arcsecond
   numbers = pyclass.gdict.idx.num
   inside_count = 0
   for x,y,num in zip(x_off,y_off,numbers):
      inside = point_inside_polygon(x,y,poly_array)
      if inside:
         if inside_count==0:
            Sic.comm("find /number {0}".format(num))
         else:
            Sic.comm("find append /number {0}".format(num))
         inside_count+=1
   #



if __name__ == "__main__":
    main()
