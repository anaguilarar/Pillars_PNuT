import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse
import math

from root_distance.general_functions import *
from root_distance.ml_functions import *

def find_edgebordersinimage(img, yposition):

    linedata = img[(int(yposition)-1):(int(yposition)+1),:]
    #print(np.where(linedata==1))
    insideroot = np.where(linedata==1)[1]
    if len(insideroot)>1:

        xminborder = insideroot.min() 
        xmaxborder = insideroot.max()
    else:
        xminborder, xmaxborder = np.nan, np.nan

    return xminborder, xmaxborder

def find_circlesinframe(circlecoords, yframe, radious):

    minborder = yframe-radious
    maxborder = yframe+radious
    
    posindict = []
    for i in circlecoords:
        coords = circlecoords[i]
        if coords[1] > minborder and coords[1] < maxborder:
            #circlesinframe.append(coords)
            posindict.append(i)
        if len(posindict) > 3:
            pass
    
    return posindict

def musttouchroot_filter(circlescoords, posindict, root_image, radious):

    edgecircles = []
    posdictafterfilter = []
    postodelete = []

    for i in posindict:
        circlecoords = circlescoords[i]
        #linedata = root_image[(int(circlecoords[1])-1):(int(circlecoords[1])+1),:]
        minval, maxval = find_edgebordersinimage(root_image, circlecoords[1])

        if np.isnan(minval):
            postodelete.append(i)
        else:
            minval = minval - int((radious*2))
            maxval = maxval + int((radious*2))

            if circlecoords[0] > minval and circlecoords[0] < maxval:
                edgecircles.append(circlecoords)
                posdictafterfilter.append(i)
            else:
                postodelete.append(i)
        
    return edgecircles, posdictafterfilter,postodelete

def find_miny(coords, mincoordiny = 512):
    mincoordiny = 512
    posmin = -1
    for i in coords:
        if coords[i][1] < mincoordiny:
            mincoordiny = coords[i][1]
            posmin = i

    return mincoordiny, posmin

def circles_thatedge_root(coords, root_image, radious):
        
    dict_coords = {}
    for i, coord in enumerate(coords):
        dict_coords[str(i)] = coord

    finalcoords = {}
    count = 0
    thereiscoords = True
    anomaly = []
    while thereiscoords:
        
        ycoord, _ = find_miny(dict_coords)
        posindict = find_circlesinframe(dict_coords, ycoord, radious)
        circlecoords, posindict, wrongcircle = musttouchroot_filter(dict_coords, posindict ,root_image, radious = radious)


        for i in posindict + wrongcircle:
            dict_coords.pop(i)
        
        if len(posindict) ==  2:
            finalcoords[str(count)] =  circlecoords
        count += 1 
        if(count > 20):
            anomaly = "more iterations than the allowed"
            thereiscoords = False
        if len(dict_coords.keys()) < 1:
            thereiscoords = False

    return finalcoords, anomaly

def lines_through_root_middle(circle_coords, root_image):

    linescoords = []
    maxval = np.max([int(i) for i in circle_coords.keys()])+1
    for count in range(maxval):

        if str(count) in circle_coords.keys() and str(count+1) in circle_coords.keys():

            topcircles = circle_coords[str(count)] 
            bottomcircles = circle_coords[str(count+1)]
            ymiddlecircles = (np.mean(np.array(topcircles).T[1]) + np.mean(np.array(bottomcircles).T[1]))/2
            xminval, xmaxval = find_edgebordersinimage(root_image, int(ymiddlecircles))
            if not np.isnan(xminval):
                linescoords.append(((xminval, int(ymiddlecircles)), (xmaxval, int(ymiddlecircles))))
        
        elif str(count) in circle_coords.keys() and str(count+2) in circle_coords.keys():
            topy = circle_coords[str(count)][0][1]
            bottomy = circle_coords[str(count+2)][1][1]
            radious = circle_coords[str(count)][0][2]

            if abs(topy - bottomy)<radious*4:
                topcircles = circle_coords[str(count)] 
                bottomcircles = circle_coords[str(count+2)]
                ymiddlecircles = (np.mean(np.array(topcircles).T[1]) + np.mean(np.array(bottomcircles).T[1]))/2
                xminval, xmaxval = find_edgebordersinimage(root_image, int(ymiddlecircles))
                if not np.isnan(xminval):
                    linescoords.append(((xminval, int(ymiddlecircles)), (xmaxval, int(ymiddlecircles))))
    
    return linescoords

def lines_tocirclescenter(coords):
    
    x1,y1, radious1 = coords[0]
    x2,y2, radious2 = coords[1]
    
    pminy = y1 if y1 < y2 else y2
    p1l = (x1 - radious1, pminy)
    p2l = (x2 + radious2, pminy)

    
    return (p1l,p2l )


def get_pillars_lines(coords):

    pillarslines = []    
    for i in coords:

        pillarslines.append(lines_tocirclescenter(coords[i]))

    return pillarslines


def distances_table(linescoords):
    distancespx = []
    distances = []
    factorcorrection= []
    count = []
    changefactor = 0.4023
    for i, (p1,p2) in enumerate(linescoords):
        distancespx.append(euc_distance(p1,p2))
        d = euc_distance(p1,p2)/changefactor
        distances.append(d)
        factorcorrection.append((d- 260)/2)
        count.append(i+1)

    return pd.DataFrame({'line_index': count, 
                        'distances_pixels': distancespx, 
                        'distances_microns': distances, 'corrected_factor': factorcorrection})


class RootonPillars():

    @property
    def image(self):
        return read_image(self.img_path).copy()
    
    @property
    def root_image(self):
        return self._root_image


    @property
    def pillars_coords(self):

        circle_coordsaslist = [] 

        for i in self._filteredpillars_coords:
            circle_coordsaslist.append(self._filteredpillars_coords[i][0])
            circle_coordsaslist.append(self._filteredpillars_coords[i][1])

        return circle_coordsaslist

    @property
    def root_intersectionlines(self):
        return lines_through_root_middle(self._filteredpillars_coords, 
                                         self.root_image[:,self._minposx:self._maxposx])

    @property
    def pillars_intersectionlines(self):

        return get_pillars_lines(self._filteredpillars_coords)
    
    @property
    def pillars_lines_as_table(self):
        df = distances_table(self.pillars_intersectionlines)
        df['object'] = 'pillars'

        return df
    
    @property
    def root_lines_as_table(self):
        df = distances_table(self.root_intersectionlines)
        df['object'] = 'root'

        return df

    def _get_pillarsrawcoords(self):

        pillars = PillarImage(self.image[:,self._minposx:self._maxposx,:],
                              minradius = self.minradius, 
                              maxradius = self.maxradius, 
                              max_circles= self.max_circles)
        pillars.find_circles()
        pillars.sort_circles()

        self._raw_pillars_coords = pillars.circle_coords
        self.radious = np.nanmean(np.array(self._raw_pillars_coords).T[2])

    def _filtered_coords(self):

        
        coords_filtered, warningmessage = circles_thatedge_root(self._raw_pillars_coords,
                                                                self.root_image[:,self._minposx:self._maxposx], 
                                                                self.radious)
        self._warning_message  = warningmessage
        self._filteredpillars_coords = coords_filtered


    def plot_root_overlapped(self):

        heatmap = cv2.applyColorMap((self.root_image*255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        output = cv2.addWeighted((self.image).astype(np.uint8), 0.5, heatmap.astype(np.uint8), 1 - 0.75, 0)
        plt.imshow(output)
        
    def plot_final_layer(self, pillars_color = (0, 153, 153), root_lines_color = (255, 102, 0), figsize = (8,8)):
        image = self.image[:,self._minposx:self._maxposx,:].copy()
        rootimagecliped = self.root_image[:,self._minposx:self._maxposx].copy()
        heatmap = cv2.applyColorMap((rootimagecliped*255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        output = cv2.addWeighted((image).astype(np.uint8), 0.5, heatmap.astype(np.uint8), 1 - 0.75, 0)

        image = draw_circles(output, self.pillars_coords, self.radious, color_circle=pillars_color,  label = False)

        middlex = int(rootimagecliped.shape[1]/2)
        image = draw_lines(image, self.root_intersectionlines, color_line= root_lines_color, numberxpos = middlex)
        image = draw_lines(image, self.pillars_intersectionlines, color_line=pillars_color, numberxpos = middlex)

        fig, ax = plt.subplots(ncols = 1,nrows = 1, figsize=figsize, dpi=80)
        ax.imshow( image)
        
    def __init__(self, image_path, weigths_path=None, architecture="vgg16",
                 minradius = 17, maxradius = 18, max_circles= 18):
        
        self.img_path = image_path
        self.minradius = minradius
        self.maxradius = maxradius
        self.max_circles = max_circles


        detector = root_detector(weigths_path, architecture = architecture)
        self._root_image = detector.detect_root(self.image)

        self._minposx, self._maxposx = shrink_to_root(self.root_image,perc=0.17)

        #pillars = PillarImage(self.image[:,self._minposx:self._maxposx,:],minradius = minradius, maxradius = maxradius, max_circles= max_circles)
        self._get_pillarsrawcoords()
        self._filtered_coords()