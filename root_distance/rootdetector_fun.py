import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse
import math

from root_distance.general_functions import *
from root_distance.ml_functions import *

def shrink_to_root(img, perc = 0.2):

    xshape = img.shape[0]
    yindexsample = random.sample(range(img.shape[0]),int(xshape*.20))

    rootxpos = []
    for i in yindexsample:
        pos = np.array(np.where(img[i] == 1)).tolist()
        if len(pos[0])>1:
            rootxpos.append(pos)

    
    posinx = list(itertools.chain.from_iterable(rootxpos))

    avxposition = np.nanmean(np.array(posinx[0]))

    minclip = int(avxposition - int(xshape*perc))
    maxclip = int(avxposition + int(xshape*perc))

    return minclip, maxclip


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

def draw_lines_and_circles(image, rootimage, pillars_coords, radious, 
                            rootlines, pillarslines,
                            pillars_color = (0, 153, 153), root_lines_color = (255, 102, 0)):
    
    rootimagecliped = rootimage
    heatmap = cv2.applyColorMap((rootimagecliped*255).astype(np.uint8), cv2.COLORMAP_PLASMA)
    output = cv2.addWeighted((image).astype(np.uint8), 0.5, heatmap.astype(np.uint8), 1 - 0.75, 0)

    image = draw_circles(output, pillars_coords, radious, color_circle=pillars_color,  label = False)

    middlex = int(rootimagecliped.shape[1]/2)
    image = draw_lines(image, rootlines, color_line= root_lines_color, numberxpos = middlex)
    image = draw_lines(image, pillarslines, color_line=pillars_color, numberxpos = middlex)

    return image

def export_images(list_images, folder = None, filenames = None, preffix = '.jpg'):

    if not os.path.exists(folder):
        os.mkdir(folder)
    if filenames is None:
        filenames = ['image_{}{}'.format(i, preffix) for i in range(len(list_images))]

    for i,img in enumerate(list_images):
        cv2.imwrite(os.path.join(folder, filenames[i]), img)


class RootonPillars():

    @property
    def image(self):
        imageinfo = read_image(self.img_path)
        
        if len(np.array(imageinfo).shape) == 3:
            imageinfo = np.expand_dims(np.array(imageinfo), axis= 0)

        
        return np.array(imageinfo)
    
    @property
    def root_image(self):
        return self._root_image


    @property
    def pillars_coords(self):

        coordslist = []
        
        for j in range(len(self._filteredpillars_coords)):
            circle_coordsaslist = [] 
            for i in self._filteredpillars_coords[j]:
                circle_coordsaslist.append(self._filteredpillars_coords[j][i][0])
                circle_coordsaslist.append(self._filteredpillars_coords[j][i][1])

            coordslist.append(circle_coordsaslist)

        return coordslist

    @property
    def root_intersectionlines(self):
        lineslist = []
        for i in range(len(self._filteredpillars_coords)):
            lineslist.append(lines_through_root_middle(
                                         self._filteredpillars_coords[i], 
                                         self.root_image[i][:,self._minposx[i]:self._maxposx[i]]))
        return lineslist

    @property
    def pillars_intersectionlines(self):
        lineslist = []
        for i in range(len(self._filteredpillars_coords)):
            lineslist.append(get_pillars_lines(self._filteredpillars_coords[i]))

        return lineslist
    
    @property
    def pillars_lines_as_table(self):
        dflist = []
        for i in range(len(self.pillars_intersectionlines)):
            df = distances_table(self.pillars_intersectionlines[i])
            df['object'] = 'pillars'
            df['image_name'] = self.image_names[i]
            dflist.append(df)

        return pd.concat(dflist).reset_index()
    
    @property
    def root_lines_as_table(self):
        dflist = []
        for i in range(len(self.root_intersectionlines)):
            df = distances_table(self.root_intersectionlines[i])
            df['object'] = 'root'
            df['image_name'] = self.image_names[i]
            dflist.append(df)

        return pd.concat(dflist).reset_index()

    def export_final_images(self, path):
        
        export_images(self._get_final_images(), 
                      path, 
                      self.image_names)



    def _get_pillarsrawcoords(self):

        pillars_coords = {}
        radious = []
        for i in range(self.image.shape[0]):
            
            pillars = PillarImage(self.image[i,:,self._minposx[i]:self._maxposx[i],:],
                              minradius = self.minradius, 
                              maxradius = self.maxradius, 
                              max_circles= self.max_circles)
            
            pillars.find_circles()
            pillars.sort_circles()

            pillars_coords[i] = pillars.circle_coords
            radious.append(np.nanmean(np.array(pillars_coords[i]).T[2]))
            
        self._raw_pillars_coords = pillars_coords
        self.radious = radious

    def _filtered_coords(self):
        
        pillars_coords = {}
        
        for i in range(len(self._raw_pillars_coords.keys())):
            coords_filtered, warningmessage = circles_thatedge_root(self._raw_pillars_coords[i],
                                                                self.root_image[i,:,self._minposx[i]:self._maxposx[i]], 
                                                                self.radious[i])
            pillars_coords[i] = coords_filtered

        #self._warning_message  = warningmessage
        self._filteredpillars_coords = pillars_coords

    def lines_table_as_csv(self, filename):

        pd.concat([self.root_lines_as_table,self.pillars_lines_as_table]).to_csv(filename)

    def plot_root_overlapped(self, maximages = None, figsize = (8,12)):

        if maximages is None:
            maximages = self.root_image.shape[0]
        
        fig, ax = plt.subplots(ncols=1, nrows= maximages, figsize =figsize)
        
        for i in range(self.root_image.shape[0]):

            heatmap = cv2.applyColorMap((self.root_image[i]*255).astype(np.uint8), cv2.COLORMAP_PLASMA)
            output = cv2.addWeighted((self.image[i]).astype(np.uint8), 0.5, heatmap.astype(np.uint8), 1 - 0.75, 0)
            if maximages>1:
                ax[i].imshow(output)
            else:
                ax.imshow(output)
        

    def _get_final_images(self, pillars_color = (0, 153, 153), root_lines_color = (255, 102, 0)):
        imagestoplot = []

        for i in range(self.image.shape[0]):
            imagestoplot.append(draw_lines_and_circles(self.image[i][:,self._minposx[i]:self._maxposx[i],:].copy(),
                                   self.root_image[i][:,self._minposx[i]:self._maxposx[i]].copy(),
                                   self.pillars_coords[i],
                                   self.radious[i],
                                   self.root_intersectionlines[i],
                                   self.pillars_intersectionlines[i],
                                   pillars_color = pillars_color, root_lines_color = root_lines_color))
        
        return imagestoplot

    def plot_final_layer(self, ncols = 1, nrows = 1, maximages = None, pillars_color = (0, 153, 153), root_lines_color = (255, 102, 0), figsize = (8,8)):
        
        imagestoplot = self._get_final_images(pillars_color = pillars_color, root_lines_color = root_lines_color)
        if maximages is None:
            maximages = self.root_image.shape[0]
            nrows= maximages
        
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize =figsize, dpi = 80)
        for i in range(self.image.shape[0]):
            if i <= maximages:
                if maximages>1:
                    ax[i].imshow(imagestoplot[i])

                else:
                    ax.imshow(imagestoplot[i])

        
    
    def _dic_root_xlocation(self):
        minposx = [0] * len(self._root_image)
        maxposx = [0] * len(self._root_image)
        for i in range(len(self._root_image)):
            minposx[i], maxposx[i] = shrink_to_root(self.root_image[i],perc=0.17)
        
        self._minposx = minposx
        self._maxposx = maxposx


    def __init__(self, image_path, weigths_path=None, architecture="vgg16",
                 minradius = 17, maxradius = 18, max_circles= 18):
        
        self.image_names = None
        self.img_path = image_path
        self.minradius = minradius
        self.maxradius = maxradius
        self.max_circles = max_circles


        self.image_names =get_filenames(self.img_path)
        if type(self.image_names) is str:
            self.image_names = [self.image_names]

        detector = root_detector(weigths_path, architecture = architecture)

        if type(self.image) is list:
            self._root_image = detector.detect_root(np.concatenate(self.image, axis=0))
        else:
            self._root_image = detector.detect_root(self.image)

        
        self._dic_root_xlocation()

        #pillars = PillarImage(self.image[:,self._minposx:self._maxposx,:],minradius = minradius, maxradius = maxradius, max_circles= max_circles)
        self._get_pillarsrawcoords()
        self._filtered_coords()
