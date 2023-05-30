
import random
import cv2
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

def check_dims(image, referenceshape):
    originalshape = image.shape[:2]
    if originalshape[0] != referenceshape[0] and originalshape[1] != referenceshape[1]:
        img_resized = cv2.resize(image.copy(), referenceshape, interpolation = cv2.INTER_AREA)
    else:
        img_resized = image.copy()

    return img_resized, originalshape


def get_seppillarsdistances(img):
    distances = []
    points = []
    for p1,p2 in img.root_circlecoords:
        x1,y1 = p1
        x2,y2 = p2

        pminy = y1 if y1 < y2 else y2
        p1l = (x1 - img.radious, pminy)
        p2l = (x2 + img.radious, pminy)
        points.append((p1l,p2l))


def find_distancesperrow(coords, radious, threshhold = 1.5):
    yminpos = np.argmin(np.array(coords).T[1])

    ypos = np.array(coords).T[1][yminpos]
    xpos = np.array(coords).T[0][yminpos]

    cond1 = np.array(coords).T[1] > ypos-(radious*threshhold)
    cond2 = np.array(coords).T[1]< ypos+(radious*threshhold)
    postrue = cond1 * cond2
    rowy = np.array(coords).T[1][postrue]
    rowx = np.array(coords).T[0][postrue]

    xxmin = np.max(rowx)
    corrdsorted = []
    for i in np.argsort(rowx):
        xx,yy = rowx[i],rowy[i]
        corrdsorted.append((xx,yy))
    eucs = []
    for i in range(1,len(corrdsorted)):
        euc = euc_distance((corrdsorted[i-1][0],corrdsorted[i-1][1]),(corrdsorted[i][0],corrdsorted[i][1]))
        eucs.append(euc)

    coords = [coords[i] for i in range(len(coords)) if i not in np.argwhere(postrue).T[0]]
    return eucs , coords, corrdsorted

def get_x_y_data(coordinates):

    xpos = [float(coordinates[i][0]) for i in range(len(coordinates))]
    ypos = [float(coordinates[i][1]) for i in range(len(coordinates))]

    return xpos,ypos 

def minmax_scale(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def filter_bytreshhold(coords, central_point,trh ):
    line_listidx = []
    for i in range(len(coords)):
        xval = float(coords[i][0]) - float(central_point[0])

        if xval < trh:
            line_listidx.append(i)
    return line_listidx


def euc_distance(p1,p2):
    
    dist = math.sqrt(
        math.pow((p1[0]-p2[0]),2) + math.pow((p1[1]-p2[1]),2))

    return dist


def organized_circles(circles, radious):
    lencircles = len(circles)
    corner,ind = find_corner(circles)
    organized_data = [corner]
    
    
    changes = 0
    maxncirclespercol = 0
    nrows = []
    count = 0
    while len(circles)>2:
        
        disttocorner = []

        for circ in circles:
            disttocorner.append(euc_distance((corner[0],corner[1]),(circ[0],circ[1])))

        alldist = np.argsort(disttocorner).copy()
        if len(circles) <= (maxncirclespercol) and abs(circles[alldist[1]][0] - corner[0])  < radious:
            indnext = alldist[1]
            
        else:
            x1,x2 = abs(circles[alldist[1]][0] - corner[0]), abs(circles[alldist[2]][0] - corner[0])
            indnext = alldist[1] if x1<x2 else alldist[2]

            st = 1
            while abs(circles[indnext][0] - corner[0])  > radious and st<(len(alldist)-2):
                indnext = alldist[st+1]
                alldist = alldist[st:-1]
                st +=1 
                
        if abs(circles[indnext][0] - corner[0])  > radious:
            circles = np.delete(circles, ind, axis = 0)
            corner,ind = find_corner(circles)
            changes+=1
            tmp = len(organized_data)//changes
            maxncirclespercol = tmp if tmp>maxncirclespercol else maxncirclespercol
            nrows.append(count+1)
            count = -1

        else:
            corner =  circles[indnext]
            circles = np.delete(circles, ind, axis = 0)
            for i in range(len(circles)):
                if '{}-{}'.format(corner[0],corner[1])=='{}-{}'.format(circles[i][0],circles[i][1]):
                    ind = i
        count +=1
        organized_data.append(corner)

    circles = np.delete(circles, ind, axis = 0)
    corner = circles[0]
    
    organized_data.append(corner)

    return organized_data,nrows


def find_corner(coords, indexes= None):
    if indexes is None:
        indexes = list(range(len(coords)))

    x_pos, y_pos = coords.T[0],coords.T[1]
    sumarrays = np.array(x_pos) + np.array(y_pos)
    return coords[[indexes[i] for i in np.argsort(sumarrays)][0]],[indexes[i] for i in np.argsort(sumarrays)][0]
    

def get_circles(img, max_circles = 32, minradius = 17, maxradius = 18, limit = 1000):
    
    ncircles = list(range(max_circles-int(max_circles*0.2),max_circles))

    param2_tunning = list(range(1,50,1))
    param1_tunning = list(range(1,50,1))

    count = 0
    num_circles= 0
    detected_circles = 0
    while num_circles not in ncircles and count < limit:
        param2 = random.choice(param2_tunning)
        param1 = random.choice(param1_tunning)
        detected_circles = cv2.HoughCircles(img,
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = param1,
                param2 = param2, minRadius = minradius, maxRadius = maxradius)

        if detected_circles is not None:
            num_circles = detected_circles.shape[1]
        if count%100 == 0 and minradius > minradius-2:
            minradius = minradius-1
            maxradius = maxradius+1

        count += 1
        
    
    if detected_circles is None:
        raise ValueError("None circle were found {} {}".format(minradius, maxradius))    
    return detected_circles, (minradius + maxradius)/2


def get_lines_coordinates(img):

    linescoords = []
    for p1,p2 in img.root_circlecoords:
        x1,y1 = p1
        x2,y2 = p2

        pminy = y1 if y1 < y2 else y2
        p1l = (x1 - img.radious, pminy)
        p2l = (x2 + img.radious, pminy)

        linescoords.append((p1l,p2l ))

    return linescoords

def draw_lines(img, linescoords, numberxpos, color_line = (0,255,0) ):
    count = 0
    for p1,p2 in linescoords:
        try:
            cv2.line(img,(int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), color_line,2)
            pminy = p2[1] if p2[1]<p1[1] else p1[1]
            cv2.putText(img, str(count+1), (int(numberxpos), int(pminy-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color_line, 1)

            count +=1
        except:
            pass

    return img

def draw_circles(img, circles_coords,radious, label = True, color_circle = (0, 255, 0)):
    circles_coords = np.uint16(np.around(circles_coords))
    for i, pt in enumerate(circles_coords):
        try:
            
            a, b, r = pt[0], pt[1], math.ceil(radious)

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, color_circle, 2)

            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            if label:
                cv2.putText(img, str(i+1), (a - int(r/2), b - int(r/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color_circle, 2)
        except:
            pass
            
    return img
    
def change_img_contrast(image, alpha = 3,beta = 50  ):

    new_image = np.zeros(image.shape, image.dtype)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
                
    return new_image

def get_filenames(imgpath, suffix = 'jpg'):

    if type(imgpath) is str:
        if os.path.exists(imgpath):
            fn = os.path.basename(imgpath)
        else: 
            raise ValueError("the {} file does not exist".format(imgpath))

    elif type(imgpath) is list:
        fn = [get_filenames(i) for i in imgpath]

    else:
        fn = None

    return fn


def read_image(imgpath):
    if type(imgpath) is str:
        if os.path.exists(imgpath):
            img = cv2.imread(imgpath)
        else: 
            raise ValueError("the file does not exist")
    elif type(imgpath) is list:
        img = [read_image(i) for i in imgpath]
    else:
        img = imgpath

    return img

class PillarImage:
    """

    Returns:
        _type_: _description_
    """
    @property
    def img_data(self):
        return read_image(self.img_path)

    @property
    def dict_imgs(self):
        return self._dict
    
    @property
    def all_pillars_distancesperrow(self):
        return self._distances_perrow
    
    @property
    def all_coords_sorted_perrow(self):
        return self._coords_sortedperrow
    @property
    def root_img(self):
        (ymin,ymax),(xmin,xmax) = self.crop_onlyroot()
        return self.img_data[ymin:ymax,xmin:xmax]

    def get_root_centerpillar_xdist(self):
        
        distancesfiltered = [self._distances_perrow[i] 
            for i in range(len(self._distances_perrow)) 
                if len(self._distances_perrow[i]) == self.ncols-1]
        npospillar = np.argmax(np.array(distancesfiltered).mean(axis = 0))

        coords_rowfilt= [self._coords_sortedperrow[i] 
            for i in range(len(self._coords_sortedperrow)) 
                if len(self._coords_sortedperrow[i]) == self.ncols]

        coordsfiltered = [(coords_rowfilt[i][npospillar],coords_rowfilt[i][npospillar+1]) for i in range(len(coords_rowfilt))]

        self.root_xcoord = np.array(coordsfiltered).T[0].mean()
        self.root_circlecoords = coordsfiltered


    def _organizing_distperrow(self):
        origdist = self.circle_coords.copy()
        eucsdist = []
        coords_sortedperrow = []
        while len(origdist)>1:
            
            euc,origdist , coordssorted= find_distancesperrow(origdist,self.radious)
            eucsdist.append(euc)
            coords_sortedperrow.append(coordssorted)

        self._distances_perrow = eucsdist
        self._coords_sortedperrow = coords_sortedperrow
        self.max_cols()
        self.get_root_centerpillar_xdist()

    def max_cols(self):
        maxcols = len(self._coords_sortedperrow[0])

        for i in range(len(self._coords_sortedperrow)):
            if len(self._coords_sortedperrow[i]) > maxcols:
                maxcols = len(self._coords_sortedperrow[i]) 

        self.ncols = maxcols

    def plot_circles(self, figsize = (16, 28)):

        image = self.img_data.copy()
        image = draw_circles(image, self.circle_coords,self.radious, label = True)

        plt.figure(figsize=figsize, dpi=80)
        plt.imshow(image)
        if self.root_xcoord is not None:
            plt.axvline(self.root_xcoord, color ='green', lw = 2, alpha = 0.75)
    
    def plot_only_root(self, figsize = (16, 28)):


        image = self._dict['gray'].copy()
        image = minmax_scale(image) *255
        image = draw_circles(image, self.circle_coords,self.radious, label = True)

        plt.figure(figsize=figsize, dpi=80)
        (ymin,ymax),(xmin,xmax) = self.crop_onlyroot()
        cv2.line(image,(int(self.root_xcoord),ymin), (int(self.root_xcoord),int(ymax)), (255,0,0),2)
        plt.imshow( image[ymin:ymax,xmin:xmax])
        #plt.axvline(self.root_xcoord, color ='white', lw = 2, alpha = 0.75)
    
    def plot_root_lines(self,figsize=(6,8)):

        imagegray = self.gray.copy()
        image = cv2.cvtColor(imagegray,cv2.COLOR_GRAY2RGB)
        image = change_img_contrast(image)
        imagegray = draw_circles(imagegray, self.circle_coords,self.radious, label = False)
        linecoords = get_lines_coordinates(self)
        image = draw_lines(image, linecoords, self.root_xcoord)
        imagegray = draw_lines(imagegray, linecoords, self.root_xcoord)

        fig, ax = plt.subplots(ncols = 2,nrows = 1, figsize=figsize, dpi=80)

        (ymin,ymax),(xmin,xmax) = self.crop_onlyroot()
        ax[0].imshow( image[ymin:ymax,xmin:xmax])
        ax[1].imshow( imagegray[ymin:ymax,xmin:xmax])
        
        return fig

    def crop_onlyroot(self):

        import math
        ymaxval = np.array(self.root_circlecoords).T[1].max() + int(self.radious+1)

        xminval = np.array(self.root_circlecoords).T[0].min() - int(self.radious+1)
        xmaxval = np.array(self.root_circlecoords).T[0].max() + int(self.radious+1)

        return [0,math.ceil(ymaxval)],[math.floor(xminval),math.ceil(xmaxval)]

    def get_distances_table(self):
        linescoords = get_lines_coordinates(self)
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

    def find_circles(self, **kwargs):


        circlesmax = []
        for i in self._dict.keys():
            circles,radious = get_circles(self._dict[i],
                                          minradius=  self._minradquery, 
                                          maxradius=self._maxradquery, 
                                          max_circles = self._max_circles, **kwargs)
            #print(len(circles[0]))
            if len(circlesmax)<len(circles[0]):
                circlesmax = circles[0]
        print(f"{len(circlesmax)} circles were found")
        self.circle_coords = circlesmax
        self.radious = radious
        self._organizing_distperrow()

    def sort_circles(self):
        self.circle_coords, self._ncolumnsperrow = organized_circles(self.circle_coords, self.radious)

    def __init__(self,
                 img_path = None,
                 gray = True,
                 blur = True,
                 bilateral = True,
                 otsu_blur = True,
                 minradius = 17, maxradius = 18,
                 max_circles = None):

        self._dict = {}
        self.img_path = img_path
        self._ncolumnsperrow = None
        self.root_xcoord = None
        self._minradquery = minradius
        self._maxradquery = maxradius
        self._max_circles = max_circles

        if gray:
            self.gray = cv2.cvtColor(self.img_data, cv2.COLOR_BGR2GRAY)
            self._dict['gray'] = self.gray
        if blur:
            self.blur = cv2.GaussianBlur(src=self.gray, ksize=(5,5), sigmaX=0, sigmaY=0)
            self._dict['blur'] = self.blur
        if bilateral:
            self.bilateral_filter = cv2.bilateralFilter(src=self.gray, d=3, sigmaColor=50, sigmaSpace=50)
            self._dict['bilateral_filter'] = self.bilateral_filter
        if otsu_blur:
            th, im_gray_th_otsu = cv2.threshold(self.blur, 150, 255, cv2.THRESH_OTSU)
            self.otsu_blur = im_gray_th_otsu
            self._dict['otsu_blur'] = self.otsu_blur
