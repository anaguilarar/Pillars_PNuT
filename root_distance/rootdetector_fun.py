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
        circlecoords, posindict, wrongcircle = musttouchroot_filter(
            dict_coords, posindict ,root_image, radious = radious)


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

def add_line_ontop(firstcircle_coords, root_image):
  ## get y coordinate
  ycircle = np.mean(np.array(firstcircle_coords).T[1])
  ## get circle radious
  firstcircle_rad = np.mean(np.array(firstcircle_coords).T[2])
  ## calculate a y coordinate intersection reference
  yintersect = ycircle-int((firstcircle_rad*1.5))
  
  ## check if the inteserction is inside the image
  yintersect = yintersect if yintersect > 0 else int(firstcircle_rad*0.5)

  # find borders
  xminval, xmaxval = find_edgebordersinimage(root_image, int(yintersect))
  # return line coordinates
  return ((xminval, int(yintersect)), (xmaxval, int(yintersect)))

def add_line_onbottom(lastcircle_coords, root_image):
  ## get y coordinate
  ycircle = np.mean(np.array(lastcircle_coords).T[1])
  ## get circle radious
  lastcircle_rad = np.mean(np.array(lastcircle_coords).T[2])
  ## calculate a y coordinate intersection reference
  yintersect = ycircle+int((lastcircle_rad*1.5))
  bottom = root_image.shape[0]
  ## check if the inteserction is inside the image
  yintersect = yintersect if yintersect < bottom else bottom-int(lastcircle_rad*0.5)

  # find borders
  xminval, xmaxval = find_edgebordersinimage(root_image, int(yintersect))
  # return line coordinates
  return ((xminval, int(yintersect)), (xmaxval, int(yintersect)))

def lines_through_root_middle(circle_coords, root_image):
    """
    function to find the lines coordinates that are used to measure root width

    Args:
        circle_coords (list of numpy 1D array): pillars coordinates
        root_image (numpy 2D array): root mask image

    Returns:
        list: lines coordinates
    """
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


def distances_table(linescoords,scale_factor = 0.4023):
    distancespx = []
    distances = []
    factorcorrection= []
    count = []
    
    for i, (p1,p2) in enumerate(linescoords):
        distancespx.append(euc_distance(p1,p2))
        d = euc_distance(p1,p2)/scale_factor
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
        cv2.imwrite(os.path.join(folder, filenames[i]), img[:,:,[2,1,0]])

class SingleRootandPillarsdetection(object):
    """
    a class to detect the root and pillars position in an image
    
    ...
    Attributes
    ----------
    
    image: a list of numpy 2D array
        display all images information
    root_image: numpy 3D array 
        display the root masks
    pillars_coords: list of numpy 1D array
        contains each coordinates position (X and Y) and the circle radious (pixel)
    root_intersectionlines: list
        lines coordinates that are used to measured the root width
    pillars_intersectionlines: list
        lines coordinates that are used to measured the pillars separation
        
    Returns:
        _type_: _description_
    """
    
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

            rootmaskimg = self.root_image[i][:,self._minposx[i]:self._maxposx[i]]
            pillarscoords = self._filteredpillars_coords[i]

            if len(list(pillarscoords.keys()))>0:
                firstpillar = pillarscoords[list(pillarscoords.keys())[0]]
                
                firstline = add_line_ontop(firstpillar, 
                                rootmaskimg)
                
                linesinsigleimg = lines_through_root_middle(
                                                pillarscoords, 
                                                rootmaskimg)
                
                if not np.isnan(firstline[0][0]):
                    lou = [firstline]+linesinsigleimg
                else:
                    lou = linesinsigleimg
                
                lastpillar = pillarscoords[list(pillarscoords.keys())[-1]]
                
                lastline = add_line_onbottom(lastpillar, 
                                rootmaskimg)
                
                if not np.isnan(lastline[0][0]):
                    lou = lou+[lastline]
                
                lineslist.append(lou)
                
        return lineslist

    @property
    def pillars_intersectionlines(self):
        lineslist = []
        for i in range(len(self._filteredpillars_coords)):
            lineslist.append(get_pillars_lines(self._filteredpillars_coords[i]))

        return lineslist
    

    def _get_pillarsrawcoords(self):

        pillars_coords = {}
        radious = []
        for i in range(self.image.shape[0]):
            
            pillars = PillarImage(self.image[i,:,self._minposx[i]:self._maxposx[i],:],
                              minradius = self.minradius, 
                              maxradius = self.maxradius, 
                              max_circles= self.max_circles)
            
            pillars.find_circles(findlines = False)
            pillars.sort_circles()

            pillars_coords[i] = mergecoords(pillars.circle_coords.copy(),pillars.radious)
            print(f"{len(pillars_coords[i])} pillars were found")
            radious.append(np.nanmean(np.array(pillars_coords[i]).T[2]))
            
        self._raw_pillars_coords = pillars_coords
        self.radious = radious

    def _filtered_coords(self):
        
        pillars_coords = {}
        
        for i in range(len(self._raw_pillars_coords.keys())):
            coords_filtered, _ = circles_thatedge_root(self._raw_pillars_coords[i],
                                                                self.root_image[i,:,self._minposx[i]:self._maxposx[i]], 
                                                                self.radious[i])
            pillars_coords[i] = coords_filtered

        #self._warning_message  = warningmessage
        self._filteredpillars_coords = pillars_coords

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
        
           
    def _dic_root_xlocation(self, perc = 0.17):
        minposx = [0] * len(self._root_image)
        maxposx = [0] * len(self._root_image)
        for i in range(len(self._root_image)):
            minposx[i], maxposx[i] = shrink_to_root(self.root_image[i],perc=perc)
        
        self._minposx = minposx
        self._maxposx = maxposx

    def _reset(self):
      #self.root_intersectionline = None
      #self.pillars_coords = None
      #self.pillars_intersectionlines = None
      self.image = None
      self._minposx = None
      self._maxposx = None

    def classify_image(self, image_path):

        self._reset()
        self.image =  read_image(image_path)
        
        if len(np.array(self.image).shape) == 3:
            self.image = np.expand_dims(np.array(self.image), axis= 0)
    
        if type(self.image) is list:
            self._root_image = self.detector.detect_root(np.concatenate(self.image, axis=0))
        else:
            self._root_image = self.detector.detect_root(self.image)
        
        self._dic_root_xlocation(perc = self._root_red_perc)
        
        self._get_pillarsrawcoords()
        self._filtered_coords()
        
    def __init__(self, 

                 detector=None,
                 scale_factor = 0.4023,
                 minradius = 17, maxradius = 18, max_pillars_around_root= 18):
        
        """
        class initialization
        
        Args:
            imagery_path (str): directory path that contains the images.
            weigths_path (str, optional): directory path that contains the segmentation model weights. Defaults None.
            architecture (str, optional): segmentation model name. Defaults to vgg16.
            minradius (int, optional): minimun circle radious in pixels. Defaults to 17.
            maxradius (int, optional): maximun circle radious in pixels. Defaults to 18.
            max_circles (int, optional): maximun number of pillars around the root. Defaults to 18.
            imgsuffix (str, optional): images extension. Defaults to .jpg.

        """
        
        self.minradius = minradius
        self.maxradius = maxradius
        self.max_circles = max_pillars_around_root+2
        perc = 0.17
        self.scale_factor = scale_factor
        if self.max_circles == 22:
            self.minradius = 13
            self.maxradius = 14
            perc = 0.12
        self._root_red_perc = perc
        if self.max_circles == 18:
            self.minradius = 17
            self.maxradius = 18
            
        self.detector = detector
        

class RootandPillars(SingleRootandPillarsdetection):
    def __init__(self, imagery_path, 
                 weigths_path=None, 
                 architecture="vgg16", 
                 scale_factor=0.4023, 
                 minradius=17, maxradius=18, 
                 max_pillars_around_root=18, 
                 imgsuffix='.jpg'):
        
        self._already_processed = {}
        self.image_names = None
        self.img_path = [os.path.join(imagery_path,i) for i in os.listdir(imagery_path) if i.endswith(imgsuffix)]
        
        self.image_names =get_filenames(self.img_path)
        
        if type(self.image_names) is str:
            self.image_names = [self.image_names]
        
        detector = root_detector(weigths_path, architecture = architecture)
        
        super().__init__(detector, scale_factor, minradius, maxradius, max_pillars_around_root)


    def lines_as_table(self, linetype="pillar_lines"):
        dflist = []
        for img_path in list(self._already_processed.keys()):

            imginfo = self._already_processed[img_path]
            if imginfo is not None:
              df = distances_table(imginfo[linetype], self.scale_factor)
              df['object'] = 'pillar' if linetype == "pillar_lines" else 'root'
              df['image_name'] = get_filenames(img_path)
              dflist.append(df)

        if len(dflist)>0:
          dflist = pd.concat(dflist).reset_index()

        return dflist

    def classify_single_img(self, img_id):

        assert img_id <= len(self.img_path)

        imgpath = self.img_path[img_id]
        
        if imgpath not in list(self._already_processed.keys()):
          try:
            self.classify_image(imgpath)
            imgdict = {
              'root_image': self.root_image[0][:,self._minposx[0]:self._maxposx[0]].copy(),
              'clipped_image': self.image[0][:,self._minposx[0]:self._maxposx[0],:].copy(),
              'pillars_coords': self.pillars_coords[0],
              'radious':self.radious[0],
              'root_lines':self.root_intersectionlines[0],
              'pillar_lines':self.pillars_intersectionlines[0]
            }
            
          
          except:
            print("***** {} image was not possible to process".format(imgpath))
            imgdict = None
          
          self._already_processed[imgpath] = imgdict
        else:
          imgdict = self._already_processed[imgpath]
          

        return imgdict
    
    def classify_all_images(self):
      for i in range(len(self.img_path)):
        self.classify_single_img(i)


    def _get_final_images(self, pillars_color = (0, 153, 153), root_lines_color = (255, 102, 0)):
        imagestoplot = {}
        self.classify_all_images()

        for img_path in self.img_path:
          # check if the image was already processed
            
            imginfo = self._already_processed[img_path]
            #print(imginfo)
            if imginfo is not None:
                imagestoplot[img_path] = draw_lines_and_circles(
                    imginfo["clipped_image"],
                    imginfo["root_image"],
                    imginfo["pillars_coords"],
                    imginfo["radious"],
                    imginfo["root_lines"],
                    imginfo["pillar_lines"],
                    pillars_color = pillars_color, root_lines_color = root_lines_color)
            
        return imagestoplot

    def export_final_images(self, path, **kwargs):
        
        imagestoplot = self._get_final_images(**kwargs)
        image_names= []
        listimgs = []
        for img_path in list(imagestoplot.keys()):
          imginfo = imagestoplot[img_path]
          if imginfo is not None:
            listimgs.append(imagestoplot[img_path])
            image_names.append(get_filenames(img_path))

        export_images(listimgs, 
                      path, 
                      self.image_names)

    def export_detection_as_csv(self, filename):
        self.classify_all_images()
        rootlines = self.lines_as_table("root_lines")
        pillarlines = self.lines_as_table("pillar_lines")
        if len(rootlines)>0:
          pd.concat([rootlines,pillarlines]).to_csv(filename)
        else:
          print("No information was")

    def plot_final_layer(self, maximages = None, pillars_color = (0, 153, 153), root_lines_color = (255, 102, 0), figsize = (8,8)):
        """
        function to plot the detected root and pillars 

        Args:
            maximages (int, optional): a muximun number of images to be plotted. Defaults to None.
            pillars_color (tuple, optional): which color (rgb) will be used to plot the pillar. Defaults to (0, 153, 153).
            root_lines_color (tuple, optional): which color (rgb) will be used to fill the root. Defaults to (255, 102, 0).
            figsize (tuple, optional): figure size. Defaults to (8,8).

        Returns:
            matplotlib plot: plot
        """
        imagestoplot = self._get_final_images(pillars_color = pillars_color, root_lines_color = root_lines_color)
        processedimgspath = list(imagestoplot.keys())

        if maximages is None:
            nrows= self.root_image.shape[0]
            
        else:
            nrows = maximages
        
        fig, ax = plt.subplots(ncols=1, nrows=nrows, figsize =figsize, dpi = 80)
        i = 0
        while i < nrows:
            if maximages>1:
                ax[i].imshow(imagestoplot[processedimgspath[i]])
                ax[i].set_title(processedimgspath[i])
            else:
                ax.imshow(imagestoplot[processedimgspath[i]])
                ax.set_title(processedimgspath[i])
            i +=1
        
        return fig
