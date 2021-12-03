import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import feature

class Image:
  name = None
  size = None
  img = None # cv2 image
  grayscale = None # cv2 grayscale img
  contour = None # array of points defining the contour
  contour_approx = None # array of points defining an approximation of the contour
  num_lobes = None 
  aspect_ratio = None # Perimeter/Area
  lobe_aspect_ratio = None # Perimeter/Area for one lobe 
  bounding_ratio = None # ratio (base/height) of the bounding box of the leaf
  area = None
  equi_diam = None
  rectangularity = None
  perimeter = None
  circularity = None

  def __init__(self):
    None
    
    
  def __str__(self):
    return self.name
  
ref_images = []


"""
"""
def calc_img_contour(img: Image):
    #Get contour
    cont = []
    contours, _ = cv2.findContours(img.grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= len(cont):
            cont = contour

    img.contour = cont
    
"""
Removed point that are too close too each other in the approximate contour
"""
def removeClosedPoints(points, offset):
    # all points within the range of the current point will be replaced by a single point including this one.
    
    newPoints = []
    removedIdx = []
    
    for idx, pt in enumerate(points):
        if (idx in removedIdx):
          continue
      
        x = pt[0][0]
        y = pt[0][1]
        
        if ( idx == (len(points) -1) ):  
          x2 = points[0][0][0]
          y2 = points[0][0][1]  
        else:
          x2 = points[idx + 1][0][0]
          y2 = points[idx + 1][0][1]  
        
        if ( (x2 < (x + offset) and x2 > (x - offset)) and (y2 < (y + offset) and y2 > (y - offset))):
          # calc point that is in between those two points 
          newPt = [int(round((x + x2)/2)), int(round((y + y2)/2))]
          removedIdx.append(idx+1)
        else:
          newPt = [int(pt[0][0]),int(pt[0][1])]
          
        newPoints.append(newPt)

    ctr = np.array(newPoints).reshape((-1,1,2)).astype(np.int32)
  
    return ctr

"""
Get the angle between 3 points at b
"""
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang
  
"""
Remove too wide or too narrow angles in the approximate countour 
"""    
def removeBadAngles(points, amin, amax):
  
  newPoints = []
  for idx, pt in enumerate(points):
    if (idx == (len(points) -2)):
      ang = getAngle(pt[0], points[idx +1][0], points[0][0])  
      newPt = points[idx +1]
      
    elif (idx == (len(points) -1)):
      ang = getAngle(pt[0], points[0][0], points[1][0])  
      newPt = points[0]
      
    else:
      ang = getAngle(pt[0], points[idx +1][0], points[idx + 2][0])
      newPt = points[idx +1]
      
    if (ang > 180):
      ang = 360 - ang

        
    if (ang > amax or ang < amin):
      continue 
    
    newPoints.append(newPt)
  
  ctr = np.array(newPoints).reshape((-1,1,2)).astype(np.int32)
  return ctr
    
def calc_img_approx_contour(img: Image):
    resolution = 0.01
    approx_raw= cv2.approxPolyDP(img.contour, resolution * cv2.arcLength(img.contour, True), True)

    offset = img.size[0] * 0.06 #boundary around point of which you should remove adjacent points based on a 50px / 700px image width 
    img.contour_approx = removeClosedPoints(approx_raw, offset) 
    
    #cv2.drawContours(img.img, [img.contour_approx], 0, (0, 255, 0), 5)
    #plt.imshow(img.img)
    #plt.show()
    
    

    
  
"""
------  LEAF LOBE SECTION ------
"""


"""
"""
def get_num_lobes(img: Image):

    angles = []
    for idx, pt in enumerate(img.contour_approx):

      if (idx == (len(img.contour_approx) -2)):
        ang = getAngle(pt[0], img.contour_approx[idx +1][0], img.contour_approx[0][0])  
        
      elif (idx == (len(img.contour_approx) -1)):
        ang = getAngle(pt[0], img.contour_approx[0][0], img.contour_approx[1][0])  
        
      else:
        ang = getAngle(pt[0], img.contour_approx[idx +1][0], img.contour_approx[idx + 2][0])
        
      if (ang > 180):
        ang = 360 - ang

          
      if (ang > 125 or ang < 45):
        continue 
      
      angles.append(ang)
    
    img.num_lobes = len(angles)/2

"""
"""
def compare_lobes(test_img: Image) -> Image:
    global ref_images
    
    match_value = 999999
    match_img = None
    
    for ref_img in ref_images:    
        diff = abs(ref_img.num_lobes - test_img.num_lobes)
        #if diff < match_value and diff <= 2 and test_img.num_lobes > 2:
        if diff < match_value :
            match_value = diff
            match_img = ref_img

    return match_img


"""
------  LEAF ASPECT RATIO ------
"""
        
def get_center_line(p1, p2):
    return [ int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2) ]

def extract_triangle_points(image: Image, start_index: int):
    p1 = image.contour_approx[0 + start_index][0]
    p2 = image.contour_approx[1 + start_index][0]
    p3 = image.contour_approx[2 + start_index][0]

    return p1, p2, p3

def extract_two_pts_on_bisect(p1, p2):
    if (p1[0] > p2[0]):
      x1 = p2[0] + 1/3 * (p1[0] - p2[0]) 
      x2 = p1[0] - 1/3 * (p1[0] - p2[0]) 
      a = (p2[1]-p1[1])/(p2[0]-p1[0])
    else:
      x1 = p1[0] + 1/3 * (p2[0] - p1[0]) 
      x2 = p2[0] - 1/3 * (p2[0] - p1[0]) 
      a = (p1[1]-p2[1])/(p1[0]-p2[0])
      
    b = p1[1] - a * p1[0] 
    
    y1 = a * x1 + b
    y2 = a * x2 + b
    
    return [
      [int(x1),int(y1)],
      [int(x2),int(y2)]
      ]

def calc_lobe_aspact_ratio(image: Image):
    
    tresh_value = 0
    current_index = 0
    current_points_color = [tresh_value, tresh_value]
    
    ratios = [] 
    
    if (len(image.contour_approx) <= 3 or image.num_lobes < 2 ):
      # no lobes
      image.lobe_aspect_ratio = 0
      return
  
    while( current_index < (len(image.contour_approx)-3) ):
          
        p1, p2, p3 = extract_triangle_points(image, current_index)      
        p1p2_center = get_center_line(p1,p2)
        points = extract_two_pts_on_bisect(p1p2_center, p3)
        
        # Accessing a pixel in a grayscale : (y,x)  !!! and not (x,y) 
        current_points_color[0] = image.grayscale[points[0][1],points[0][0]]
        current_points_color[1] = image.grayscale[points[1][1],points[1][0]]
      
        if (current_points_color[0] == tresh_value or current_points_color[1] == tresh_value):
            current_index += 1
            continue
      
        area = 0.5*abs( p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]) )
            
        a = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        b = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
        c = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
        perimeter = a + b + c
        
        ratios.append(area/perimeter)
      
        #ctr = np.array([p1,p2,p3]).reshape((-1,1,2)).astype(np.int32)
        #cv2.drawContours(image.img, [image.contour_approx], 0, (0, 255, 0), 5)
        #cv2.drawContours(image.img, [ctr], 0, (0, 0, 255), 2)
        #cv2.circle(image.img, (points[0][0],points[0][1]), radius=1, color=(255, 0, 0), thickness=5)
        #cv2.circle(image.img, (points[1][0],points[1][1]), radius=1, color=(255, 0, 0), thickness=5)
        #plt.imshow(image.img)
        #plt.show()
        
        current_index += 1
    
    avg_ratio = np.sum(ratios)/len(ratios)
    
    image.lobe_aspect_ratio = avg_ratio
     
def compare_lobe_ratios(test_img: Image):
    global ref_images
    
    match_value = 999999
    match_img = None
    
    for ref_img in ref_images:
        
        diff = abs(ref_img.lobe_aspect_ratio - test_img.lobe_aspect_ratio)
        if diff < match_value:
            match_value = diff
            match_img = ref_img

    return match_img


def calc_aspact_ratio(img: Image):

    area = cv2.contourArea(img.contour)
    perimeter = cv2.arcLength(img.contour, True)
    img.area = area
    img.perimeter = perimeter
    img.aspect_ratio = perimeter/area 

def get_hu(img: Image):
    image = cv2.bitwise_not(img.contour)
    #cv2.imshow("Bitwise Not", image)
    huInvars = cv2.HuMoments(cv2.moments(image)).flatten()  # Obtain hu moments from normalised moments in an array
    huInvars = -np.sign(huInvars) * np.log10(np.abs(huInvars))
    # huInvars /= huInvars.sum()
    print(huInvars)
    return huInvars

def obtainHuMoments(image_path, image):  # Obtains hu moments from image at path location
    hu = get_hu(image)
    image = cv2.imread(image_path, 0)
    lbp = LocalBinaryPatterns(24, 8)
    hist = lbp.describe(image)

    return hu, hist

def compare_aspect_ratio(test_img: Image):

    global ref_images
    
    match_value = 999999
    match_img = None
    
    for ref_img in ref_images:
        
        diff = abs(ref_img.aspect_ratio - test_img.aspect_ratio)
        if diff < match_value:
            match_value = diff
            match_img = ref_img

    return match_img


# Bounding Box
def calc_bounding_box_aspect_ratio(img: Image):
    
    new_contour = removeBadAngles(img.contour_approx, 45, 179)
    
    rect = cv2.minAreaRect(new_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    base = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
    height = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)
        
        
    # the closer the ration is to 1 the more square the leaf is.
    if (base > height):
      img.bounding_ratio = base/height
    else:
      img.bounding_ratio = height/base
          
    #cv2.drawContours(img.img,[box],0,(0,0,255),2)
    #cv2.drawContours(img.img, [img.contour_approx], 0, (0, 255, 0), 5)
    #cv2.drawContours(img.img, [new_contour], 0, (255, 0, 0), 5)
    #plt.imshow(img.img)
    #plt.show()

def compare_bbox_ratios(test_img: Image):
    global ref_images
    
    match_value = 999999
    match_img = None
    
    for ref_img in ref_images:
        
        diff = abs(ref_img.bounding_ratio - test_img.bounding_ratio)
        if diff < match_value:
            match_value = diff
            match_img = ref_img

    return match_img

def calc_equi_diameter(img:Image):
  equi_diam = math.sqrt(2*img.area/math.pi)
  img.equi_diam = equi_diam

def compare_equi_diam(test_img:Image):
  global ref_images
    
  match_value = 999999
  match_img = None
  
  for ref_img in ref_images:
      
      diff = abs(ref_img.equi_diam - test_img.equi_diam)
      if diff < match_value:
          match_value = diff
          match_img = ref_img

  return match_img

def calc_rect_circularity(img:Image):
  new_contour = removeBadAngles(img.contour_approx, 30, 179)
  x,y,w,h = cv2.boundingRect(new_contour)
  rectangularity = w*h/img.area
  circularity = ((img.perimeter)**2)/img.area

  img.rectangularity = rectangularity
  img.circularity = circularity

def compare_rect(test_img:Image):
  global ref_images
    
  match_value = 999999
  match_img = None
  
  for ref_img in ref_images:
      
      diff = abs(ref_img.rectangularity - test_img.rectangularity)
      if diff < match_value:
          match_value = diff
          match_img = ref_img

  return match_img

def compare_circularity(test_img:Image):
  global ref_images
    
  match_value = 999999
  match_img = None
  
  for ref_img in ref_images:
      
      diff = abs(ref_img.circularity - test_img.circularity)
      if diff < match_value:
          match_value = diff
          match_img = ref_img

  return match_img


"""
GENERAL FUNCTIONS
"""

"""
"""
def basic_manip(path: str, image: Image) :
    main_img = cv2.imread(path)
    
    img_size = main_img.shape
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)

    #Converting image to grayscale
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    #Smoothing image using Guassian filter of size (25,25)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    
    #Adaptive image thresholding using Otsu's thresholding method
    _, img_gs= cv2.threshold(gs,150,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
      
    image.img = main_img
    image.size = img_size
    image.grayscale = img_gs

"""
"""
def get_ref_imgs(path):
    global ref_images
    
    refs = os.listdir(path)
    if ('.DS_Store' in refs): 
      refs.remove('.DS_Store')
      
    for ref in refs:
        newRefImg = Image()
        basic_manip(path + '/' + ref, newRefImg)
        newRefImg.name = ref
        calc_img_contour(newRefImg)
        calc_img_approx_contour(newRefImg)
        get_num_lobes(newRefImg)
        calc_aspact_ratio(newRefImg)
        calc_lobe_aspact_ratio(newRefImg)
        calc_bounding_box_aspect_ratio(newRefImg)
        calc_equi_diameter(newRefImg)
        calc_rect_circularity(newRefImg)
        
        ref_images.append(newRefImg)
    
"""
"""
def get_best_match(list_matches):
    best_match = 0
    match_name = []
    num_match_per_ref = []
    
    for idx, ref_img in enumerate(ref_images):
        num_match_per_ref.append(list_matches.count(ref_img))
        
    total_number_of_matches = np.sum(num_match_per_ref)
    
    for idx, ref_img in enumerate(ref_images):

        match_percent = num_match_per_ref[idx]/total_number_of_matches # between 0 and 1
        
        print('leaf: ' + ref_images[idx].name.split(".")[0] + ' -> matching: ' + str("{:.2f}".format(match_percent*100)) + '%')
               
        if match_percent > best_match:
            best_match = match_percent
            match_name = [ref_images[idx].name]
    
    return match_name




"""
------  MAIN ------
"""


if __name__ == '__main__':

    match_list = []

    get_ref_imgs('data')
    
    test_img_path= 'test/unknown_2.jpg'
    test_img = Image()
    test_img.name = test_img_path
    basic_manip(test_img_path, test_img)
  
    calc_img_contour(test_img)
    calc_img_approx_contour(test_img)
    get_num_lobes(test_img)
    calc_aspact_ratio(test_img)
    calc_lobe_aspact_ratio(test_img)
    calc_bounding_box_aspect_ratio(test_img)
    calc_equi_diameter(test_img)
    calc_rect_circularity(test_img)
    get_hu(test_img)

    match_list.append(compare_lobes(test_img))
    match_list.append(compare_lobe_ratios(test_img))
    match_list.append(compare_bbox_ratios(test_img))
    match_list.append(compare_aspect_ratio(test_img)) # --> NOT RELEVANT - DOES NOT ADD PRECISION 
    match_list.append(compare_equi_diam(test_img))
    match_list.append(compare_circularity(test_img))
    match_list.append(compare_rect(test_img))    
    
    for ref in ref_images:
      print("------- Ref")
      print(ref.name)
      #print(ref.num_lobes)
      #print(ref.lobe_aspect_ratio)
      #print(ref.bounding_ratio)
      #print(ref.aspect_ratio)
      #print(ref.equi_diam)
      print(ref.rectangularity)
      print(ref.circularity)

    
    
    print("\n------- Test")
    print(test_img.name)
    #print(test_img.num_lobes)
    #print(test_img.lobe_aspect_ratio)
    #print(test_img.bounding_ratio)
    #print(test_img.aspect_ratio)
    #print(test_img.equi_diam)
    print(test_img.rectangularity)
    print(test_img.circularity)

    
    
    print("\n\n------------------")
    print("----- RESULT -----")
    get_best_match(match_list)
    print("\n")
    