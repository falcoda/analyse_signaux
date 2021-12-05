import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import feature
from typing import List
import warnings


"""
"""
class Leaf:

    name = None
    img = None                # cv2 image
    grayscale = None          # cv2 grayscale img
    size = None               # w, h

    contour = None            # array of points defining the contour
    contour_approx = None     # array of points defining an approximation of the contour

    num_lobes = None          # number of lobes of the leaf
    lobe_aspect_ratio = None  # Perimeter/Area for one lobe

    perimeter = None          # Length of the countour of the leaf
    area = None               # Area of the leaf
    aspect_ratio = None       # Perimeter/Area of leaf
    
    bounding_ratio = None     # Aspect-ratio (base/height) of the bounding box (rectangle) of the leaf

    equi_diam = None          #
    rectangularity = None     #
    circularity = None        #

    props_list = []           # list containing all the caracterising proprties values

    def __init__(self, path: str):
        self.name = self.get_name(path)
        self.img = cv2.imread(path)
        self.grayscale = self.get_grayscale()
        self.size = self.img.shape

        self.contour = self.get_contour()
        self.contour_approx = self.get_approx_contour(0.01)
        
        self.num_lobes = self.get_num_lobes()
        self.lobe_aspect_ratio = self.get_lobe_aspact_ratio()
        
        self.perimeter = self.get_perimeter()
        self.area = self.get_area()
        self.aspect_ratio = self.get_aspact_ratio()
        
        self.bounding_ratio = self.get_bounding_box_aspect_ratio()
        
        self.equi_diam = self.get_equivalent_diameter()
        self.rectangularity = self.get_rectangularity()
        self.circularity =  self.get_circularity()

        self.props_list = self.get_props_list()
        
    def __repr__(self):
        return '''
        {name}:
        \tnumber of lobes: \t{num_lobes}
        \tlobe AR: \t\t{lobe_aspect_ratio}
        \tboundingBox AR: \t{bounding_ratio}
        \tgeneral AR: \t\t{aspect_ratio}
        \tequivalent diameter: \t{equi_diam}
        \trectangularity: \t{rectangularity}
        \tcircularity: \t\t{circularity}
        '''.format(
            name = self.name,
            num_lobes = str(self.num_lobes),
            lobe_aspect_ratio = str(self.lobe_aspect_ratio),
            bounding_ratio = str(self.bounding_ratio),
            aspect_ratio = str(self.aspect_ratio),
            equi_diam = str(self.equi_diam),
            rectangularity = str(self.rectangularity),
            circularity = str(self.circularity)
        )

    ################################
    # CLASS ATTRIBUTE CALC FUNCTIONS
    ################################

    def get_name(self, path: str) -> str:
      return path.split('/')[-1].split("\\")[-1].split('.')[0]

    def get_grayscale(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        # Converting image to grayscale
        gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Smoothing image using Guassian filter of size (25,25)
        blur = cv2.GaussianBlur(gs, (25, 25), 0)

        # Adaptive image thresholding using Otsu's thresholding method
        _, img_gs = cv2.threshold(gs, 150, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        return img_gs

    def get_contour(self):
        cont = []
        contours, _ = cv2.findContours(
            self.grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= len(cont):
                cont = contour

        return cont

    def get_approx_contour(self, resolution: float):
        approx_raw = cv2.approxPolyDP(
            self.contour, resolution * cv2.arcLength(self.contour, True), True)

        # boundary around point of which you should remove adjacent points based on a 50px / 700px image width
        offset = self.size[0] * 0.06
        return self.filter_contour_points(approx_raw, offset)

        # cv2.drawContours(self.img, [self.contour_approx], 0, (0, 255, 0), 5)
        # plt.imshow(self.img)
        # plt.show()

    def get_num_lobes(self):
        angles = []
        for idx, pt in enumerate(self.contour_approx):

            if (idx == (len(self.contour_approx) - 2)):
                ang = self.get_angle(pt[0], self.contour_approx[idx + 1]
                              [0], self.contour_approx[0][0])

            elif (idx == (len(self.contour_approx) - 1)):
                ang = self.get_angle(pt[0], self.contour_approx[0]
                              [0], self.contour_approx[1][0])

            else:
                ang = self.get_angle(pt[0], self.contour_approx[idx + 1]
                              [0], self.contour_approx[idx + 2][0])

            if (ang > 180):
                ang = 360 - ang

            if (ang > 125 or ang < 45):
                continue

            angles.append(ang)

        return len(angles)/2

    def get_lobe_aspact_ratio(self):
        tresh_value = 0
        current_index = 0
        current_points_color = [tresh_value, tresh_value]

        ratios = []

        if (len(self.contour_approx) <= 3 or self.num_lobes < 2):
            # no lobes
            return 0

        while(current_index < (len(self.contour_approx)-3)):

            p1, p2, p3 = self.extract_successive_points(current_index)
            p1p2_center = self.get_center_of_line(p1, p2)
            points = self.extract_two_pts_on_bisect(p1p2_center, p3)

            # Accessing a pixel in a grayscale : (y,x)  !!! and not (x,y)
            current_points_color[0] = self.grayscale[points[0][1], points[0][0]]
            current_points_color[1] = self.grayscale[points[1][1], points[1][0]]

            if (current_points_color[0] == tresh_value or current_points_color[1] == tresh_value):
                current_index += 1
                continue

            area = 0.5*abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

            a = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            b = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
            c = math.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
            perimeter = a + b + c

            ratios.append(area/perimeter)

            # ctr = np.array([p1,p2,p3]).reshape((-1,1,2)).astype(np.int32)
            # cv2.drawContours(self.img, [self.contour_approx], 0, (0, 255, 0), 5)
            # cv2.drawContours(self.img, [ctr], 0, (0, 0, 255), 2)
            # cv2.circle(self.img, (points[0][0],points[0][1]), radius=1, color=(255, 0, 0), thickness=5)
            # cv2.circle(self.img, (points[1][0],points[1][1]), radius=1, color=(255, 0, 0), thickness=5)
            # plt.imshow(self.img)
            # plt.show()

            current_index += 1

        avg_ratio = np.sum(ratios)/len(ratios)

        return avg_ratio

    def get_perimeter(self):
        return cv2.arcLength(self.contour, True)
    
    def get_area(self):
        return cv2.contourArea(self.contour)

    def get_aspact_ratio(self):
        return  self.perimeter/self.area

    def get_bounding_box_aspect_ratio(self):
      
        new_contour = self.filter_invalid_angles(self.contour_approx, 45, 179)

        rect = cv2.minAreaRect(new_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        base = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
        height = math.sqrt((box[1][0] - box[2][0])**2 + (box[1][1] - box[2][1])**2)

        # the closer the ration is to 1 the more square the leaf is.
        if (base > height):
            return base/height
        else:
            return height/base

        # cv2.drawContours(self.img,[box],0,(0,0,255),2)
        # cv2.drawContours(self.img, [self.contour_approx], 0, (0, 255, 0), 5)
        # cv2.drawContours(self.img, [new_contour], 0, (255, 0, 0), 5)
        # plt.imshow(self.img)
        # plt.show()
    
    def get_equivalent_diameter(self):
        return math.sqrt(2*self.area/math.pi)
    
    def get_rectangularity(self):
      new_contour = self.filter_invalid_angles(self.contour_approx, 30, 179)
      x, y, w, h = cv2.boundingRect(new_contour)
      return w*h/self.area

    def get_circularity(self):
      return ((self.perimeter)**2)/self.area
        
    def get_props_list(self) -> List[float]:
        return [
            self.num_lobes,
            self.lobe_aspect_ratio, 
            self.bounding_ratio, 
            #self.aspect_ratio, #! => is way too volatile, doesn't bring any added value -> rejected 
            #self.equi_diam, #! => is dépendant on image resolution -> rejected
            self.rectangularity, 
            self.circularity
        ]        

    ########################
    # CLASS HELPER FUNCTIONS
    ########################
    """
    Removed point that are too close too each other in the approximate contour
    """
    def filter_contour_points(self, points, offset):
        # all points within the range of the current point will be replaced by a single point including this one.

        newPoints = []
        removedIdx = []

        for idx, pt in enumerate(points):
            if (idx in removedIdx):
              continue

            x = pt[0][0]
            y = pt[0][1]

            if (idx == (len(points) - 1)):
              x2 = points[0][0][0]
              y2 = points[0][0][1]
            else:
              x2 = points[idx + 1][0][0]
              y2 = points[idx + 1][0][1]

            if ((x2 < (x + offset) and x2 > (x - offset)) and (y2 < (y + offset) and y2 > (y - offset))):
              # calc point that is in between those two points
              newPt = [int(round((x + x2)/2)), int(round((y + y2)/2))]
              removedIdx.append(idx+1)
            else:
              newPt = [int(pt[0][0]), int(pt[0][1])]

            newPoints.append(newPt)

        ctr = np.array(newPoints).reshape((-1, 1, 2)).astype(np.int32)

        return ctr

    """
    Get the angle between 3 points at b
    """
    def get_angle(self, a, b, c):
        ang = math.degrees(math.atan2(
            c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang

    """
    Remove too wide or too narrow angles in the approximate countour 
    """
    def filter_invalid_angles(self, points, min_angle: float, max_angle: float):
        newPoints = []
        for idx, pt in enumerate(points):
            if (idx == (len(points) - 2)):
                ang = self.get_angle(pt[0], points[idx + 1][0], points[0][0])
                newPt = points[idx + 1]

            elif (idx == (len(points) - 1)):
                ang = self.get_angle(pt[0], points[0][0], points[1][0])
                newPt = points[0]

            else:
                ang = self.get_angle(pt[0], points[idx + 1][0], points[idx + 2][0])
                newPt = points[idx + 1]

            if (ang > 180):
                ang = 360 - ang

            if (ang > max_angle or ang < min_angle):
                continue

            newPoints.append(newPt)

        ctr = np.array(newPoints).reshape((-1, 1, 2)).astype(np.int32)
        return ctr
    
    """
    Get the center point between two points 
    """
    def get_center_of_line(self, p1, p2):
        return [int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)]

    """
    Get 3 successive points at index 
    """
    def extract_successive_points(self, start_index: int):
        p1 = self.contour_approx[0 + start_index][0]
        p2 = self.contour_approx[1 + start_index][0]
        p3 = self.contour_approx[2 + start_index][0]

        return p1, p2, p3

    """
    Get two points equally spaced in between two outer points.
    """
    def extract_two_pts_on_bisect(self, p1, p2):
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
            [int(x1), int(y1)],
            [int(x2), int(y2)]
        ]

"""
"""
class LeafGroup:
    name: str = None
    leafs: List[Leaf] = None
    
    def __init__(self, path) -> None:
        self.name = self.get_name(path)
        self.leafs = []
    
    def __repr__(self) -> str:
        return self.name
    
    def get_name(self, path) -> str:
        return path.split('/')[-1].split("\\")[-1].split('.')[0]
    
    def add_leaf(self, leaf: Leaf) -> None:
        self.leafs.append(leaf)
    

ref_leaf_groups: List[LeafGroup] = []

is_first_run: bool = True

def get_ref_imgs(path):
    global ref_leaf_groups

    refs = os.listdir(path)

    if ('.DS_Store' in refs):
        refs.remove('.DS_Store')

    for ref in refs:
        leafs = os.listdir(path + '/' + ref)
        new_leafGroup = LeafGroup(ref)
        if ('.DS_Store' in leafs):
            leafs.remove('.DS_Store')
        for leaf in leafs:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    new_reference_leaf = Leaf(path + '/' + ref + '/' + leaf)
                    new_leafGroup.add_leaf(new_reference_leaf)
                except Exception as e: 
                    continue
          
        ref_leaf_groups.append(new_leafGroup)  

def calc_leaf_match(ref_leaf: Leaf, test_leaf: Leaf ) -> float: 
    prop_scores = []
    for idx, prop in enumerate(ref_leaf.props_list):
        diff = abs(test_leaf.props_list[idx] - prop)
        if diff == 0:
            prop_scores.append(0)
        else:
            if prop == 0:
                # discard this prop
                continue
            prop_scores.append(abs(diff/prop))
        
    return sum(prop_scores) / len(prop_scores)

def calc_matches(test_leaf: Leaf):
    global ref_leaf_groups
    
    no_match_scores = [] 
    for idx, leaf_group in enumerate(ref_leaf_groups):
        leaf_group_props_score = [] 
        for leaf in leaf_group.leafs:
            score = calc_leaf_match(leaf, test_leaf)
            leaf_group_props_score.append(score)
        
        no_match_mean_score = sum(leaf_group_props_score) / len(leaf_group_props_score)
        no_match_scores.append(no_match_mean_score)
        no_match_prc = round(no_match_mean_score*100,2)
        print('{match_prct} {ref}'.format(ref=leaf_group, match_prct=no_match_prc))
    
    return no_match_scores
        
"""     
------  MAIN ------
"""
def main(img: str = None):
    global is_first_run
    
    if is_first_run:
        get_ref_imgs('data')
        is_first_run = False

    if img != None:
        test_img_path = img
    else:
        test_img_path = 'test/unknown_11.jpg'
        
    test_img = Leaf(test_img_path)

    print('----------------------')
    print('MATCHING SCORE: (The lower the better)\n')
    match_scores = calc_matches(test_img)
    best_match = ref_leaf_groups[match_scores.index(min(match_scores))]
    
    print('===================')
    print('BEST MATCH ===> {match}'.format(match=best_match))
    print('===================')
    
    return best_match

if __name__ == '__main__':
  main()