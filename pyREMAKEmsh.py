import numpy
import math
import pygmsh
import time
import gmsh
import copy

import matplotlib.pyplot as plt
from numpy import linalg as LA

###################################################################################################
#
# Versions:
#
# Python 3.7.6
# NumPy 1.19.0
# pygmsh 7.1.8
# Gmsh 4.7.1
#
# This software is published under the GPLv3 license (https://www.gnu.org/licenses/gpl-3.0.en.html).
###################################################################################################

class pyREMAKEmsh:
    
    def __init__(self, geometry_data, tol, input_data_name):
        """
        Construct dictionary from input .json file and set tolerance for Gmsh.
        """
        
        self.geometry_data = geometry_data
        self.geometry_data['Points'] = {int(k): v for k, v in self.geometry_data['Points'].items()}
        self.geometry_data['Edges'] = {int(k): v for k, v in self.geometry_data['Edges'].items()}
        self.geometry_data['Stiffeners'] = {int(k): v for k, v in self.geometry_data['Stiffeners'].items()}
        self.geometry_data['Surfaces'] = {int(k): v for k, v in self.geometry_data['Surfaces'].items()}
        self.geometry_data['WebSurfaces'] = {int(k): v for k, v in self.geometry_data['WebSurfaces'].items()}
        self.geometry_data['SurfacesWithHoles'] = {int(k): v for k, v in self.geometry_data['SurfacesWithHoles'].items()}
        self.geometry_data['PointsForResponse'] = {int(k): v for k, v in self.geometry_data['PointsForResponse'].items()}

        self.tol = tol
        self.input_data_name = input_data_name
        self.MakeDict()


    def HoleInfo(self, surface_id):
        """
        Returns hole information from input dictionary.
        """
        surface_holes_info = list()
        for i in range(len(self.geometry_data['SurfacesWithHoles'][surface_id]['holes'])):
            
            hole_info = self.geometry_data['SurfacesWithHoles'][surface_id]['holes'][i]

            hole_center = [hole_info['x'], hole_info['y']]
            hole_rotation = hole_info['phi']
            hole_length_a = hole_info['a']
            hole_length_b = hole_info['b']
            hole_radius_on_edge = hole_info['r']

            surface_holes_info.append([hole_center, hole_length_a, hole_length_b, hole_radius_on_edge, hole_rotation])
        
        return surface_holes_info
        
    def MakeHole(self,surface_id):
        """
        Construct holes and update geometry dictionary.
        """
        def PlaneEquation(x, y, z):
            """
            Finds coefficients defining the plane defined with points x,y,z.
            """            
            a = (y[1] - x[1])*(z[2] - x[2]) - (z[1] - x[1])*(y[2] - x[2])
            b = (y[2] - x[2])*(z[0] - x[0]) - (z[2] - x[2])*(y[0] - x[0])
            c = (y[0] - x[0])*(z[1] - x[1]) - (z[0] - x[0])*(y[1] - x[1])
            d = -(a*x[0] + b*x[1] + c*x[2])
            arr = [a, b, c, d]
            
            return numpy.array(arr)

        def angle_triangle(point1, point2, point3):  
            """
            Calculate angle(in point1) in a triangle defined with point1, point2, point3.  
            """
            x1 = point1[0]
            y1 = point1[1]
            z1 = point1[2]

            x2 = point2[0]
            y2 = point2[1]
            z2 = point2[2]

            x3 = point3[0]
            y3 = point3[1]
            z3 = point3[2]

            num = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1)  
        
            den = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)*math.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)  
        
            angle = math.degrees(math.acos(num / den))  
        
            return round(angle, 3)

        def angle_triangle_2(point1, point2, point3):  
            """
            Calculating cosine of angle(in point1) in triangle defined with point1, point2, point3.
            """
            x1 = point1[0]
            y1 = point1[1]
            z1 = point1[2]

            x2 = point2[0]
            y2 = point2[1]
            z2 = point2[2]

            x3 = point3[0]
            y3 = point3[1]
            z3 = point3[2]

            num = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1)  
        
            den = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)*math.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)  
        
            return num / den

        def project_point_onto_plane(plane_coefficients, point_on_plane, point_to_project):
            """
            Calculates projection of a point_to_project onto plane defined with coefficients plane_coefficients.
            point_on_plane needed for calculation of direction.
            """
            plane_normal = numpy.array([plane_coefficients[0],plane_coefficients[1],plane_coefficients[2]])
            plane_normal = plane_normal/LA.norm(plane_normal)

            point_on_plane = numpy.array(point_on_plane)
            point_to_project = numpy.array(point_to_project)

            distance = numpy.dot(point_to_project - point_on_plane, plane_normal)
            projected_point = point_to_project - distance*plane_normal

            return projected_point

        def get_endpoints_for_virtual_stiffener(point0, point1, point2, point3, referent_point):
            """
            Returns endpoints of a virtual stiffener(rod) on a plane defined with point0, point1, point2, point3 w.r.t. referent_point.
            """
            point0 = numpy.array(point0)
            point1 = numpy.array(point1)
            point2 = numpy.array(point2)
            point3 = numpy.array(point3)
            referent_point = numpy.array(referent_point)

            dist0 = LA.norm(referent_point - point3)
            dist1 = LA.norm(referent_point - point0)

            angle0 = angle_triangle_2(point3, referent_point, point2)
            angle1 = angle_triangle_2(point0, point1, referent_point)
            
            x0 = dist0*angle0
            x1 = dist1*angle1

            coeff0 = x0/LA.norm(point3 - point2)
            coeff1 = x1/LA.norm(point1 - point0)

            endpoint0 = point3 - coeff0*(point3 - point2)
            endpoint1 = point0 + coeff1*(point1 - point0)
            endpoint0 = numpy.around(endpoint0, 16)
            endpoint1 = numpy.around(endpoint1, 16)
            endpoints = [list(endpoint0), list(endpoint1)]

            return endpoints

        def IfPointIsInSegment(P, Q1, Q2):
            """
            Checks if point P is in segment defined with points Q1 and Q2.
            """
            point_x = P[0]
            point_y = P[1]
            point_z = P[2]
            segment_start_x = Q1[0]
            segment_start_y = Q1[1]
            segment_start_z = Q1[2]
            segment_end_x = Q2[0]
            segment_end_y = Q2[1]
            segment_end_z = Q2[2]
            ab = math.sqrt((segment_end_x - segment_start_x)**2 + (segment_end_y - segment_start_y)**2 + (segment_end_z - segment_start_z)**2)
            ap = math.sqrt((point_x - segment_start_x)**2 + (point_y - segment_start_y)**2 + (point_z - segment_start_z)**2)
            pb = math.sqrt((point_x - segment_end_x)**2 + (point_y - segment_end_y)**2 + (point_z - segment_end_z)**2)
            
            if abs(ab - ap - pb) < self.tol:
                return True
            else:
                return False

        def OnEdgeOfSurfacesWith4Edges(P, Q1, Q2, Q3, Q4):
            """
            Checks if point P is inside surface defined with points Q1, Q2, Q3, Q4
            """
            if IfPointIsInSegment(P, Q1, Q2) or IfPointIsInSegment(P, Q2, Q3) or IfPointIsInSegment(P, Q3, Q4) or IfPointIsInSegment(P, Q4, Q1):
                return True
            else:
                False

        # get all data from surface with surface_id 
        tmp_surface_data = self.geometry_data['SurfacesWithHoles'][surface_id] 
        nodes_info = tmp_surface_data['nodesId']

        # if hole is on girder put virtual stiffeners, else skip
        if tmp_surface_data['noneWebFlange'] == "Web":
            add_virtual_stiffeners_flag = True
        else:
            add_virtual_stiffeners_flag = False

        holes_info = self.HoleInfo(surface_id)
        
        self.number_of_holes_counter =  self.number_of_holes_counter + len(holes_info)

        number_of_points = list()

        # check if surface has at least one right angle
        if angle_triangle(self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]],self.geometry_data['Points'][nodes_info[3]]) < 90 + self.tol  and angle_triangle(self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]],self.geometry_data['Points'][nodes_info[3]]) > 90 - self.tol:
            for i in range(len(holes_info)):

                # calculate vectors spanning local coordinate system 
                x_vector = numpy.array(numpy.array(self.geometry_data['Points'][nodes_info[1]]) - numpy.array(self.geometry_data['Points'][nodes_info[0]]))
                x_vector = x_vector/LA.norm(x_vector)

                y_vector = numpy.array(numpy.array(self.geometry_data['Points'][nodes_info[len(nodes_info)-1]]) - numpy.array(self.geometry_data['Points'][nodes_info[0]]))
                y_vector = y_vector/LA.norm(y_vector)

                tmp_points = list()
                tmp_surf = list()
                tmp_number_of_points_before = len(tmp_points)
                hole_center = numpy.array(self.geometry_data['Points'][nodes_info[0]]) + holes_info[i][0][0]*x_vector + holes_info[i][0][1]*y_vector  

                # read hole info
                a = holes_info[i][1]
                b = holes_info[i][2]
                r = holes_info[i][3]
                phi = holes_info[i][4]

                # define new vectors w.r.t. hole rotation
                if phi == 0.0:
                    x_vector_a = x_vector
                    y_vector_a = y_vector

                if phi > 0 and phi < 90:
                    phi_rad = phi*math.pi/180
                    x_vector_a = x_vector + y_vector*math.tan(phi_rad)
                    x_vector_a = x_vector_a/LA.norm(x_vector_a)

                    y_vector_a = y_vector - x_vector*math.tan(phi_rad)
                    y_vector_a = y_vector_a/LA.norm(y_vector_a)
                
                if phi > -90 and phi < 0:
                    phi_rad = phi*math.pi/180
                    x_vector_a = x_vector + y_vector*math.tan(phi_rad)
                    x_vector_a = x_vector_a/LA.norm(x_vector_a)

                    y_vector_a = y_vector - x_vector*math.tan(phi_rad)
                    y_vector_a = y_vector_a/LA.norm(y_vector_a)

                if phi == 90:
                    x_vector_a = y_vector
                    y_vector_a = -x_vector
                
                if phi == -90:
                    x_vector_a = -y_vector
                    y_vector_a = x_vector

            
                if not self.geometry_data['DistancesForMeshSize']:
                    tmp_dist = self.geometry_data["meshMaxSize"]
                else:
                    tmp_dist = 0.25*(min(self.geometry_data['DistancesForMeshSize']) + max(self.geometry_data['DistancesForMeshSize']))

                # check if hole has radius at the edges
                if r > 0:
                    # variable number of points for defining hole radius at the edges
                    number_of_points_on_circle = math.ceil(((r*math.pi)/2)/(2*tmp_dist))
                    coefficient_for_points = round(1/(number_of_points_on_circle+1),2)

                    # find points defining hole
                    if a != 0.0 and b != 0.0:
                        
                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)

                        tmp_points.append(hole_center + 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)

                        tmp_points.append(hole_center - 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector
                            
                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)

                    if a!=0 and b == 0:

                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a) 
                        tmp_points.append(hole_center + 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        tmp_points.append(hole_center - 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector

                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)

                    if a == 0 and b!=0:

                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))

                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)
                        
                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector
                          
                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)

                    if a == 0 and b == 0:

                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)

                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector

                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)


                    tmp_len_points = len(self.geometry_data['Points'].keys())
                    tmp_len_surface = list(self.geometry_data['Surfaces'].keys())[-1]
                    
                    for i in range(len(tmp_points)):
                        self.geometry_data['Points'][tmp_len_points + 1 + i] = tmp_points[i]
                        tmp_surf.append(tmp_len_points + 1 + i)
                    
                    self.geometry_data['Surfaces'][tmp_len_surface + 1] = tmp_surf

                    # add virtual stiffeners
                    if add_virtual_stiffeners_flag == True:
                        tmp_len_points = len(self.geometry_data['Points'].keys())                        
                        tmp_len_virtual_stiffeners = list(self.virtual_stiffeners_dict.keys())[-1]

                        if OnEdgeOfSurfacesWith4Edges(endpoints1[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints1[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 1] = endpoints1[0]
                            self.geometry_data['Points'][tmp_len_points + 2] = endpoints1[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 1] = [tmp_len_points + 1, tmp_len_points + 2]
                        if OnEdgeOfSurfacesWith4Edges(endpoints2[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints2[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 3] = endpoints2[0]
                            self.geometry_data['Points'][tmp_len_points + 4] = endpoints2[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 2] = [tmp_len_points + 3, tmp_len_points + 4]

                else:
                    point1 = hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a
                    point3 = hole_center + 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a
                    point5 = hole_center + (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a
                    point7 = hole_center - 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a
                    
                    tmp_points.append(point1)
                    tmp_points.append(point3)
                    tmp_points.append(point5)
                    tmp_points.append(point7)
                    
                    if add_virtual_stiffeners_flag == True:
                        if phi != 0:
                            point10 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            point20 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                        else:
                            point10 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                            point20 = hole_center + (0.5*a + r + tmp_dist)*x_vector

                        endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point10)
                        endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point20)
                    
                    tmp_len_points = len(self.geometry_data['Points'].keys())
                    tmp_len_surface = list(self.geometry_data['Surfaces'].keys())[-1]

                    for i in range(len(tmp_points)):
                        self.geometry_data['Points'][tmp_len_points + 1 + i] = tmp_points[i]
                        tmp_surf.append(tmp_len_points + 1 + i)
                    
                    self.geometry_data['Surfaces'][tmp_len_surface + 1] = tmp_surf

                    if add_virtual_stiffeners_flag == True:
                        tmp_len_points = len(self.geometry_data['Points'].keys())
                        tmp_len_virtual_stiffeners = list(self.virtual_stiffeners_dict.keys())[-1]

                        if OnEdgeOfSurfacesWith4Edges(endpoints1[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints1[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 1] = endpoints1[0]
                            self.geometry_data['Points'][tmp_len_points + 2] = endpoints1[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 1] = [tmp_len_points + 1, tmp_len_points + 2]
                        if OnEdgeOfSurfacesWith4Edges(endpoints2[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints2[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 3] = endpoints2[0]
                            self.geometry_data['Points'][tmp_len_points + 4] = endpoints2[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 2] = [tmp_len_points + 3, tmp_len_points + 4] 
                    
                tmp_number_of_points_after = len(tmp_points)
                number_of_points.append(tmp_number_of_points_after - tmp_number_of_points_before)

        # check if surface has at least one right angle
        else:
            for i in range(len(holes_info)):
                
                # calculate vectors spanning local coordinate system 
                x_vector = numpy.array(numpy.array(self.geometry_data['Points'][nodes_info[1]]) - numpy.array(self.geometry_data['Points'][nodes_info[0]]))
                x_vector = x_vector/LA.norm(x_vector)
                
                plane_coeff_1 = PlaneEquation(self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]],self.geometry_data['Points'][nodes_info[2]])           
                plane_coeff_2 = [x_vector[0], x_vector[1], x_vector[2], 0]
            
                y_vector = numpy.array(numpy.array(self.geometry_data['Points'][nodes_info[0]]) + numpy.cross(numpy.array(plane_coeff_1[0:3]), numpy.array(plane_coeff_2[0:3])))
                y_vector = y_vector/LA.norm(y_vector)

                tmp_points = list()
                tmp_surf = list()
                tmp_number_of_points_before = len(tmp_points)
                hole_center = numpy.array(self.geometry_data['Points'][nodes_info[0]]) + holes_info[i][0][0]*x_vector + holes_info[i][0][1]*y_vector  

                # read hole info
                a = holes_info[i][1]
                b = holes_info[i][2]
                r = holes_info[i][3]
                phi = holes_info[i][4]

                # define new vectors w.r.t. hole rotation
                if phi == 0.0:
                    x_vector_a = x_vector
                    y_vector_a = y_vector

                if phi > 0 and phi < 90:
                    phi_rad = phi*math.pi/180
                    x_vector_a = x_vector + y_vector*math.tan(phi_rad)
                    x_vector_a = x_vector_a/LA.norm(x_vector_a)

                    y_vector_a = y_vector - x_vector*math.tan(phi_rad)
                    y_vector_a = y_vector_a/LA.norm(y_vector_a)
                
                if phi > -90 and phi < 0:
                    phi_rad = phi*math.pi/180
                    x_vector_a = x_vector + y_vector*math.tan(phi_rad)
                    x_vector_a = x_vector_a/LA.norm(x_vector_a)

                    y_vector_a = y_vector - x_vector*math.tan(phi_rad)
                    y_vector_a = y_vector_a/LA.norm(y_vector_a)

                if phi == 90:
                    x_vector_a = y_vector
                    y_vector_a = -x_vector
                
                if phi == -90:
                    x_vector_a = -y_vector
                    y_vector_a = x_vector
                    
                if not self.geometry_data['DistancesForMeshSize']:
                    tmp_dist = self.geometry_data["meshMaxSize"]
                else:
                    tmp_dist = 0.25*(min(self.geometry_data['DistancesForMeshSize']) + max(self.geometry_data['DistancesForMeshSize']))

                # check if hole has radius at the edges
                if r > 0:
                    # variable number of points for defining hole radius at the edges
                    number_of_points_on_circle = math.ceil(((r*math.pi)/2)/(2*tmp_dist))
                    coefficient_for_points = round(1/(number_of_points_on_circle+1),2)

                    # find points defining hole
                    if a != 0.0 and b != 0.0:
                        
                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)

                        tmp_points.append(hole_center + 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)

                        tmp_points.append(hole_center - 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)

                        for i in range(len(tmp_points)):
                            tmp_points[i] = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]], tmp_points[i])

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector
                            
                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)

                    if a!=0 and b == 0:

                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a) 
                        tmp_points.append(hole_center + 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        tmp_points.append(hole_center - 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))

                        for i in range(len(tmp_points)):
                            tmp_points[i] = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]], tmp_points[i])

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector

                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)

                    if a == 0 and b!=0:

                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))

                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a)
                        
                        for i in range(len(tmp_points)):
                            tmp_points[i] = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]], tmp_points[i])

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector
                          
                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)

                    if a == 0 and b == 0:

                        tmp_points.append(hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a - 0.5*b*y_vector_a - r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center - 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a)
                        
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a - 0.5*b*y_vector_a + r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))
                        tmp_points.append(hole_center + (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a)
                        
                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center + (0.5*a)*x_vector_a + 0.5*b*y_vector_a + r*((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a)/LA.norm((1-tmp_coeff)*x_vector_a + tmp_coeff*y_vector_a))
                        tmp_points.append(hole_center + 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a)

                        for i in range(1,number_of_points_on_circle+1):
                            tmp_coeff = i*coefficient_for_points
                            tmp_points.append(hole_center - (0.5*a)*x_vector_a + 0.5*b*y_vector_a - r*(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a)/LA.norm(tmp_coeff*x_vector_a - (1-tmp_coeff)*y_vector_a))

                        for i in range(len(tmp_points)):
                            tmp_points[i] = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]], tmp_points[i])

                        if add_virtual_stiffeners_flag == True:
                            if phi != 0:
                                point1 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                                point2 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            else:
                                point1 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                                point2 = hole_center + (0.5*a + r + tmp_dist)*x_vector

                            endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point1)
                            endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point2)


                    tmp_len_points = len(self.geometry_data['Points'].keys())
                    tmp_len_surface = list(self.geometry_data['Surfaces'].keys())[-1]
                    
                    for i in range(len(tmp_points)):
                        self.geometry_data['Points'][tmp_len_points + 1 + i] = tmp_points[i]
                        tmp_surf.append(tmp_len_points + 1 + i)
                    
                    self.geometry_data['Surfaces'][tmp_len_surface + 1] = tmp_surf

                    if add_virtual_stiffeners_flag == True:
                        tmp_len_points = len(self.geometry_data['Points'].keys())                        
                        tmp_len_virtual_stiffeners = list(self.virtual_stiffeners_dict.keys())[-1]
                        
                        if OnEdgeOfSurfacesWith4Edges(endpoints1[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints1[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 1] = endpoints1[0]
                            self.geometry_data['Points'][tmp_len_points + 2] = endpoints1[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 1] = [tmp_len_points + 1, tmp_len_points + 2]
                        if OnEdgeOfSurfacesWith4Edges(endpoints2[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints2[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 3] = endpoints2[0]
                            self.geometry_data['Points'][tmp_len_points + 4] = endpoints2[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 2] = [tmp_len_points + 3, tmp_len_points + 4]
                
                else:
                    point1 = hole_center - (0.5*a + r)*x_vector_a - 0.5*b*y_vector_a
                    point3 = hole_center + 0.5*a*x_vector_a - (0.5*b + r)*y_vector_a
                    point5 = hole_center + (0.5*a + r)*x_vector_a + 0.5*b*y_vector_a
                    point7 = hole_center - 0.5*a*x_vector_a + (0.5*b + r)*y_vector_a

                    point1 = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]],point1)
                    point3 = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]],point3)
                    point5 = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]],point5)
                    point7 = project_point_onto_plane(plane_coeff_1,self.geometry_data['Points'][nodes_info[0]],point7)

                    tmp_points.append(point1)
                    tmp_points.append(point3)
                    tmp_points.append(point5)
                    tmp_points.append(point7)
                
                    if add_virtual_stiffeners_flag == True:
                        if phi != 0:
                            point10 = hole_center - (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                            point20 = hole_center + (math.sqrt(0.25*a**2 + 0.25*b**2) + r + tmp_dist)*x_vector
                        else:
                            point10 = hole_center - (0.5*a + r + tmp_dist)*x_vector
                            point20 = hole_center + (0.5*a + r + tmp_dist)*x_vector

                        endpoints1 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point10)
                        endpoints2 = get_endpoints_for_virtual_stiffener(self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]], point20)
                    
                    tmp_len_points = len(self.geometry_data['Points'].keys())
                    tmp_len_surface = list(self.geometry_data['Surfaces'].keys())[-1]

                    for i in range(len(tmp_points)):
                        self.geometry_data['Points'][tmp_len_points + 1 + i] = tmp_points[i]
                        tmp_surf.append(tmp_len_points + 1 + i)
                    
                    self.geometry_data['Surfaces'][tmp_len_surface + 1] = tmp_surf

                    # add virtual stiffeners
                    if add_virtual_stiffeners_flag == True:
                        tmp_len_points = len(self.geometry_data['Points'].keys())
                        tmp_len_virtual_stiffeners = list(self.virtual_stiffeners_dict.keys())[-1]

                        if OnEdgeOfSurfacesWith4Edges(endpoints1[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints1[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 1] = endpoints1[0]
                            self.geometry_data['Points'][tmp_len_points + 2] = endpoints1[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 1] = [tmp_len_points + 1, tmp_len_points + 2]
                        if OnEdgeOfSurfacesWith4Edges(endpoints2[0], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[2]], self.geometry_data['Points'][nodes_info[3]]) and OnEdgeOfSurfacesWith4Edges(endpoints2[1], self.geometry_data['Points'][nodes_info[0]],self.geometry_data['Points'][nodes_info[1]], self.geometry_data['Points'][nodes_info[0]], self.geometry_data['Points'][nodes_info[3]]):
                            self.geometry_data['Points'][tmp_len_points + 3] = endpoints2[0]
                            self.geometry_data['Points'][tmp_len_points + 4] = endpoints2[1]
                            self.virtual_stiffeners_dict[tmp_len_virtual_stiffeners + 2] = [tmp_len_points + 3, tmp_len_points + 4] 
                                         
                tmp_number_of_points_after = len(tmp_points)
                number_of_points.append(tmp_number_of_points_after - tmp_number_of_points_before)

    def ChangeDictionaryForWarpedSurfaces(self):
        """ 
        Divide into 16 new surfaces and make dictionary for Warped surfaces from input dictionary - key: surface id, value: point id-s for that surface
        """
        def PlaneEquation(x, y, z):
            """
            Finds coefficients defining the plane
            """
            a = (y[1] - x[1])*(z[2] - x[2]) - (z[1] - x[1])*(y[2] - x[2])
            b = (y[2] - x[2])*(z[0] - x[0]) - (z[2] - x[2])*(y[0] - x[0])
            c = (y[0] - x[0])*(z[1] - x[1]) - (z[0] - x[0])*(y[1] - x[1])
            d = -(a*x[0] + b*x[1] + c*x[2])
            arr = [a, b, c, d]
            return arr

        def OnPlane(P, plane_coefficients):
            """
            Checks if point P lies on plane defined with plane_coefficients
            """
            if abs(P[0]*plane_coefficients[0] + P[1]*plane_coefficients[1] + P[2]*plane_coefficients[2] + plane_coefficients[3]) < self.tol:
                return True
            else:
                return False
   
        surfaces_dict = {}
        surfaces_dict_tmp = {}
        points_dict = {}
        surfaces = self.geometry_data['Surfaces']
        
        self.dist = list()
        self.old_warped_new_warped = {}
        self.old_warped_new_warped_curve_loop = {}
        self.old_warped_new_warped_surface_ids = {}

        #for k,v in self.geometry_data["Surfaces"].items(): print(k , v)
        #print(len(self.geometry_data["Surfaces"].keys()))
        self.warped_flag = 0
        for i in surfaces.keys():
            if len(surfaces[i]) == 4:
                point0 = self.geometry_data['Points'][surfaces[i][0]]
                point1 = self.geometry_data['Points'][surfaces[i][1]]
                point2 = self.geometry_data['Points'][surfaces[i][2]]
                point3 = self.geometry_data['Points'][surfaces[i][3]]
                plane_coefficients = PlaneEquation(point0, point1, point2)
                if OnPlane(point3, plane_coefficients) == False:
                    flag = 1
                    self.warped_flag = 1
                    surfaces_dict_key = i
                    surfaces_dict[surfaces_dict_key] = surfaces[i]
        for i in surfaces_dict.keys():
            surfaces.pop(i, None)
        
        tmp_length_surfaces = list(self.geometry_data['Surfaces'].keys())[-1]
        
        tmp_length_points = list(self.geometry_data['Points'].keys())[-1]
        
        #for k, v in surfaces_dict.items(): print(k, v) # better way to print!
        #for k, v in self.points_dict_copy.items(): print(k, v) # better way to print!

        for i in surfaces_dict.keys():
            self.old_warped_new_warped[i] = list()
            self.old_warped_new_warped_curve_loop[i] = list()
            self.old_warped_new_warped_surface_ids[i] = list()

            point0 = numpy.array(self.geometry_data['Points'][surfaces_dict[i][0]])
            point1 = numpy.array(self.geometry_data['Points'][surfaces_dict[i][1]])
            point2 = numpy.array(self.geometry_data['Points'][surfaces_dict[i][2]])
            point3 = numpy.array(self.geometry_data['Points'][surfaces_dict[i][3]])

            self.dist.append(LA.norm(0.25*(point3 - point0)))
            self.dist.append(LA.norm(0.25*(point2 - point1)))

            point4 = list(point0 + 0.25*(point3 - point0))
            point6 = list(point0 + 0.5*(point3 - point0))
            point8 = list(point0 + 0.75*(point3 - point0))

            point5 = list(point1 + 0.25*(point2 - point1))
            point7 = list(point1 + 0.5*(point2 - point1))
            point9 = list(point1 + 0.75*(point2 - point1))

            point10 = list(point0 + 0.25*(point1 - point0))
            point12 = list(point0 + 0.5*(point1 - point0))
            point14 = list(point0 + 0.75*(point1 - point0))

            point11 = list(point3 + 0.25*(point2 - point3))
            point13 = list(point3 + 0.5*(point2 - point3))
            point15 = list(point3 + 0.75*(point2 - point3))

            point10_nmp = numpy.array(point10)
            point11_nmp = numpy.array(point11)
            point12_nmp = numpy.array(point12)
            point13_nmp = numpy.array(point13)
            point14_nmp = numpy.array(point14)
            point15_nmp = numpy.array(point15)

            point16 = list(point10 + 0.25*(point11_nmp - point10_nmp))
            point17 = list(point10 + 0.5*(point11_nmp - point10_nmp))
            point18 = list(point10 + 0.75*(point11_nmp - point10_nmp))

            point19 = list(point12 + 0.25*(point13_nmp - point12_nmp))
            point20 = list(point12 + 0.5*(point13_nmp - point12_nmp))
            point21 = list(point12 + 0.75*(point13_nmp - point12_nmp))
            
            point22 = list(point14 + 0.25*(point15_nmp - point14_nmp))
            point23 = list(point14 + 0.5*(point15_nmp - point14_nmp))
            point24 = list(point14 + 0.75*(point15_nmp - point14_nmp))


            points_dict[tmp_length_points + 1] = point4
            points_dict[tmp_length_points + 2] = point5
            points_dict[tmp_length_points + 3] = point6
            points_dict[tmp_length_points + 4] = point7
            points_dict[tmp_length_points + 5] = point8
            points_dict[tmp_length_points + 6] = point9
            points_dict[tmp_length_points + 7] = point10
            points_dict[tmp_length_points + 8] = point11
            points_dict[tmp_length_points + 9] = point12
            points_dict[tmp_length_points + 10] = point13
            points_dict[tmp_length_points + 11] = point14
            points_dict[tmp_length_points + 12] = point15
            points_dict[tmp_length_points + 13] = point16
            points_dict[tmp_length_points + 14] = point17
            points_dict[tmp_length_points + 15] = point18
            points_dict[tmp_length_points + 16] = point19
            points_dict[tmp_length_points + 17] = point20
            points_dict[tmp_length_points + 18] = point21
            points_dict[tmp_length_points + 19] = point22
            points_dict[tmp_length_points + 20] = point23
            points_dict[tmp_length_points + 21] = point24

            surfaces_dict_tmp[tmp_length_surfaces + 1] = [surfaces_dict[i][0], tmp_length_points + 7, tmp_length_points + 13, tmp_length_points + 1]
            surfaces_dict_tmp[tmp_length_surfaces + 2] = [tmp_length_points + 1, tmp_length_points + 13, tmp_length_points + 14, tmp_length_points + 3]
            surfaces_dict_tmp[tmp_length_surfaces + 3] = [tmp_length_points + 3, tmp_length_points + 14, tmp_length_points + 15, tmp_length_points + 5]
            surfaces_dict_tmp[tmp_length_surfaces + 4] = [tmp_length_points + 5, tmp_length_points + 15, tmp_length_points + 8, surfaces_dict[i][3]]

            surfaces_dict_tmp[tmp_length_surfaces + 5] = [tmp_length_points + 7, tmp_length_points + 9, tmp_length_points + 16, tmp_length_points + 13]
            surfaces_dict_tmp[tmp_length_surfaces + 6] = [tmp_length_points + 13, tmp_length_points + 16, tmp_length_points + 17, tmp_length_points + 14]
            surfaces_dict_tmp[tmp_length_surfaces + 7] = [tmp_length_points + 14, tmp_length_points + 17, tmp_length_points + 18, tmp_length_points + 15]
            surfaces_dict_tmp[tmp_length_surfaces + 8] = [tmp_length_points + 15, tmp_length_points + 18, tmp_length_points + 10, tmp_length_points + 8]

            surfaces_dict_tmp[tmp_length_surfaces + 9] = [tmp_length_points + 9, tmp_length_points + 11, tmp_length_points + 19, tmp_length_points + 16]
            surfaces_dict_tmp[tmp_length_surfaces + 10] = [tmp_length_points + 16, tmp_length_points + 19, tmp_length_points + 20, tmp_length_points + 17]
            surfaces_dict_tmp[tmp_length_surfaces + 11] = [tmp_length_points + 17, tmp_length_points + 20, tmp_length_points + 21, tmp_length_points + 18]
            surfaces_dict_tmp[tmp_length_surfaces + 12] = [tmp_length_points + 18, tmp_length_points + 21, tmp_length_points + 12, tmp_length_points + 10]

            surfaces_dict_tmp[tmp_length_surfaces + 13] = [tmp_length_points + 11, surfaces_dict[i][1], tmp_length_points + 2, tmp_length_points + 19]
            surfaces_dict_tmp[tmp_length_surfaces + 14] = [tmp_length_points + 19, tmp_length_points + 2, tmp_length_points + 4, tmp_length_points + 20]
            surfaces_dict_tmp[tmp_length_surfaces + 15] = [tmp_length_points + 20, tmp_length_points + 4, tmp_length_points + 6, tmp_length_points + 21]
            surfaces_dict_tmp[tmp_length_surfaces + 16] = [tmp_length_points + 21, tmp_length_points + 6, surfaces_dict[i][2], tmp_length_points + 12]

            for j in range(1,17):
                self.old_warped_new_warped[i].append(tmp_length_surfaces + j)

            tmp_length_points = tmp_length_points + 21
            tmp_length_surfaces = tmp_length_surfaces + 16 
            
        self.geometry_data["Surfaces"].update(surfaces_dict_tmp)
        self.geometry_data["Points"].update(points_dict)
        
        #for k,v in self.geometry_data["Surfaces"].items(): print(k , v)
        #print(len(self.geometry_data["Surfaces"].keys()))

    def ConstructGeometry(self):
        """
        Making geometry and mesh
        """
        with pygmsh.occ.Geometry() as geom:

            def PlaneEquation(x, y, z):
                """
                Finds coefficients defining the plane
                """
                a = (y[1] - x[1])*(z[2] - x[2]) - (z[1] - x[1])*(y[2] - x[2])
                b = (y[2] - x[2])*(z[0] - x[0]) - (z[2] - x[2])*(y[0] - x[0])
                c = (y[0] - x[0])*(z[1] - x[1]) - (z[0] - x[0])*(y[1] - x[1])
                d = -(a*x[0] + b*x[1] + c*x[2])
                arr = [a, b, c, d]
                return arr

            def OnPlane(P, plane_coefficients):
                """
                Checks if point P lies on plane defined with plane_coefficients
                """
                if abs(P[0]*plane_coefficients[0] + P[1]*plane_coefficients[1] + P[2]*plane_coefficients[2] + plane_coefficients[3]) < self.tol:
                    return True
                else:
                    return False

            def InTriangle(P, Q1, Q2, Q3):
                """
                Checks if point P is inside triangle defined with points Q1, Q2, Q3.
                """
                if abs(P[0] - Q1[0]) < self.tol and abs(P[1] - Q1[1]) < self.tol and abs(P[2] - Q1[2]) < self.tol:
                    return True 
                if abs(P[0] - Q2[0]) < self.tol and abs(P[1] - Q2[1]) < self.tol and abs(P[2] - Q2[2]) < self.tol:
                    return True 
                if abs(P[0] - Q3[0]) < self.tol and abs(P[1] - Q3[1]) < self.tol and abs(P[2] - Q3[2]) < self.tol:
                    return True 

                ab = [(Q2[0] - Q1[0]), (Q2[1] - Q1[1]), (Q2[2] - Q1[2])]
                ac = [(Q3[0] - Q1[0]), (Q3[1] - Q1[1]), (Q3[2] - Q1[2])]
                
                triangle_area = LA.norm(numpy.cross(ab,ac))/2

                pa = [(Q1[0] - P[0]), (Q1[1] - P[1]), (Q1[2] - P[2])]
                pb = [(Q2[0] - P[0]), (Q2[1] - P[1]), (Q2[2] - P[2])]
                pc = [(Q3[0] - P[0]), (Q3[1] - P[1]), (Q3[2] - P[2])]

                alpha = LA.norm(numpy.cross(pb, pc))/(2*triangle_area)
                beta = LA.norm(numpy.cross(pc, pa))/(2*triangle_area)
                gamma = LA.norm(numpy.cross(pa, pb))/(2*triangle_area)
                abg = alpha + beta + gamma

                if alpha >=-self.tol and beta >=-self.tol and gamma >=-self.tol and abg> 1 - self.tol and abg< 1 + self.tol:
                    return True
                else:
                    return False

            def InSurfaceWith3Edges(P, Q1, Q2, Q3):
                """
                Checks if point P is inside surface defined with points Q1, Q2, Q3.
                """
                if InTriangle(P, Q1, Q2, Q3):
                    return True
                else:
                    False
            
            def InSurfaceWith4Edges(P, Q1, Q2, Q3, Q4):
                """
                Checks if point P is inside surface defined with points Q1, Q2, Q3, Q4.
                """
                if InTriangle(P, Q1, Q2, Q3) or InTriangle(P, Q2, Q3, Q4) or InTriangle(P, Q3, Q4, Q1) or InTriangle(P, Q4, Q1, Q2):
                    return True
                else:
                    False

            def IfPointIsInSegment(point_x, point_y, point_z, segment_start_x, segment_start_y, segment_start_z, segment_end_x, segment_end_y, segment_end_z):
                """
                Checks if point with coordinates point_x, point_y, point_z is inside segment 
                with endpoints whose coordinates are segment_start_x, segment_start_y, segment_start_z, segment_end_x, segment_end_y, segment_end_z.
                """
                ab = math.sqrt((segment_end_x - segment_start_x)**2 + (segment_end_y - segment_start_y)**2 + (segment_end_z - segment_start_z)**2)
                ap = math.sqrt((point_x - segment_start_x)**2 + (point_y - segment_start_y)**2 + (point_z - segment_start_z)**2)
                pb = math.sqrt((point_x - segment_end_x)**2 + (point_y - segment_end_y)**2 + (point_z - segment_end_z)**2)
                
                if abs(ab - ap - pb) < self.tol:
                    return True
                else:
                    return False
                          
            self.virtual_stiffeners_dict = {0:0}
            self.number_of_holes_counter = 0

            for key in self.geometry_data['SurfacesWithHoles'].keys():
                self.MakeHole(key)

            del self.virtual_stiffeners_dict[0]

            # add points
            self.Points_done = {}
            for i in self.geometry_data['Points'].keys():
                tmp_add_point_var = geom.add_point(self.geometry_data['Points'][i]) 
                self.Points_done[i] = tmp_add_point_var
        
            # add stiffeners
            self.Lines_on_2edged_surface_done = list()
            for i in self.geometry_data['Stiffeners'].keys():
                self.Lines_on_2edged_surface_done.append(geom.add_line(self.Points_done[self.geometry_data['Stiffeners'][i][0]], self.Points_done[self.geometry_data['Stiffeners'][i][1]]))

            # add edges
            self.Edges_done = list()
            for i in self.geometry_data['Edges'].keys():
                self.Edges_done.append(geom.add_line(self.Points_done[self.geometry_data['Edges'][i][0]], self.Points_done[self.geometry_data['Edges'][i][1]]))
            
            # add virtual stiffeners
            self.edges_done_last_index = len(self.Edges_done)
            if self.warped_flag == 0:
                for i in self.virtual_stiffeners_dict.keys():
                    self.Edges_done.append(geom.add_line(self.Points_done[self.virtual_stiffeners_dict[i][0]], self.Points_done[self.virtual_stiffeners_dict[i][1]]))

            # add surfaces
            if self.warped_flag == 0:
                self.Surfaces_done = list()
                for i in self.geometry_data['Surfaces'].keys():               
                    edge_points = list()
                    for j in range(len(self.geometry_data['Surfaces'][i])):
                        edge_points.append(list(self.geometry_data['Points'][self.geometry_data['Surfaces'][i][j]]))
                    self.Surfaces_done.append(geom.add_polygon(edge_points))

            self.Curve_loop_ids = list()           
            if self.warped_flag == 1:
                self.Surfaces_done = list()
                self.Lines_ids = list()
                
                for i in self.geometry_data['Surfaces'].keys():               
                    edge_points = list()
                    for j in range(len(self.geometry_data['Surfaces'][i])):
                        edge_points.append(list(self.geometry_data['Points'][self.geometry_data['Surfaces'][i][j]]))

                    if len(edge_points)>4:
                        self.Surfaces_done.append(geom.add_polygon(edge_points))

                    if len(edge_points)<4:
                        self.Surfaces_done.append(geom.add_polygon(edge_points))

                    if len(edge_points)==4:
                        Lines_ids_tmp = list()
                        plane_coefficients = PlaneEquation(edge_points[0], edge_points[1], edge_points[2])
                        if OnPlane(edge_points[3], plane_coefficients) == True:
                            self.Surfaces_done.append(geom.add_polygon(edge_points))
                        else:
                            for j in range(len(self.geometry_data['Surfaces'][i])):    
                                if j == len(self.geometry_data['Surfaces'][i]) - 1:
                                    tmp_add_line_1 = geom.add_line(self.Points_done[self.geometry_data['Surfaces'][i][j]], self.Points_done[self.geometry_data['Surfaces'][i][0]])
                                    Lines_ids_tmp.append(tmp_add_line_1)
                                    self.Lines_ids.append(tmp_add_line_1)
                                else:
                                    tmp_add_line_2 = geom.add_line(self.Points_done[self.geometry_data['Surfaces'][i][j]], self.Points_done[self.geometry_data['Surfaces'][i][j+1]])
                                    Lines_ids_tmp.append(tmp_add_line_2)
                                    self.Lines_ids.append(tmp_add_line_2)

                            tmp_curve_loop = geom.add_curve_loop([Lines_ids_tmp[0], Lines_ids_tmp[1], Lines_ids_tmp[2], Lines_ids_tmp[3]])
                            self.Curve_loop_ids.append(tmp_curve_loop)

                            for k in self.old_warped_new_warped_curve_loop.keys():
                                if i in self.old_warped_new_warped[k]:
                                    self.old_warped_new_warped_curve_loop[k].append(tmp_curve_loop._id)
                                    break

            self.Points_done_not_on_any_entity = {key: self.Points_done[key]
                for key in self.geometry_data['PointsForResponse'].keys()}
           
            list_of_keys_of_used_points = copy.deepcopy(list(self.Points_done_not_on_any_entity.keys()))

            # force mesh to contain points for response as nodes
            if self.Points_done_not_on_any_entity and self.Surfaces_done:
                for i in self.Points_done_not_on_any_entity.keys():
                    for surface in self.Surfaces_done:
                        if OnPlane(self.Points_done_not_on_any_entity[i].x, PlaneEquation(surface.points[0].x,surface.points[1].x,surface.points[2].x)):
                            if len(surface.points) == 3:
                                if InSurfaceWith3Edges(self.Points_done_not_on_any_entity[i].x, surface.points[0].x,surface.points[1].x,surface.points[2].x):
                                    geom.boolean_fragments(surface, self.Points_done_not_on_any_entity[i], delete_first = True, delete_other = True)
                                    list_of_keys_of_used_points.remove(i)
                                    break
                            if len(surface.points) == 4:
                                if InSurfaceWith4Edges(self.Points_done_not_on_any_entity[i].x, surface.points[0].x,surface.points[1].x,surface.points[2].x,surface.points[3].x):
                                    geom.boolean_fragments(surface, self.Points_done_not_on_any_entity[i], delete_first = True, delete_other = True)
                                    list_of_keys_of_used_points.remove(i)
                                    break
            
            # only force mesh to contain points which are not already forced in surface loop
            if list_of_keys_of_used_points and self.Lines_on_2edged_surface_done:
                for i in list_of_keys_of_used_points:
                    for line in self.Lines_on_2edged_surface_done:
                        if IfPointIsInSegment(self.Points_done_not_on_any_entity[i].x[0], self.Points_done_not_on_any_entity[i].x[1], self.Points_done_not_on_any_entity[i].x[2],line.points[0].x[0], line.points[0].x[1], line.points[0].x[2], line.points[1].x[0], line.points[1].x[1], line.points[1].x[2]):
                            geom.boolean_fragments(line, self.Points_done_not_on_any_entity[i], delete_first = False, delete_other = False)
                            break
            
            # set minimal and maximal element edge length in mesh, i.e. mesh element size
            if self.geometry_data['DistancesForMeshSize']:
        
                geom.characteristic_length_min = 0.5*(min(self.geometry_data['DistancesForMeshSize']) + max(self.geometry_data['DistancesForMeshSize']))
                geom.characteristic_length_max = 1.5*geom.characteristic_length_min

            else:

                geom.characteristic_length_min = self.geometry_data["meshMinSize"]
                geom.characteristic_length_max = self.geometry_data["meshMaxSize"]

            self.geo_name = "GeometryBeforeBooleanOperations-" + self.input_data_name + ".geo_unrolled"
            geom.save_geometry(self.geo_name)

            self.start_time_101 = time.time()

            # reclassify geometry and force mesh to contain specified entities in mesh
            if len(self.Surfaces_done) > 1:
                if self.number_of_holes_counter == 0:
            
                    new_surfaces = geom.boolean_fragments(self.Surfaces_done[0], self.Surfaces_done[1:])

                    if len(self.Curve_loop_ids):
                        new_surfaces = geom.boolean_fragments(new_surfaces[0:], self.Lines_ids[0:])

                    if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) > 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0:])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0:])

                    if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0:])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0])
                    
                    if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 0:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0:])

                    if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) > 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0:])
                    
                    if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0])
                    
                    if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 0:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0])

                    if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) > 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Edges_done[0:])
                    
                    if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) == 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Edges_done[0])

                    gmsh.model.occ.removeAllDuplicates()

                else: 
                    if len(self.Surfaces_done) == (self.number_of_holes_counter + 1):
                        new_surfaces = geom.boolean_difference(self.Surfaces_done[0], self.Surfaces_done[len(self.Surfaces_done) - self.number_of_holes_counter:len(self.Surfaces_done)])
                    else:
                        new_surfaces = geom.boolean_difference(self.Surfaces_done[0:len(self.Surfaces_done) - self.number_of_holes_counter], self.Surfaces_done[len(self.Surfaces_done) - self.number_of_holes_counter:len(self.Surfaces_done)])

                    if len(self.Curve_loop_ids):
                        new_surfaces = geom.boolean_fragments(new_surfaces[0:], self.Lines_ids[0:])

                    if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) > 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0:])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0:])

                    if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0:])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0])
                    
                    if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 0:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0:])

                    if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) > 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0:])
                    
                    if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0])
                        new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0])
                    
                    if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 0:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Lines_on_2edged_surface_done[0])

                    if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) > 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Edges_done[0:])
                    
                    if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) == 1:
                        new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Edges_done[0])

                    gmsh.model.occ.removeAllDuplicates()
                    
            if len(self.Surfaces_done) == 1:

                if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) > 1:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0:])
                    new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0:])

                if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 1:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0:])
                    new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0])
                
                if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 0:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0:])

                if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) > 1:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0])
                    new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0:])
                
                if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 1:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0])
                    new_surfaces3 = geom.boolean_fragments(new_surfaces2[0:], self.Edges_done[0])
                
                if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 0:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0])

                if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) > 1:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Edges_done[0:])
                
                if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) == 1:
                    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Edges_done[0])

                gmsh.model.occ.removeAllDuplicates()

            if len(self.Surfaces_done) == 0:

                if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) > 1:
                    new_surfaces = geom.boolean_fragments(self.Lines_on_2edged_surface_done[0], self.Lines_on_2edged_surface_done[1:])
                    new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Edges_done[0:])

                if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 1:
                    new_surfaces = geom.boolean_fragments(self.Lines_on_2edged_surface_done[0], self.Lines_on_2edged_surface_done[1:])
                    new_surfaces2 = geom.boolean_fragments(new_surfaces[0:], self.Edges_done[0])
                
                if len(self.Lines_on_2edged_surface_done) > 1 and len(self.Edges_done) == 0:
                    new_surfaces = geom.boolean_fragments(self.Lines_on_2edged_surface_done[0], self.Lines_on_2edged_surface_done[1:])

                if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) > 1:
                    new_surfaces = geom.boolean_fragments(self.Lines_on_2edged_surface_done[0], self.Edges_done[0:])
                
                if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 1:
                    new_surfaces = geom.boolean_fragments(self.Lines_on_2edged_surface_done[0], self.Edges_done[0])
                
                #if len(self.Lines_on_2edged_surface_done) == 1 and len(self.Edges_done) == 0:
                #    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Lines_on_2edged_surface_done[0])

                #if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) > 1:
                #    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Edges_done[0:])
                
                #if len(self.Lines_on_2edged_surface_done) == 0 and len(self.Edges_done) == 1:
                #    new_surfaces2 = geom.boolean_fragments(self.Surfaces_done[0:], self.Edges_done[0])
                
                gmsh.model.occ.removeAllDuplicates()

            self.end_time_101 = time.time()
            
            # add warped surfaces
            if self.warped_flag == 1:
                for i in range(len(self.Curve_loop_ids)):               
                    tmp = gmsh.model.occ.addSurfaceFilling(self.Curve_loop_ids[i]._id)
                    self.Surfaces_done.append(tmp)                
                    for k in self.old_warped_new_warped_curve_loop.keys():
                        if self.Curve_loop_ids[i]._id in self.old_warped_new_warped_curve_loop[k]:
                            self.old_warped_new_warped_surface_ids[k].append(tmp)
                
            self.geo_name = "GeometryAfterBooleanOperations-" + self.input_data_name + ".geo_unrolled"
            geom.save_geometry(self.geo_name)
            
            self.start_time_102 = time.time()

            # meshing geometry, see geometry.py from pygmsh for more input info
            
            gmsh.option.setNumber("Mesh.Algorithm", 9)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
            self.mesh = geom.generate_mesh(dim=2, verbose = False)
            self.end_time_102 = time.time()
            self.msh_name = "NonRecombinedMesh-" + self.input_data_name + ".msh"
            pygmsh.write(self.msh_name)

            gmsh.model.mesh.removeDuplicateNodes()
            
            self.start_time_103 = time.time()
            gmsh.model.mesh.recombine()
            self.end_time_103 = time.time()
            self.msh_name = "RecombinedFinalMesh-" + self.input_data_name + ".msh"
            pygmsh.write(self.msh_name)

    def ComputeStatistics(self):
        """
        Reading data from Gmsh mesh file.
        """
        f = open(self.msh_name, "r")
        counter = 0
        tmp_string = ""

        mesh_data = {}
        nodes_data = {}
        triangle_data = {}
        quad_data = {}
        element_data = {}
        line_in_mesh_data = {}

        for x in f:
            counter = counter + 1
            x = x.rstrip()
            if x == "$Nodes":
                tmp_string = "$Nodes"
                counter_on_nodes = counter
            if x == "$Elements":
                tmp_string = "$Elements"
                counter_on_elements = counter
            if x == "$EndElements":
                break

            if tmp_string == "$Nodes":
                x = x.split()
                if len(x) == 4 and int(x[3]) != 0 and counter != counter_on_nodes + 1:
                    number_of_nodes = int(x[3])
                    tmp_list = list()
                    
                    for _ in range(2*number_of_nodes):
                        x = next(f)
                        x = x.rstrip()
                        x = x.split()
                        if len(x) == 1:
                            tmp_list.append(int(x[0]))
                        if len(x) == 3:
                            tmp_list.append(numpy.array([float(x[0]), float(x[1]), float(x[2])]))
                    
                    for i in range(number_of_nodes):
                        nodes_data[tmp_list[i]] = tmp_list[i + number_of_nodes]

            if tmp_string == "$Elements":
                x = x.split()
                if len(x) == 4 and int(x[0]) == 2 and counter != counter_on_elements + 1:
                    number_of_elements = int(x[3])
                    tmp_list = list()
                    surface_id = int(x[1])
                    for _ in range(number_of_elements):
                        x = next(f)
                        x = x.rstrip()
                        x = x.split()
                        if len(x) == 4:
                            surface_type = 'triangle'
                            key = int(x[0])
                            value = [int(x[1]), int(x[2]), int(x[3])]
                            triangle_data[key] = value
                            element_data[key] = {'nodesId': value, 'surfaceId': surface_id}
                        if len(x) == 5:
                            surface_type = 'quad'
                            key = int(x[0])
                            value = [int(x[1]), int(x[2]), int(x[3]), int(x[4])]
                            quad_data[key] = value
                            element_data[key] = {'nodesId': value, 'surfaceId': surface_id}

                if len(x) == 4 and int(x[0]) == 1 and counter != counter_on_elements + 1:
                    number_of_elements = int(x[3])
                    tmp_list = list()
                    line_id = int(x[1])
                    for _ in range(number_of_elements):
                        x = next(f)
                        x = x.rstrip()
                        x = x.split()
                        if len(x) == 3:
                            key = int(x[0])
                            value = [int(x[1]), int(x[2])]
                            line_in_mesh_data[key] = value

        length_for_stiffeners = list(element_data.keys())[-1]

        nodes_nmb = len(nodes_data)
        triangle_nmb = len(triangle_data)
        quad_nmb = len(quad_data)
        element_nmb = len(element_data)

        ##########################################################################
        ##########################################################################
        """
        Compute statistics about mesh and returns figure.
        """
        def calculate_area(distance1, distance2, distance3):
            """
            Calculate area of traingle using Heron's formula.
            """
            s = (distance1 + distance2 + distance3)/2
            return math.sqrt(s*(s - distance1)*(s - distance2)*(s - distance3))

        def angle_triangle(point1, point2, point3):  
            """
            Calculate angle(in point1) in a triangle defined with point1, point2, point3.  
            """
            x1 = point1[0]
            y1 = point1[1]
            z1 = point1[2]

            x2 = point2[0]
            y2 = point2[1]
            z2 = point2[2]

            x3 = point3[0]
            y3 = point3[1]
            z3 = point3[2]

            num = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1)  
        
            den = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)*math.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2)  
        
            angle = math.degrees(math.acos(num / den))  
        
            return round(angle, 3)

        triangle_ratio = {}
        quad_ratio = {}
        triangle_area = 0
        quad_area = 0

        # Calculate area of the model.

        for key in triangle_data.keys():
            point1 = nodes_data[triangle_data[key][0]]
            point2 = nodes_data[triangle_data[key][1]]
            point3 = nodes_data[triangle_data[key][2]]
            
            distances = list()
            
            distances.append(LA.norm(point1 - point2))
            distances.append(LA.norm(point2 - point3))
            distances.append(LA.norm(point3 - point1))

            triangle_ratio[key] = min(distances)/max(distances)
            triangle_area = triangle_area + calculate_area(distances[0],distances[1],distances[2])

        for key in quad_data.keys():
            point1 = nodes_data[quad_data[key][0]]
            point2 = nodes_data[quad_data[key][1]]
            point3 = nodes_data[quad_data[key][2]]
            point4 = nodes_data[quad_data[key][3]]
            
            distances = list()
            
            distances.append(LA.norm(point1 - point2))
            distances.append(LA.norm(point2 - point3))
            distances.append(LA.norm(point3 - point4))
            distances.append(LA.norm(point4 - point1))

            diag_dist = LA.norm(point1 - point3)

            quad_area = quad_area + calculate_area(distances[0],distances[1], diag_dist) + calculate_area(diag_dist, distances[2], distances[3]) 
            quad_ratio[key] = min(distances)/max(distances)

        geometry_area = triangle_area + quad_area

        ##########################################################################

        # Calculate number of traingles and quads with bad ratio between sides.
        bad_traingles_counter = 0
        bad_triangles_id = {}
        for key in triangle_ratio.keys():
            if triangle_ratio[key] < 0.2:
                bad_traingles_counter = bad_traingles_counter + 1
                bad_triangles_id[key] = triangle_ratio[key]

        mesh_data["%T with ratio < 0.2"] = round(100*bad_traingles_counter/len(element_data),2)

        bad_quad_counter = 0
        bad_quad_id = {}
        for key in quad_ratio.keys():
            if quad_ratio[key] < 0.3334:
                bad_quad_counter = bad_quad_counter + 1
                bad_quad_id[key] = quad_ratio[key]

        mesh_data["%Q with ratio < 0.334"] = round(100*bad_quad_counter/len(element_data),2)

        ##########################################################################

        # Calculate number of quads with all angles between 80 and 100 degrees. 
        # Calculate percentage of the area quads with all angles between 80 and 100 degrees are covering
        # Calculate percentage of triangle area.
        lower_thr = 80  
        upper_thr = 100

        quad_angles = {}
        bad_quad_angle_id = {}
        bad_quad_angles_counter = 0
        good_quad_area = 0
        for key in quad_data.keys():
            point1 = nodes_data[quad_data[key][0]]
            point2 = nodes_data[quad_data[key][1]]
            point3 = nodes_data[quad_data[key][2]]
            point4 = nodes_data[quad_data[key][3]]

            angle1 = angle_triangle(point1, point4, point2)
            tmp_angle2 = angle_triangle(point2, point1, point4)
            tmp_angle3 = angle_triangle(point4, point2, point1)

            tmp_angle4 = angle_triangle(point2, point4, point3)
            tmp_angle5 = angle_triangle(point4, point3, point2)
            angle3 = angle_triangle(point3, point2, point4)

            angle2 = tmp_angle2 + tmp_angle4
            angle4 = tmp_angle3 + tmp_angle5

            quad_angles[key] = [angle1, angle2, angle3, angle4]

            if angle1 > lower_thr and angle2 > lower_thr and angle3 > lower_thr and angle4 > lower_thr  and angle1 < upper_thr and angle2 < upper_thr and angle3 < upper_thr and angle4 < upper_thr:
                bad_quad_angle_id[key] = quad_angles[key]
                bad_quad_angles_counter = bad_quad_angles_counter + 1

                distances = list()

                distances.append(LA.norm(point1 - point2))
                distances.append(LA.norm(point2 - point3))
                distances.append(LA.norm(point3 - point4))
                distances.append(LA.norm(point4 - point1))

                diag_dist = LA.norm(point1 - point3)

                good_quad_area = good_quad_area + calculate_area(distances[0],distances[1], diag_dist) + calculate_area(diag_dist, distances[2], distances[3]) 

        mesh_data["%Q > 80 and < 100"] = round(100*bad_quad_angles_counter/len(element_data),2)
        mesh_data["% Area 80<Q<100"] = round(100*good_quad_area/geometry_area,2)
        mesh_data["% Area T"] = round(100*triangle_area/geometry_area,2)
        good_quads = bad_quad_angles_counter

        # Calculate number of quads that have at least one angle lower then 45 degrees or larger then 100 degrees. 
        lower_thr = 45  
        upper_thr = 135

        quad_angles = {}
        bad_quad_angle_id = {}
        bad_quad_angles_counter = 0

        for key in quad_data.keys():
            point1 = nodes_data[quad_data[key][0]]
            point2 = nodes_data[quad_data[key][1]]
            point3 = nodes_data[quad_data[key][2]]
            point4 = nodes_data[quad_data[key][3]]

            angle1 = angle_triangle(point1, point4, point2)
            tmp_angle2 = angle_triangle(point2, point1, point4)
            tmp_angle3 = angle_triangle(point4, point2, point1)

            tmp_angle4 = angle_triangle(point2, point4, point3)
            tmp_angle5 = angle_triangle(point4, point3, point2)
            angle3 = angle_triangle(point3, point2, point4)

            angle2 = tmp_angle2 + tmp_angle4
            angle4 = tmp_angle3 + tmp_angle5

            quad_angles[key] = [angle1, angle2, angle3, angle4]

            if angle1 < lower_thr or angle2 < lower_thr or angle3 < lower_thr or angle4 < lower_thr  or angle1 > upper_thr or angle2 > upper_thr or angle3 > upper_thr or angle4 > upper_thr:
                bad_quad_angle_id[key] = quad_angles[key]
                bad_quad_angles_counter = bad_quad_angles_counter + 1

        mesh_data["%Q < 45 or > 135"] = round(100*bad_quad_angles_counter/len(element_data),2)

        # Calculate number of triangles that have at least one angle lower then 45 degrees or larger then 100 degrees. 
        triangle_angles = {}
        bad_triangle_angle_id = {}
        bad_triangle_angles_counter = 0

        for key in triangle_data.keys():
            point1 = nodes_data[triangle_data[key][0]]
            point2 = nodes_data[triangle_data[key][1]]
            point3 = nodes_data[triangle_data[key][2]]

            angle1 = angle_triangle(point1, point2, point3)
            angle2 = angle_triangle(point2, point3, point1)
            angle3 = angle_triangle(point3, point1, point2)
            triangle_angles[key] = [angle1, angle2, angle3]

            if angle1 < lower_thr or angle2 < lower_thr or angle3 < lower_thr  or angle1 > upper_thr or angle2 > upper_thr or angle3 > upper_thr :
                bad_triangle_angles_counter = bad_triangle_angles_counter + 1
                bad_triangle_angle_id[key] = triangle_angles[key]

        mesh_data["%T < 45 or > 135"] = round(100*bad_triangle_angles_counter/len(element_data),2)

        ##########################################################################

        # Calculate number of quads that have at least one angle lower then 30 degrees or larger then 150 degrees. 
        lower_thr = 30  
        upper_thr = 150

        quad_angles = {}
        bad_quad_angle_id = {}
        bad_quad_angles_counter = 0

        for key in quad_data.keys():
            point1 = nodes_data[quad_data[key][0]]
            point2 = nodes_data[quad_data[key][1]]
            point3 = nodes_data[quad_data[key][2]]
            point4 = nodes_data[quad_data[key][3]]

            angle1 = angle_triangle(point1, point4, point2)
            tmp_angle2 = angle_triangle(point2, point1, point4)
            tmp_angle3 = angle_triangle(point4, point2, point1)

            tmp_angle4 = angle_triangle(point2, point4, point3)
            tmp_angle5 = angle_triangle(point4, point3, point2)
            angle3 = angle_triangle(point3, point2, point4)

            angle2 = tmp_angle2 + tmp_angle4
            angle4 = tmp_angle3 + tmp_angle5

            quad_angles[key] = [angle1, angle2, angle3, angle4]

            if angle1 < lower_thr or angle2 < lower_thr or angle3 < lower_thr or angle4 < lower_thr  or angle1 > upper_thr or angle2 > upper_thr or angle3 > upper_thr or angle4 > upper_thr:
                bad_quad_angle_id[key] = quad_angles[key]
                bad_quad_angles_counter = bad_quad_angles_counter + 1

        mesh_data["%Q < 30 or > 150"] = round(100*bad_quad_angles_counter/len(element_data),2)

        # Calculate number of triangles that have at least one angle lower then 30 degrees or larger then 150 degrees. 
        triangle_angles = {}
        bad_triangle_angle_id = {}
        bad_triangle_angles_counter = 0

        for key in triangle_data.keys():
            point1 = nodes_data[triangle_data[key][0]]
            point2 = nodes_data[triangle_data[key][1]]
            point3 = nodes_data[triangle_data[key][2]]

            angle1 = angle_triangle(point1, point2, point3)
            angle2 = angle_triangle(point2, point3, point1)
            angle3 = angle_triangle(point3, point1, point2)
            triangle_angles[key] = [angle1, angle2, angle3]

            if angle1 < lower_thr or angle2 < lower_thr or angle3 < lower_thr  or angle1 > upper_thr or angle2 > upper_thr or angle3 > upper_thr :
                bad_triangle_angles_counter = bad_triangle_angles_counter + 1
                bad_triangle_angle_id[key] = triangle_angles[key]

        mesh_data["%T < 30 or > 150"] = round(100*bad_triangle_angles_counter/len(element_data),2)

        ##########################################################################

        # Calculate number of quads that have at least one angle lower then 10 degrees or larger then 170 degrees. 
        lower_thr = 10  
        upper_thr = 170

        quad_angles = {}
        bad_quad_angle_id = {}
        bad_quad_angles_counter = 0

        for key in quad_data.keys():
            point1 = nodes_data[quad_data[key][0]]
            point2 = nodes_data[quad_data[key][1]]
            point3 = nodes_data[quad_data[key][2]]
            point4 = nodes_data[quad_data[key][3]]

            angle1 = angle_triangle(point1, point4, point2)
            tmp_angle2 = angle_triangle(point2, point1, point4)
            tmp_angle3 = angle_triangle(point4, point2, point1)

            tmp_angle4 = angle_triangle(point2, point4, point3)
            tmp_angle5 = angle_triangle(point4, point3, point2)
            angle3 = angle_triangle(point3, point2, point4)

            angle2 = tmp_angle2 + tmp_angle4
            angle4 = tmp_angle3 + tmp_angle5

            quad_angles[key] = [angle1, angle2, angle3, angle4]

            if angle1 < lower_thr or angle2 < lower_thr or angle3 < lower_thr or angle4 < lower_thr  or angle1 > upper_thr or angle2 > upper_thr or angle3 > upper_thr or angle4 > upper_thr:
                bad_quad_angle_id[key] = quad_angles[key]
                bad_quad_angles_counter = bad_quad_angles_counter + 1

        mesh_data["%Q < 10 or > 170"] = round(100*bad_quad_angles_counter/len(element_data),2)

        # Calculate number of traingles that have at least one angle lower then 10 degrees or larger then 170 degrees. 
        triangle_angles = {}
        bad_triangle_angle_id = {}
        bad_triangle_angles_counter = 0

        for key in triangle_data.keys():
            point1 = nodes_data[triangle_data[key][0]]
            point2 = nodes_data[triangle_data[key][1]]
            point3 = nodes_data[triangle_data[key][2]]

            angle1 = angle_triangle(point1, point2, point3)
            angle2 = angle_triangle(point2, point3, point1)
            angle3 = angle_triangle(point3, point1, point2)
            triangle_angles[key] = [angle1, angle2, angle3]

            if angle1 < lower_thr or angle2 < lower_thr or angle3 < lower_thr  or angle1 > upper_thr or angle2 > upper_thr or angle3 > upper_thr :
                bad_triangle_angles_counter = bad_triangle_angles_counter + 1
                bad_triangle_angle_id[key] = triangle_angles[key]

        mesh_data["%T < 10 or > 170"] = round(100*bad_triangle_angles_counter/len(element_data),2)

        ##########################################################################

        bad_elements_id = list()
        for key in bad_triangle_angle_id.keys():
            bad_elements_id.append(key)
        for key in bad_quad_angle_id.keys():
            bad_elements_id.append(key)
        for key in bad_quad_id.keys():
            bad_elements_id.append(key)
        for key in bad_triangles_id.keys():
            bad_elements_id.append(key)

        bad_elements_id = list(set(bad_elements_id))
        bad_elements_id.sort()

        ##########################################################################
        """
        Plotting mesh data
        """
        width = 1.0
        mesh_data_Length = len(mesh_data)
        Max_Key_Length = 20
        Sorted_Dict_Values = sorted(mesh_data.values(), reverse=True)
        Sorted_Dict_Keys = sorted(mesh_data, key=mesh_data.get, reverse=True)
        for i in range(0,mesh_data_Length):
            Key = Sorted_Dict_Keys[i]
            Key = Key[:Max_Key_Length]
            Sorted_Dict_Keys[i] = Key
        X = numpy.arange(mesh_data_Length)
        Colors = ('b','g','r','c')  # blue, green, red, cyan

        Figure = plt.figure(figsize = (20,10))
        Axis = Figure.add_subplot(1,1,1)

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                Axis.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%f' % float(height),
                        ha='center', va='bottom')

        for i in range(0,mesh_data_Length):
            rects1 = Axis.bar(X[i], Sorted_Dict_Values[i], align='center',width=0.5, color=Colors[i%len(Colors)])
            autolabel(rects1)

        Axis.set_xticks(X)
        xtickNames = Axis.set_xticklabels(Sorted_Dict_Keys)
        plt.setp(Sorted_Dict_Keys)
        plt.xticks(rotation=40)
        ymax = max(Sorted_Dict_Values) + 1
        plt.ylim(0,ymax+20)
        plt.title(self.input_data_name + "\n" + "Nodes:" +  str(nodes_nmb) + ", Elements:" + str(element_nmb) + ", Percentage of traingles:" + str(round(100*triangle_nmb/element_nmb,2)) + "%" + ", Percentage of Quads:" + str(round(100*quad_nmb/element_nmb,2)) + "%" + ", Percentage of quads with angles between 80 and 100:" + str(round(100*good_quads/element_nmb,2)) + "%")
        title_str = 'MeshStatistics-' + self.input_data_name + '.png'
        plt.savefig(title_str)

    def ComputeTimeStatistcs(self):
        """
        Computes time of algorithm phases and returns figure.
        """
        times = list()
        times.append(1000*(self.end_time_101 - self.start_time_101))
        times.append(1000*(self.end_time_102 - self.start_time_102))
        times.append(1000*(self.end_time_103 - self.start_time_103))

        colors = ['#8390FA', '#6EAF46', '#FAC748']
        labels = ['Boolean','Triangulation','Recombination']
        fig, ax = plt.subplots(1, figsize=(7, 6))
        left = 0
        for idx, time in enumerate(times):
            plt.barh(0, time, left = left, color=colors[idx])
            left = left + time
        # title, legend, labels etc.
        plt.title('Time statistics for algorithm phases\n', loc='center', fontsize = 14, y = 1.05)
        plt.legend(labels, bbox_to_anchor=([0.05, 1.1, 0, 0]), ncol=3, frameon=False, loc = 'upper left', fontsize = 12)
        plt.xlabel('Times in ms', fontsize = 14, y = -0.1)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.set_yticks([])
        plt.xticks(fontsize = 14)
        ax.set_axisbelow(True)
        plt.ylim([-0.8, 0.8])
        plt.ylabel(self.input_data_name, fontsize = 14, x = -0.1)
        ax.xaxis.grid(color='gray', linestyle='dashed')
        figure_name_string = 'TimeStatistics-' + self.input_data_name + '.png'
        plt.savefig(figure_name_string)

    def MakeDict(self):
        """
        Calls ConstructGeometry, ComputeStatistic and ComputeTimeStatistcs.
        """
        self.ChangeDictionaryForWarpedSurfaces()
        self.ConstructGeometry()
        self.ComputeStatistics()
        self.ComputeTimeStatistcs()
