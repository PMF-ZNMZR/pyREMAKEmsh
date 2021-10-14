import numpy
import math
import pygmsh
import time
import gmsh
import copy

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
# It is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
# or FITNESS FOR A PARTICULAR PURPOSE. 
###################################################################################################

class pyREMAKEmsh:
    
    def __init__(self, geometry_data, tol):
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

    def ConstructGeometry(self):
        """
        Making geometry and mesh
        """
        with pygmsh.occ.Geometry() as geom:
            
            
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
            for i in self.virtual_stiffeners_dict.keys():
                self.Edges_done.append(geom.add_line(self.Points_done[self.virtual_stiffeners_dict[i][0]], self.Points_done[self.virtual_stiffeners_dict[i][1]]))

            # add surfaces
            self.Surfaces_done = list()
            for i in self.geometry_data['Surfaces'].keys():               
                edge_points = list()
                for j in range(len(self.geometry_data['Surfaces'][i])):
                    edge_points.append(list(self.geometry_data['Points'][self.geometry_data['Surfaces'][i][j]]))
                self.Surfaces_done.append(geom.add_polygon(edge_points))
            
            self.Points_done_not_on_any_entity = {key: self.Points_done[key]
                for key in self.geometry_data['PointsForResponse'].keys()}

            def PlaneEquation(x, y, z):
                """
                Finds coefficients defining the plane.
                """
                a = (y[1] - x[1])*(z[2] - x[2]) - (z[1] - x[1])*(y[2] - x[2])
                b = (y[2] - x[2])*(z[0] - x[0]) - (z[2] - x[2])*(y[0] - x[0])
                c = (y[0] - x[0])*(z[1] - x[1]) - (z[0] - x[0])*(y[1] - x[1])
                d = -(a*x[0] + b*x[1] + c*x[2])
                arr = [a, b, c, d]
                return arr

            def OnPlane(P, plane_coefficients):
                """
                Checks if point P lies on plane defined with plane_coefficients.
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
            
            # set minimal and maximal element edge length
            if self.geometry_data['DistancesForMeshSize']:
        
                geom.characteristic_length_min = 0.5*(min(self.geometry_data['DistancesForMeshSize']) + max(self.geometry_data['DistancesForMeshSize']))
                geom.characteristic_length_max = 1.5*geom.characteristic_length_min

            else:

                geom.characteristic_length_min = self.geometry_data["meshMinSize"]
                geom.characteristic_length_max = self.geometry_data["meshMaxSize"]

            self.geo_name = "GeometryBeforeBooleanOperations.geo_unrolled"
            geom.save_geometry(self.geo_name)

            self.start_time_101 = time.time()

            # reclassify geometry and force mesh to contain specified entities in mesh
            if len(self.Surfaces_done) > 1:
                if self.number_of_holes_counter == 0:
            
                    new_surfaces = geom.boolean_fragments(self.Surfaces_done[0], self.Surfaces_done[1:])

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

                    geom.remove_all_duplicates()

                else: 
                    if len(self.Surfaces_done) == (self.number_of_holes_counter + 1):
                        new_surfaces = geom.boolean_difference(self.Surfaces_done[0], self.Surfaces_done[len(self.Surfaces_done) - self.number_of_holes_counter:len(self.Surfaces_done)])
                    else:
                        new_surfaces = geom.boolean_difference(self.Surfaces_done[0:len(self.Surfaces_done) - self.number_of_holes_counter], self.Surfaces_done[len(self.Surfaces_done) - self.number_of_holes_counter:len(self.Surfaces_done)])

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

                    geom.remove_all_duplicates()
                    
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

                geom.remove_all_duplicates()

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
                
                geom.remove_all_duplicates()

            self.end_time_101 = time.time()

            self.geo_name = "GeometryAfterBooleanOperations.geo_unrolled"
            geom.save_geometry(self.geo_name)
            
            self.start_time_102 = time.time()

            # meshing geometry, see geometry.py from pygmsh for more input info
            #  
            self.mesh = geom.generate_mesh(dim=2, verbose = False)
            self.end_time_102 = time.time()
            self.msh_name = "NonRecombinedMesh.msh"
            pygmsh.write(self.msh_name)

            gmsh.model.mesh.removeDuplicateNodes()
            self.start_time_103 = time.time()
            gmsh.model.mesh.recombine()
            self.end_time_103 = time.time()
            self.msh_name = "RecombinedFinalMesh.msh"
            pygmsh.write(self.msh_name)

    def MakeDict(self):
        """
        Calls ConstructGeometry and prints time statistics.
        """
        self.ConstructGeometry()
        
        print("\nStatistics summary\n")
        print("Boolean operations --- %s seconds ---" % (self.end_time_101 - self.start_time_101))
        print("Triangulation --- %s seconds ---" % (self.end_time_102 - self.start_time_102))
        print("Recombination --- %s seconds ---" % (self.end_time_103 - self.start_time_103))

