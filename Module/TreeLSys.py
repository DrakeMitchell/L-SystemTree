import numpy as np
import math
import bpy
import Module

class Tree:
    class Node:
        theta = math.pi/6
        rot_matrix = {
            '+': np.matrix([[math.cos(theta), -math.sin(theta), 0],
                            [math.sin(theta), math.cos(theta),  0],
                            [0,               0,                1]]),
            '-': np.matrix([[math.cos(-theta), -math.sin(-theta), 0],
                            [math.sin(-theta), math.cos(-theta),  0],
                            [0,               0,                1]]),
            '*': np.matrix([[math.cos(theta),  0, math.sin(theta)],
                            [0,                 1,               0],
                            [-math.sin(theta), 0, math.cos(theta)]]),
            '/': np.matrix([[math.cos(-theta),  0, math.sin(-theta)],
                            [0,                  1,                0],
                            [-math.sin(-theta), 0, math.cos(-theta)]]),
            '^': np.matrix([[1,             0,                  0],
                            [0, math.cos(theta), -math.sin(theta)],
                            [0, math.sin(theta),  math.cos(theta)]]),
            '!': np.matrix([[1,              0,                   0],
                            [0, math.cos(-theta), -math.sin(-theta)],
                            [0, math.sin(-theta),  math.cos(-theta)]])
        }

        def rotate(self, terminal): self.rot = self.rot@self.rot_matrix[terminal]
        def has_next(self): return next is not None or len(branches) != 0

        def __init__(self, loc = tuple((0, 0, 0)), vec = np.matrix([[0],[0],[1]]), parent = None, k=1):
            self.loc = loc
            self.vec = vec
            self.parent = parent
            self.next = None
            self.branches = []
            self.rot = np.identity(3)
            self.out_verts = {-1: [], -2: []}
            self.k = k
        
        def __getitem__(self, idx):
            if idx == -1: return self.next
            if idx == -2: return self.parent
            return self.branches[idx]

        def __setitem__(self, idx, node):
            if idx == -1: self.next = node
            if idx == -2: self.parent = node
            self.branches[idx] = node

        def height(self):
            #if self.height: return self.height
            if self.next is None:
                if len(self.branches) == 0: 
           #         self.height = 1
                    return 1
            #    self.height = 1 + sum(branch.height() for branch in self.branches) 
                return 1 + sum(branch.height() for branch in self.branches)
            #self.height = 1 + self.next.height() + sum(branch.height() for branch in self.branches)            
            return 1 + self.next.height() + sum(branch.height() for branch in self.branches)            

        def extend(self, isBranching = False):
            temp = Tree.Node(tuple(float(self.loc[x] + (self.k*self.rot@self.vec)[x, 0]) for x in range(3)), self.rot@self.vec, self)
            if (isBranching):
                self.branches.append(temp)
                return 
            self.next = temp
        
        def mesh(self, dtheta:float =.1, k=.005):
            vec_sub = lambda a, b, c=1: tuple(c*(a[x] - b[x]) for x in range(len(a)))
            vec_add = lambda a, b, c=1: tuple(c*(a[x] + b[x]) for x in range(len(a)))

            def find_points(node, theta):
                points = []
                r = lambda r: math.sqrt(k * r.height()/math.pi)   
                f = lambda x, rf: h(x, rf, 2) if node.height() < rf.height() else g(x, rf, 2)     
                g = lambda x, rf, c: r(rf) + (self.k-x)/self.k*(r(node)-r(rf) if math.fabs(rf.height()-node.height()) < c else math.sqrt(k*3/math.pi))       
                h = lambda x, rf, c: r(node) + x/self.k*(r(rf)-r(node) if math.fabs(rf.height()-node.height()) < c else math.sqrt(k*3/math.pi))
                
                if node.next is not None:
                    temp = self.find_intersection(vec_sub(node.next.loc, node.loc, 30*k), f(30*k, node.next), theta)
                    if temp is not None:
                        node.out_verts[-1].append(temp)
                        points.append([temp1 + (-1,) for temp1 in temp])

                for x in range(len(node.branches)):
                    temp = self.find_intersection(vec_sub(node.branches[x].loc, node.loc, 30*k), f(30*k, node.branches[x]), theta)
                    if temp is not None:
                        try:
                            node.out_verts[x].append(temp)
                        except:
                            node.out_verts[x] = [temp]
                        points.append([temp1+(x,) for temp1 in temp])

                if node.parent is not None:
                    temp = self.find_intersection(vec_sub(node.parent.loc, node.loc, 30*k), f(30*k, node.parent), theta)
                    if temp is not None:
                        node.out_verts[-2].append(temp)
                        points.append([temp1+(-2,) for temp1 in temp])
                return points

            def angle(a, b, pos=True):
                theta = math.acos(sum(a[x]*b[x] for x in range(2))/math.sqrt(sum(a[x]**2 for x in range(2))*sum(b[x]**2 for x in range(2))))
                if not pos and theta > 0:
                    theta -= 2*math.pi
                if pos and theta < 0:
                    theta += 2*math.pi
                return theta
            
            def rot(v, theta): # y-z plane -> positional space
                return tuple((v[0]*math.sin(theta), v[0]*math.cos(theta), v[1]))
            
            def proj(v, theta):
                u = tuple((math.cos(theta), math.sin(theta), 0))
                return tuple((sum(v[x]*u[x] for x in range(len(v))), v[2]))

            def find_radial(v, theta, dt=.05):
                r = lambda r: math.sqrt(k * r.height()/math.pi)   
                f = lambda x, rf: h(x, rf, 2) if self.height() < rf.height() else g(x, rf, 2)     
                g = lambda x, rf, c: r(rf) + (self.k-x)/self.k*(r(self)-r(rf) if math.fabs(rf.height()-self.height()) < c else math.sqrt(k*3/math.pi))       
                h = lambda x, rf, c: r(self) + x/self.k*(r(rf)-r(self) if math.fabs(rf.height()-self.height()) < c else math.sqrt(k*3/math.pi))
                
                mag = lambda v: math.sqrt(sum(v1**2 for v1 in v))

                radial_prime = vec_sub(self[v[2]].loc, self.loc)
                temp = rot(v, theta)
                temp = tuple((f(self.k, self[v[2]])/mag(temp)-1)*temp1 for temp1 in temp) # mag(vec_sub(self[v[2]].loc, self.loc)) -> self.k
                radial = proj(vec_add(radial_prime, temp), theta)
                return tuple(rad*dt/mag(radial) for rad in radial)
            
            v = []
            for theta in range(math.floor(math.pi/dtheta)):
                points = find_points(self, dtheta*theta)
                #[[a, b], [c, d], [e, f]]
                for point in points:
                    for x in range(2):
                        try:
                            neighbor = min([y[i] for y in points if y is not point for i in range(2)], key=lambda k: angle(k,point[x],x==1))
                            # point[x], point[x] + radial vec, neighbor, neighbor + radial vec
                            # print(point[x][2])
                            radial1 = vec_add(find_radial(point[x], theta*dtheta), point[x])
                            radial2 = vec_add(find_radial(neighbor, theta*dtheta), neighbor)
                            
                            v.append(rot(radial1, theta*dtheta))
                            v.append(rot(radial1, theta*dtheta))
                            v.append(rot(point[x], theta*dtheta))
                            v.append(rot(neighbor, theta*dtheta))
                        except Exception as e:
                            v.append(rot(point[x], theta*dtheta))
            if v:
                try:
                    node_data = bpy.data.meshes.new("node_data")
                    node_data.from_pydata(v, [], [])
                    node_mesh = bpy.data.objects.new("node", node_data)
                    node_mesh.location = self.loc
                    bpy.context.collection.objects.link(node_mesh)
                except Exception as e:
                    print(e) # test
                    
        @staticmethod
        def connect_parametric(v):
            theta = math.acos(sum(math.prod(v[i][x] for i in range(4)) for x in range(3))/math.prod(math.sqrt(sum(v[i][x]**2 for x in range(3))) for i in range(4)))
            def sub_ellipse(positions):
                x1, y1 = positions[0][0], positions[0][1]
                x2, y2 = positions[1][0], positions[1][1]
                x3, y3 = positions[2][0], positions[2][1]
                x4, y4 = positions[3][0], positions[3][1]
                
                # Constants for ellipse calculation
                i = y3**2 * (x1**2 - x2**2) + (x3**2 - x1**2) * y2**2 - (x3**2 - x2**2) * y1**2
                j = y1**2 * (x3 - x2) - y2**2 * (x3 - x1) + y3**2 * (x2 - x1)
                k = (y3 - y2) * (x4**2 - x3**2) + (y4 - y3) * (x2**2 - x3**2)
                l = (y4 - y3) * (x3 - x2) - (y3 - y2) * (x4 - x3)
                d = (y4**2 - y3**2) * (x2**2 - x3**2) - (y2**2 - y3**2) * (x4**2 - x3**2)
                f = (x3 - x2) * (y4**2 - y3**2) + (x4 - x3) * (y2**2 - y3**2)
                g = (x3**2 - x2**2) * (y2 - y1) + (y3 - y2) * (x1**2 - x2**2)
                h = (y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2)
                
                # Solving for c1 and c2
                c1 =  math.sqrt(((f*g + d*h - j*k - l*i)**2 / (4 * (f*h - l*j)) + i*k - g*d) / (4 * (f*h - l*j))) - (f*g + d*h - j*k - l*i) / (4 * (f*h - l*j))
                c2 = -math.sqrt(((f*g + d*h - j*k - l*i)**2 / (4 * (f*h - l*j)) + i*k - g*d) / (4 * (f*h - l*j))) - (f*g + d*h - j*k - l*i) / (4 * (f*h - l*j))
                
                # Calculate b(c)
                def b(c):
                    return (y3**2 * (x1**2 - x2**2 + 2 * c * (x2 - x1)) + y2**2 * ((x3 - c)**2 - (x2 - c)**2 - (x1**2 - x2**2 + 2 * c * (x2 - x1))) - y1**2 * (x3**2 - x2**2 - 2 * c * (x3 - x2))) / (2 * ((x3**2 - x2**2 - 2 * c * (x3 - x2)) * (y2 - y1) + (y3 - y2) * (x1**2 - x2**2 + 2 * c * (x2 - x1))))
                
                b1 = b(c1)
                b2 = b(c2)
                
                # Determine valid ellipse
                def is_valid_ellipse(c, b):
                    return all(abs(((y - b)**2 - b**2 + 2*b*y3 - y3**2) / (y2**2 - y3**2 + 2*b*(y3 - y2)) - ((x - c)**2 - c**2 + 2*c*x3 - x3**2) / (x2**2 - x3**2 + 2*c*(x3 - x2))) < 0.05 for x, y in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
                
                b = b1 if is_valid_ellipse(c1, b1) else b2
                c = c1 if is_valid_ellipse(c1, b1) else c2
                
        @staticmethod
        def find_intersection(v, r, theta):
            theta1 = math.atan2(v[2],math.sqrt(v[0]**2+v[1]**2))
            theta2 = theta + math.atan2(v[1], v[0]) #+ (0 if v[0] >= 0 else math.pi)
        
            def mag(v): return math.sqrt(sum(x**2 for x in v))
        
            i = math.tan(theta2)**2 - mag(v)**2/r**2 * math.cos(theta1)**2
            i = i / math.sin(math.copysign(1e-5, theta1) if math.fabs(theta1) < 1e-5 else theta1)**2 + 1
            if (i <= 0):
                return None

            t = [(math.pi/2-math.atan2(math.tan(theta2), math.sin(theta1))) - t1*(math.pi/2 - math.atan2(mag(v), math.tan(theta1)*r*math.sqrt(i))) for t1 in [-1, 1]]

            f = lambda x: mag(v) * math.cos(theta1) - r*math.sin(x)*math.sin(theta1)
            g = lambda x: mag(v) * math.sin(theta1) + r*math.sin(x)*math.cos(theta1)
            
            # [-1 , 1]
            return [(f(t1)*math.sin(theta2)+r*math.cos(t1)*math.cos(theta2), g(t1)) for t1 in t]

    class iterator:
        def __init__(self, header):
            self.header = header
    
        def next(self, isBranching = False, n=-1):
            self.header = self.header.next if not isBranching else self.header.branches[n]
        def previous(self, outBranch = False):
            while (outBranch and self.header is self.header.parent.next):
                self.header = self.header.parent
            self.header = self.header.parent

        def extend(self, isBranching=False):
            self.header.extend(isBranching)
            self.next(isBranching)
        
        def mesh(self):
            v = []
            e = [] #(a, b)

            def sub_mesh(node=self.header):
                n = len(v)
                v.append(node.loc)
                node.mesh()#add_circle_at_node(node)
                if node.next is not None:
                    e.append((n, sub_mesh(node.next)))
                for branch in node.branches:
                    e.append((n, sub_mesh(branch)))                
                return n
            sub_mesh()
            
            tree_data = bpy.data.meshes.new("tree_data")
            tree_data.from_pydata(v, e, [])
    
            tree_object = bpy.data.objects.new("tree", tree_data)
            bpy.context.collection.objects.link(tree_object)
            
    def __init__(self, string=""):
        self.root = Tree.Node()
        self.tree_cursor = Tree.iterator(self.root)
        self.load(string)
        self.mesh()
    
    def load(self, string):    
        char_map = {'F': lambda: self.tree_cursor.extend(),
                    '[': lambda: self.tree_cursor.extend(True),
                    ']': lambda: self.tree_cursor.previous(True),
                    '+': lambda: self.tree_cursor.header.rotate('+'),
                    '-': lambda: self.tree_cursor.header.rotate('-'),
                    '*': lambda: self.tree_cursor.header.rotate('*'),
                    '/': lambda: self.tree_cursor.header.rotate('/'),
                    '^': lambda: self.tree_cursor.header.rotate('^'),
                    '!': lambda: self.tree_cursor.header.rotate('!')}
        
        for s in string:
            char_map.get(s)()

    def mesh(self):
        Tree.iterator(self.root).mesh()