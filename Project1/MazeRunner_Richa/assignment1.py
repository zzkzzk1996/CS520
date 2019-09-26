# Import any libraries we might use!
import numpy as np    # General numerical array manipulation framework
from PIL import Image # Used to store mage as a nice image
import queue        # used for FiFO and LiFo used for BFS and DFS respectively
import heapq        # used for A* priority queue
import argparse     # Used to parse console arguments
import matplotlib.pyplot as plt
from matplotlib import colors
import time

standardDir = np.array([[-1,0], [0,-1], [0,1], [1,0]])
# ======================================================================================
# FUNCTIONS
# **********

def create_maze(dim, p =.005):

    # Classic double for way
    maze= np.empty((dim,dim))
    for i in range(dim):
        for j in range(dim):
            chance = np.random.random()
            if chance < p:
                maze[i,j] = 1
            else:
                maze[i,j] = 0
    # More pythonic way.
    # make a random choice of numbers 0, 1, enough to fill an array of size (dim,dim).
    # 0 and 1  have probabilities 1-p, p to occur, respectively. More compact way where
    # we make use of numpy library, also much faster!!!
    maze = np.random.choice([0,1], (dim,dim), p=[ 1-p, p])
    # Ensure  start and points are free!
    maze[0,0], maze[-1,-1] = 0, 0
    # print(maze)
    # return the value
    return maze

# --------------------------------------------------------

def draw_maze(maze, path = None, label =''):
    """ DESCRITPTION: This function draws the maze to an image format for visualization purposes.
                      It gets the dimensions of the maze, and draws an image where each cell of the
                      originnal maze is displayed as a tileSize x tileSize block in the image.
                      Empty celss (maze value =0) are displayed as white, occupied cells (amze value=1)
                      as black and path tiles as gray,
                
        ARGUMENTS: maze-> (numpy array) A table-like represenation of the maze, where 1's represents
                          occupied space
        RETURNS: image-> () Image representation of maze
                 image-> () Image representation of traversal attempt of maze
    """
    tileSize = 16 # how large a tile is displayed on the vusalization image. reduce this to see smaller tiles!
    dim = maze.shape[0]
    image = np.empty((dim*tileSize,dim*tileSize), dtype=np.uint8) 
    if path is not None:
        label = '_' + label if label is not '' else ''
    for i in range(0, maze.shape[0]):
        for j in range(0,maze.shape[1]):
            if (i == 0 and j == 0) or (i==dim-1 and j == dim-1):
                color = 255
            else:
                color = 255 if maze[i,j] == 0 else 0
            image[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color
    # Save original Maze.
    im = Image.fromarray(image.astype(np.uint8))
    im.save("maze"+label+".jpeg")
    # If there is path, paint it on the maze!
    if path is not None:
        print("Computed traversal")
        for p in path:
            i, j = p[0], p[1]
            color = 150
            image[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color

        im = Image.fromarray(image.astype(np.uint8))
        im.save("maze_traversal"+label+".jpeg")
        return im

    return image

# --------------------------------------------------------

def draw_path_on_maze_img(mazeImg, path, label=''):

    tileSize = 16 # how large a tile is displayed on the vusalization image. reduce this to see smaller tiles!
    label = '_' + label if label is not '' else ''

    print("Computed traversal")
    for p in path:
        i, j = p[0], p[1]
        color = 150
        mazeImg[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color

    im = Image.fromarray(mazeImg.astype(np.uint8))
    im.save("maze_traversal"+label+".jpeg")

    return im

# --------------------------------------------------------

def euclidean_dist(maze=None, goal=None):

    if maze is not None and goal is not None:
        hMap = np.zeros(maze.shape)     # This table holds the heuristc's value for this pixel.
        # To see if a var is equal to None, python uses 'is' to compare.
        goal = [maze.shape[0]-1,maze.shape[1]-1] if goal is None else goal
        for i in range(1,hMap.shape[0]):
            for j in range(1,hMap.shape[1]):
                dist = np.sqrt((i-goal[0])**2+(j-goal[1])**2) # x**y is used to raise x to the y power.
                hMap[i,j] = np.around(dist, decimals=3)
        return hMap
# --------------------------------------------------------
def manhattan_dist(maze=None, goal=None):

    if maze is not None and goal is not None:
        hMap = np.zeros(maze.shape)     # This table holds the heuristc's value for this pixel.
        # To see if a var is equal to None, python uses 'is' to compare.
        goal = [maze.shape[0]-1,maze.shape[1]-1] if goal is None else goal
        for i in range(1,hMap.shape[0]):
            for j in range(1,hMap.shape[1]):
                dist = np.abs(i-goal[0]) + np.abs(j-goal[1]) # Manhattan or "zic-zac" distance.
                hMap[i,j] = dist
        return hMap

# --------------------------------------------------------

def simple_weight_init(size, origin = [1,1]):
    """ DESCRIPTION: This function will assign initial weights to adjascent-to-source nodes according to how far they are from the
                     origin (1,1) [or (0,0)in the original space. So node 5,3 as a distdance of4+2 = 7 from 
                     origin (1,1), becuase we cannot move diagonally, so to go diagonally we actually need to move
                     2 tiles, one to left/right and one up/down according to where we are going. SO, general
                     function used is d(p,s) = |(x-s_x) + (y-s_y)|, where p is the point and s the source.
                     Padded elements are also initialized to a high "inf" value.
                     so map will have the form a= | inf inf inf ....... inf|
                                                  | inf  0   1  inf        |
                                                  | inf  1   2  inf        |
                                                  | ... inf inf inf        |
                                                  | inf inf inf ....... inf|
        ARGUMENTS:   size ([x,y])-> Size typle for x and y dimensions.

        RETURNS:     wMap (np array)-> table that holds the starting ddistance of each node to rigin (1,1).
    """
    s_x = origin[0]
    s_y = origin[1]
    wMap = np.ones(size) * 1000
    for i in range(1,3):
        for j in range(1,3):
            wMap[i,j] = np.abs((s_x-i) + (s_y-j))
    return wMap

# --------------------------------------------------------

def path_planning(maze, plan = "DFS", distFunction = 'euclidean', tryAll = False, verbose = True):
    """ Description: This function will call the appropriate path discovery function
                     and will return the solution, if any.

        ARGUMENTS:   maze (np array)-> Array representation of maze. [0,1]
                     strategy (string or lsit)-> Selector of the path discovery strategy. If a list is given, all
                     the algortihms in the list will be run. 
                     Options : BFS,DFS, A_star, Bi-Directional.

                     distFunction (function type)-> Selector of the distance function for A* algorithm.
                     It is a function name which is then used to provide the heuristic distance towards the goal.
                     Default is Euclidean: d = sqrt(x^2+y^2).

                     tryAll (boolean)-> If this flag is set all algorithms will be run.
                     Verbose (boolean)-> Set this flag to enable printouts.

        RETURNS:     path (list of lists)-> List that holds all the computed Traversal paths. Empty if None.
    """
    existList = []
    pathsList = []
    args = []
    # Turn input into list soo we can iterate over it.
    if type(plan) != list:
        strategy = [plan]
    if type(plan) == dict:
        args = plan['args']
        strategy = plan['strategies']
    # For any given strategy, run the appropriate algorithm. If tryAll is given all algos will be run.
    for i, s in enumerate(strategy):
        if s == "BFS"or tryAll:
            if verbose:
                print("Perform BFS")
            exists, path = BFS(maze)
            existList.append(exists)
            pathsList.append(path)

        if s == "DFS" or tryAll:
            if verbose:
                print("Perform DFS")
            searchDir = args[i] if i < len(args) else standardDir
            exists, path = DFS(maze, searchDir = searchDir)
            existList.append(exists)
            pathsList.append(path)

        if s == "A*"or tryAll:
            distFunctions = [manhattan_dist, euclidean_dist]
            distFunction = args[i] if i < len(args) else distFunction
            if not tryAll:
                distFunctions = [distFunctions[0] if distFunction=='manhattan' else distFunctions[1]]
            for i in range(len(distFunctions)):
                if verbose:
                    print("Perform A* with distance function: {}".format(distFunctions[i])) # This is how you print onto screen. Also this how you get the type of a variable
                exists, path = A_star(maze, distFunction=distFunctions[i])
                existList.append(exists)
                pathsList.append(path)

        if s == "Bi-Directional"or tryAll:
            if verbose:
                print("Perform Bi-Directional BFS")
            exists, path = Bi_directional_BFS(maze)
            existList.append(exists)
            pathsList.append(path)


    for i, exists in enumerate(existList):
        if exists == 0 and verbose:
            print("No Valid Path found. Maze cannot be escaped --> HELP!")
    return existList, pathsList
# --------------------------------------------------------

def BFS(maze):
    """ DESCRIPTION: This function will perform BFS to see if the maze has a solution.
                     It first pads the maze with 1's to avoid corner checking, This trades off memory
                     for computational efficiency. It starts from element (0,0) and traverses the maze
                     considereing nodes as adjescent ONLY if they are empty that is they have a 0 value
                     in the maze representation. It keeps track of the traversal path.

        ARGUMENTS:   maze-> (numpy array) The table representation of thr maze

        RETURNS:    path-> (list of tuples) This is a list of tuples containing the  traversal.
                    Each item is a tuple that contains the coordinates of each step in the form (x_i,y_i).
                 
    """
    # Initialize variables
    path = []                             # List that holds the path
    discoveryMap = np.zeros(maze.shape)   # np array that marks which nodes have already been discovered
    visitMap = np.zeros(maze.shape)       # np array that holds the visit sequence
    pMaze = pad_maze(maze)                # padded maze. A larger Maze that has 1's on borders to avoid corer checking.
    mSize = maze.shape[0]                 # Size of original maze.
    nodeQ = queue.Queue()                 # FIFO queue to store discovered neighbors.
    goalReached, cnt = 0,0
    # Start at 0.0
    curNode = [1,1]
    discoveryMap[0,0] = 1

    nodeQ.put(curNode)
    while not nodeQ.empty():

        # Pop the next node in queue (one discovered for the longest time)
        curNode = nodeQ.get()
        # Transloate coords to original space and Append Node to traversal path!
        path.append([curNode[0]-1, curNode[1]-1])
        # print("Current Node: "+str(curNode))

        # Check all neighbors part.
        # ***
        # We dont need to corner check as we use a larger maze that has been padded with 1's. In essence
        # We trade off memory for speed!
        x, y = curNode[0], curNode[1]
        neighbors = pMaze[x-1:x+2,y-1:y+2]
        # Scan, clockwise, to see which neighborcells are free(0). Push free neighbors to FIFO queue.
        for i in range(neighbors.shape[0]):
            for j in range(neighbors.shape[1]):
                # Check this not the current node. That its not acutally the node we are looking around to examin its connectivity.
                # If it is do nothing, continue to next neighbor.
                if x+i-1 == x and y+j-1 == y:
                    continue
                # If the neighbor is the goal return
                if x == mSize and y == mSize:
                    print("Goal Reached after exploring {} nodes!".format(cnt+1))
                    path.append([mSize-1,mSize-1]) # Append the goal, which is always point (mSize,mSize) or (end,end) to the path.
                    return 1, path
                # Coordinates translation
                # ***
                # Translate coordinates to riginal maze size. basicall x_orig = x+i-2, y_orig = y+j-2: -1 for filter pos and -1 for padding
                x_orig, y_orig = (x+i-2), (y+j-2)
                # ---|

                # Check the following: 1) If this neighbor is empty. 2) If this empty neighbor has not been visited. 
                if (neighbors[i,j] == 0) and (discoveryMap[x_orig, y_orig] != 1):
                    # print(x,y,x_orig,y_orig,neighbors, discoveryMap) # Sanity print
                    discoveryMap[x_orig,y_orig] = 1      # Mark this node as discovered. No  other node should push this node again to the queue.
                    entryNode = [x+i-1,y+j-1]            # Form the coordinates of this node, in padded image coordinate frame.
                    nodeQ.put(entryNode)
        #---|
        cnt += 1 # Count the steps it took to find a solution!

    # If we reach this point no path exists. return 0 for failure and attempted path
    return 0, path
# --------------------------------------------------------

def DFS(maze, searchDir = standardDir):
    """ DESCRIPTION: This function will perform DFS to see if the maze has a solution
        ARGUMENTS: maze-> (numpy array) The table representation of thr maze

        RETURNS: path-> (list of tuples) This is a list of tuples containing the  traversal.
                 Each item is a tuple that contains the coordinates of each step in the form (x_i,y_i).
                 
    """
    # Initialize variables
    path = []                             # List that holds the path
    pMaze = pad_maze(maze)                # padded maze. A larger Maze that has 1's on borders to avoid corer checking.
    mSize = maze.shape[0]                 # Size of original maze.
    discoveryMap = np.ones(pMaze.shape)   # np array that marks which nodes have already been discovered. All padding elems are 1.
    discoveryMap[1:-1,1:-1] = 0           # Make the traversable nodes as not discovered.  
    visitMap = np.zeros(pMaze.shape)       # np array that holds the visit sequence

    nodeQ = queue.LifoQueue()                 # FIFO queue to store discovered neighbors.
    goalReached, cnt = 0,0
    # Start at 0.0
    curNode = np.array([1,1])
    discoveryMap[0,0] = 1

    nodeQ.put(curNode)
    while not nodeQ.empty():

        # Pop the next node in queue (one discovered for the longest time)
        curNode = nodeQ.get()
        # print("Current Node: "+str(curNode))

        x, y = curNode[0], curNode[1]
        # Translate to original unpadded coord space and Append Node to traversal path!
        path.append([x-1,y-1])
        # nodeSel = np.array([[-1,0], [0,-1], [0,1], [1,0]])    # select array. Select nodes UP, LEFT, RIGHT, DOWN.
        nodeSel = searchDir
        # Check UP-LEFT_RIGHT_DOWN neighbors part.
        # ***
        # We dont need to corner check as we use a larger maze that has been padded with 1's. In essence
        # We trade off memory for speed!
        for n in nodeSel:
            # Get coords of neighbor and its occupancy value.
            x, y = n[0]+curNode[0], n[1]+curNode[1]
            nValue = pMaze[x,y]
            # print("Neighbor: ", x,y, nValue)

            # If the neighbor is the goal return
            if x == mSize and y == mSize:
                print("Goal Reached after exploring {} nodes!".format(cnt+1))
                path.append([mSize-1,mSize-1]) # Append the goal, which is always point (mSize,mSize) or (end,end) to the path.
                return 1, path
            # Check to see if node is occupied or already discovered. If not add it to the stack.
            if(nValue != 1) and (discoveryMap[x,y] !=1):
                discoveryMap[x,y] = 1      # Mark this node as discovered. No  other node should push this node again to the queue.
                entryNode = [x,y]          # Form the coordinates of this node, in padded image coordinate frame.
                nodeQ.put(entryNode)
        #---|
        cnt += 1 # Count the steps it took to find a solution!

    # If we reach this point no path exists. return 0 for failure and attempted path
    return 0, path

# --------------------------------------------------------

def A_star(maze, distFunction = euclidean_dist()):
    """ DESCRIPTION: This function will perform A* to see if the maze has a solution
        ARGUMENTS: maze-> (numpy array) The table representation of thr maze
                   distFunction-> (String) A string selecting the required dist function: either
                   Euclidean or Manhattan.

        RETURNS: path-> (list of tuples) This is a list of tuples containing the  traversal.
                 Each item is a tuple that contains the coordinates of each step in the form (x_i,y_i).
                 
    """
    # Initialize variables
    path = []                             # List that holds the path
    pMaze = pad_maze(maze)                # padded maze. A larger Maze that has 1's on borders to avoid corer checking.
    mSize = maze.shape[0]                 # Size of original maze.
    discoveryMap = np.ones(pMaze.shape)   # np array that marks which nodes have already been discovered. All padding elems are 1.
    discoveryMap[1:-1,1:-1] = 0           # Make the traversable nodes as not discovered.  
    visitMap = np.zeros(pMaze.shape)       # np array that holds the visit sequence

    nodeQ = queue.PriorityQueue()                 # FIFO queue to store discovered neighbors.
    goalReached, cnt = 0,0
    # Start at 0.0
    curNode = [0, np.array([1,1])]
    discoveryMap[1,1] = 1
    # Initialize Weights
    wMap = simple_weight_init(pMaze.shape)
    aStarDist = np.ones(pMaze.shape) * 1000
    # Compute heuristicdistance to the goal!
    endPoint = np.array([mSize, mSize])
    hMap = distFunction(pMaze, endPoint)
    # ---|
    nodeQ.put(curNode)

    while not nodeQ.empty():
        # Pop the next node in queue (one with smallest A* dist from the end)
        # Here queue returns a [dist, [x,y]] entry, where dist is the A* distance and [x,y] this node's coords.
        entry = nodeQ.get()  # Keep in mind that queue.get() "pops" the element from the q. That is after get it is no longer stored in q.
        curSrcDist = entry[0] 
        curNode = entry[1]
        # print("Current Node: "+str(curNode))

        x, y = curNode[0], curNode[1]
        # Translate to original unpadded coord space and Append Node to traversal path!
        path.append([x-1,y-1])
        nodeSel = np.array([[-1,0], [0,-1], [0,1], [1,0]])    # select array. Select nodes UP, LEFT, RIGHT, DOWN.
        # Check UP-LEFT_RIGHT_DOWN neighbors part.
        # ***
        # We dont need to corner check as we use a larger maze that has been padded with 1's. In essence
        # We trade off memory for speed!
        for n in nodeSel:
            # Get coords of neighbor and its occupancy value.
            nX, nY = n[0]+curNode[0], n[1]+curNode[1]
            nValue = pMaze[nX,nY]
            # print("Neighbor: ", nX, nY, nValue)

            # If the neighbor is the goal return.
            if nX == mSize and nY == mSize:
                print("Goal Reached after exploring {} nodes!".format(cnt+1))
                path.append([mSize-1,mSize-1]) # Append the goal, which is always point (mSize,mSize) or (end,end) to the path.
                return 1, path
            # Bussiness Logic
            # ***
            # Check to see in neighbor is not occupied. Avoid unnessacary computations!
            if nValue != 1:
                # A. Dynamic Programming part
                # ---
                # 1. Compute new neighbor-source distance. Aonly consider UP|LEFT|RIGHT|DOWN neighbors, so here move weight is always 1. 
                newSourceDist = min(wMap[nX,nY], wMap[x,y]+1) 
                # 2. Update the node-source distance with the smallest found value.
                wMap[nX,nY]   = newSourceDist
                # 3. Update A* star distance.
                aStarDist[nX,nY] = newSourceDist + hMap[nX,nY]       # With this distance, A* star sorts the next-to-explore queue!  
                # B. Insertion to priority queue part.
                # 1. Check to see if node is already discovered. Here we defer from previous algos. Here A* will sort the discovered
                # nodes queue, where the nodes closest to the goal, dist_to_goal = dist_from source + heuristic_dist_to_goal, are right on
                # top of the queue
                if discoveryMap[nX,nY] !=1:
                    discoveryMap[nX,nY] = 1      # Mark this node as discovered. No  other node should push this node again to the queue.
                    entryNode = [nX,nY]          # Form the coordinates of this node, in padded image coordinate frame.
                    nodeQ.put([aStarDist[nX,nY], entryNode]) # We need both the A* star dist and this nodes coords to be stored in our priority queue.
    
        cnt += 1
    return 0, path

# --------------------------------------------------------
def Bi_directional_BFS(maze,verbose = False):
    """ DESCRIPTION: This function will perform Bi-Directional BFS to see if the maze has a solution
        ARGUMENTS: maze-> (numpy array) The table representation of thr maze

        RETURNS: path-> (list of tuples) This is a list of tuples containing the  traversal.
                 Each item is a tuple that contains the coordinates of each step in the form (x_i,y_i).
    """
    # Initialize variables
    path  = []                            # List that holds the path fro mthe top left direction.
    path2 = []                            # List that holds the path from the bottom right direction.
    pMaze = pad_maze(maze)                # padded maze. A larger Maze that has 1's on borders to avoid corer checking.
    discoveryMap = np.zeros(pMaze.shape)  # np array that marks which nodes have already been discovered
    visitMap  = np.zeros(pMaze.shape)      # np array that holds the visit sequence
    visitMap2 = np.zeros(pMaze.shape)      # np array that holds the visit sequence
    mSize = maze.shape[0]                 # Size of original maze.
    nodeQ = queue.Queue()                 # FIFO queue to store discovered neighbors.
    nodeQ2 = queue.Queue()                # FIFO queue to store discovered neighbors, from the bottom right direction.
    goalReached, cnt = 0,0
    # Start one direction at 0.0
    curNode = [1,1]
    discoveryMap[1,1] = 1
    # Start the other direction at end,end
    curNode2 = [mSize,mSize]
    discoveryMap[mSize,mSize] = 1
    # Place start nodes in their queues.
    nodeQ.put(curNode)
    nodeQ2.put(curNode2)

    # If doth queues are empty stop.
    while not nodeQ.empty() and not nodeQ2.empty():

        # Pop the next node in queue (one discovered for the longest time), from both directions
        curNode  = nodeQ.get()
        curNode2 = nodeQ2.get()
        # Mark the node as visited!
        visitMap[curNode[0],  curNode[1]]  = 1
        visitMap2[curNode2[0], curNode2[1]] = 1
        # Append Node to traversal path! We need to translate it to the original coord frame first.
        path.append([curNode[0]-1, curNode[1]-1])
        path2.append([curNode2[0]-1, curNode2[1]-1])
        if verbose:
            print("Current Top Left Node: "+str(curNode)+ "Current Bottom Right Node: " + str(curNode2))

        # Check all neighbors part.
        # ***
        # We dont need to corner check as we use a larger maze that has been padded with 1's. In essence
        # We trade off memory for speed!
        nodeSel  = -np.array([[-1,0], [0,-1], [0,1], [1,0]])    # select array. Select nodes UP, LEFT, RIGHT, DOWN.
        nodeSel2 = -nodeSel                                    # select array. Select nodes UP, LEFT, RIGHT, DOWN.
        for n,n2 in zip(nodeSel,nodeSel2):
            # Get coords of neighbor and its occupancy value.
            nX, nY = n[0]+curNode[0], n[1]+curNode[1]
            nX2, nY2 = n2[0]+curNode2[0], n2[1]+curNode2[1]
            nValue = pMaze[nX,nY]
            nValue2 = pMaze[nX2,nY2]
            # print("Neighbor: ", nX, nY, nValue)
            if (nValue != 1) and (nValue2 !=1):
                if (nX == nX2 and nY == nY2) or (visitMap[nX2,nY2] == 1) or (visitMap2[nX, nY] ==1):
                    print(" The two directions MET at {},{} !!\nGoal Reached after exploring {} nodes! {} {}".format(nX-1,nY-1, cnt+1, len(path), len(path2))) # It is easier to print meeting p at orgi coords
                    path = path + path2[::-1] # Reverse path2, This is because the last node in path2 is the last one visited, the one that the two paths met.  List + concates lists at their left size
                    if verbose:
                        print(path)
                    return 1, path
                # If the neighbor is the goal return.
                if (nX == mSize and nY == mSize) or (nX2 == 1 and nY2 == 1):
                    print("Goal Reached after exploring {} nodes! {}".format(cnt+1))
                    path.append([mSize-1,mSize-1]) # Append the goal, which is always point (mSize,mSize) or (end,end) to the path.
                    return 1, path
            # Bussiness Logic
            # ***
            # Check to see in neighbor is not occupied. Avoid unnessacary computations!
            if nValue != 1:
                # Check the following: 1) If this neighbor is empty. 2) If this empty neighbor has not been visited. 
                if (discoveryMap[nX, nY] != 1):
                    # print(x,y,x_orig,y_orig,neighbors, discoveryMap) # Sanity print
                    discoveryMap[nX, nY] = 1      # Mark this node as discovered. No  other node should push this node again to the queue.
                    entryNode = [nX, nY]            # Form the coordinates of this node, in padded image coordinate frame.
                    nodeQ.put(entryNode)
            if nValue2 != 1:
                # Check the following: 1) If this neighbor is empty. 2) If this empty neighbor has not been visited. 
                if (discoveryMap[nX2, nY2] != 1):
                    # print(x,y,x_orig,y_orig,neighbors, discoveryMap) # Sanity print
                    discoveryMap[nX2, nY2] = 1      # Mark this node as discovered. No  other node should push this node again to the queue.
                    entryNode = [nX2, nY2]            # Form the coordinates of this node, in padded image coordinate frame.
                    nodeQ2.put(entryNode)
        #---|
        cnt += 1 # Count the steps it took to find a solution!

    # If we reach this point no path exists. return 0 for failure and attempted path. THe path is the joined paths(the bottom right one get reversed, makes
    # printing later one easier)
    return 0, path+path2[::-1]

# --------------------------------------------------------

def pad_maze(maze, value =1):
    
    paddedMaze = np.ones([maze.shape[0]+2, maze.shape[1]+2])
    paddedMaze[1:maze.shape[0]+1,1:maze.shape[1]+1] = maze
    # print(paddedMaze)
    return paddedMaze

# =======================================================================================
def main():
    """ DESCRIPTION: This is main function of the file. Here we will call all the necessary function
                   To perform our tasks, in order. iT has no arguments. THis is a classic documentation
                   type of comment in python, right after the definition of a function. NOTE: python
                   denotes strings with "" or ''. To have an end-of-line character in there you need
                   like in this documentation, """ """

      ARGUMENTS: None

      RETURNS:  None
    """

    # Crate argument parser. This will accept inputs from the console.
    # But if no inputs are given, the default values listed will be used!
    parser = argparse.ArgumentParser(prog='Maze Exploration!')
    # Tell parser to accept the following arguments, along with default vals.
    parser.add_argument('--part', type = str,  metavar = 'dim', default=1,    help="Which part of the assignment to tackle")
    parser.add_argument('--dim', type = int,   metavar = 'dim', default=8,    help="Maze dimension. Maze is Square.")
    parser.add_argument('--p',   type = float, metavar = 'p',   default=0.2,  help="Occupancy probability, per cell.")
    parser.add_argument('--s',   type = str,   metavar = 's',   default='BFS',help="Exploration Strategy. Options: DFS,BFS,A*")
    parser.add_argument('--d',   type = str,   metavar = 'd',   default='0'   ,help="Distance function to be used for the heuristic distance of A*")
    parser.add_argument('--all', action='store_true',help="Flag to enable trying out all the algortihms")
    parser.add_argument('--trials', type = int,   metavar = 't',default=100   ,help="Number of trilas for the various monte carlo tests.")
    # Parse the input from the console. To access a specific arg-> dim = args.dim
    args = parser.parse_args()

    # PART 1
    # *******
    if args.part == 1:
        # Initialize
        # ----------
        # Generate map
        mapDim     = args.dim
        occupyProb = args.p
        maze = create_maze(mapDim, occupyProb)
        # Visualize map, save it and hold in mazeImg to use it for drawing the path on.
        mazeImg =draw_maze(maze)
        # ---|

        # Try to traverse map with required algorithms
        dist = 'eucludian' if args.d is '0' else args.d
        # Time execution. We can use this to find a large enough dimension. The graph is fully connceted so complexity
        # is O(n^2).
        startTime = time.time()
        existList, pathsList = path_planning(maze, plan = args.s, distFunction=dist, tryAll=args.all)
        execTime = time.time() - startTime
        print("--- %s seconds ---" % (execTime))
        # Visualize traversal on maze.
        # Generate labels for drawing and saving.
        labels = ["BFS", "DFS", "A_star_manhattan", "A_star_euclidean", "Bi_directional"] if args.all else [args.s+'_'+args.d]
        for i in range(len(existList)):
            cMazeImg = mazeImg.copy()
            draw_path_on_maze_img(cMazeImg, pathsList[i],label = labels[i])
    # PART 2
    # *******
    if '2' in args.part:
        if 'c' in args.part:
            # Plot p vs solvability
            trials = args.trials
            pathsList = []
            successes = [0, 0, 0 ,0 ,0]
            labels = ["BFS", "DFS", "A_star_manhattan", "A_star_euclidean", "Bi_directional_BFS"]
            probs = np.arange(0.05, 0.7, 0.05)
            successList = [[]for i in range(5)]
            for i, p in enumerate(probs): 
                for j in range(trials):
                    print("Trial: {} for p: {}".format(j+1, p))
                    maze = create_maze(args.dim, p)
                    dist = 'eucludian' if args.d is '0' else args.d
                    exists, path = path_planning(maze, plan = args.s, distFunction=dist, tryAll=True)
                    # Store for all algorithms if they succeed or not.
                    for s in range(5):
                        successes[s] += exists[s]
                for s in range(5):
                    successList[s].append(successes[s]/trials)
                    successes[s] = 0
            for s in range(5):
                # Plot the suc vs occupanncy prob plot!
                plt.plot(probs, successList[s])              
                plt.title("{}: Success vs Occupancy Plot, dims = {}, trials = {}".format(labels[s], args.dim, trials)) # plot format
                plt.ylabel("Success probability s")     # name y axis
                plt.xlabel("Occupancy propability p")   # name x axis
                plt.savefig('{}_Success_vs_occupancy_dim_{}_trials_{}.png'.format(labels[s], args.dim, trials)) # save figure at the same folder as code
                plt.close()
            # ---|
        # Bullet g or 7
        if 'g' in args.part:
            # Generate map
            mapDim     = args.dim
            occupyProb = args.p
            maze = create_maze(mapDim, occupyProb)
            # Visualize map, save it and hold in mazeImg to use it for drawing the path on.
            mazeImg =draw_maze(maze)
            # ---|
            # Evaluate looking towards the goal vs the opposite way.
            plan = {'strategies':['DFS', 'DFS'], 'args': [np.array([[-1,0], [0,-1], [0,1], [1,0]]), -np.array([[-1,0], [0,-1], [0,1], [1,0]])]}
            existList, pathsList = path_planning(maze, plan = plan)
            labels = ["DFS_towards_goal", "DFS_towards_opposite_of_goal"] 
            for i in range(len(existList)):
                cMazeImg = mazeImg.copy()
                draw_path_on_maze_img(cMazeImg, pathsList[i],label = labels[i])
        # Bullet h or 8
        if 'h' in args.part:
            # Generate map
            mapDim     = args.dim
            occupyProb = args.p
            maze = create_maze(mapDim, occupyProb)
            # Visualize map, save it and hold in mazeImg to use it for drawing the path on.
            mazeImg =draw_maze(maze)
            # ---|
            # Evaluate looking towards the goal vs the opposite way.
            plan = {'strategies':['Bi-Directional', 'A*'], 'args': [np.array([[-1,0], [0,-1], [0,1], [1,0]]), 'manhattan']}
            existList, pathsList = path_planning(maze, plan = plan)
            labels = ["Bi-Directional_h", "A_star_manhttan_h"] 
            for i in range(len(existList)):
                cMazeImg = mazeImg.copy()
                draw_path_on_maze_img(cMazeImg, pathsList[i],label = labels[i])


    # PART 3
    # *******

    return 1

# =======================================================================================
# MAIN
# Standard way to call a python top level. This essentially says: if this file is called
# from the console or another programm as the top level file, run the main() function
# =======================================================================================
if __name__ == "__main__":
    main()
