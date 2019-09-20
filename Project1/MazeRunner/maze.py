# Import any libraries we might use!
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
import queue
import argparse
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
    # print(maze)
    # More pythonic way.
    # make a random choice of numbers 0, 1, enough to fill an array of size (dim,dim).
    # 0 and 1  have probabilities 1-p, p to occur, respectively. More compact way where
    # we make use of numpy library, also much faster!!!
    maze = np.random.choice([0,1], (dim,dim), p=[ 1-p, p])
    # Ensure  start and points are free!
    maze[0,0], maze[-1,-1] = 0, 0
    print(maze)
    # return the value
    return maze

# --------------------------------------------------------

def draw_maze(maze, path = None):
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
    image = np.empty((dim*tileSize,dim*tileSize)) 
    for i in range(0, maze.shape[0]):
        for j in range(0,maze.shape[1]):
            if (i == 0 and j == 0) or (i==dim-1 and j == dim-1):
                color = 255
            else:
                color = 255 if maze[i,j] == 0 else 0
            image[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color
    # Save original Maze.
    im = Image.fromarray(image.astype(np.uint8))
    im.save("maze.jpeg")
    # If there is path, paint it on the maze!
    if path is not None:
        print("Computed traversal")
        for p in path:
            i, j = p[0], p[1]
            color = 150
            image[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color

        im = Image.fromarray(image.astype(np.uint8))
        im.save("maze_traversal.jpeg")
        return im

    return im
# --------------------------------------------------------

def path_planning(maze, strategy = "DFS", distFunction = "Euclidean"):
    """ Description: This function will call the appropriate path discovery function
                     and will return the solution, if any.

        ARGUMENTS:   maze (np array)-> Array representation of maze. [0,1]
                     strategy (string)-> Selector of the path discovery strategy.
                     distFunction (string)-> Selector of the distance function for A* algorithm.
                     Default is Euclidean: d = sqrt(x^2+y^2).

        RETURNS:     path (list)->Traversal path. Empty if None.
    """
    if strategy == "DFS":
        print("Perform DFS")
        exists, path = DFS(maze)
    elif strategy == "BFS":
        print("Perform BFS")
        exists, path = BFS(maze)
    elif strategy == "A*":
        print("Perform A* with distance function: {}".format(distFunction)) # This is how you print onto screen
        path = A_star(maze, distFunction=distFunction)
    elif strategy == "Bi-Directional":
        print("Perform Bi-Directional BFS")
        path = Bi_directional_BFS(maze)

    if exists == 0:
        print("No Valid Path found. Maze cannot be escaped --> HELP!")
    return exists, path
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

    print(maze.shape) 
    nodeQ.put(curNode)
    while not nodeQ.empty():

        # Pop the next node in queue (one discovered for the longest time)
        curNode = nodeQ.get()
        # Transloate coords to original space and Append Node to traversal path!
        path.append([curNode[0]-1, curNode[1]-1])
        print("Current Node: "+str(curNode))

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

def DFS(maze):
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
        print("Current Node: "+str(curNode))

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
            x, y = n[0]+curNode[0], n[1]+curNode[1]
            nValue = pMaze[x,y]
            print("Neighbor: ", x,y, nValue)

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

def A_star(maze, distFunction = 'Euclidean'):
    """ DESCRIPTION: This function will perform A* to see if the maze has a solution
        ARGUMENTS: maze-> (numpy array) The table representation of thr maze
                   distFunction-> (String) A string selecting the required dist function: either
                   Euclidean or Manhattan.

        RETURNS: path-> (list of tuples) This is a list of tuples containing the  traversal.
                 Each item is a tuple that contains the coordinates of each step in the form (x_i,y_i).
                 
    """
    path = []

    return path

# --------------------------------------------------------

def Bi_directional_BFS(maze):
    """ DESCRIPTION: This function will perform Bi-Directional BFS to see if the maze has a solution
        ARGUMENTS: maze-> (numpy array) The table representation of thr maze

        RETURNS: path-> (list of tuples) This is a list of tuples containing the  traversal.
                 Each item is a tuple that contains the coordinates of each step in the form (x_i,y_i).
    """
    path = []

    return path

# --------------------------------------------------------

def pad_maze(maze, value =1):
    
    paddedMaze = np.ones([maze.shape[0]+2, maze.shape[1]+2])
    paddedMaze[1:maze.shape[0]+1,1:maze.shape[1]+1] = maze
    print(paddedMaze)
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
    parser.add_argument('--dim', type = int,   metavar = 'dim', default=8,    help="Maze dimension. Maze is Square.")
    parser.add_argument('--p',   type = float, metavar = 'p',   default=0.2,  help="Occupancy probability, per cell.")
    parser.add_argument('--s',   type = str,   metavar = 's',   default='BFS',help="Exploration Strategy. Options: DFS,BFS,A*")
    # Parse the input from the console. To access a specific arg-> dim = args.dim
    args = parser.parse_args()

    # PART 1
    # *******
    # Initialize
    # ----------
    # Generate map
    mapDim     = args.dim
    occupyProb = args.p
    maze =create_maze(mapDim, occupyProb)
    # Visualize map
    draw_maze(maze)
    # ---|

    # Try to traverse map with required algorithms
    exists, path = path_planning(maze, strategy = args.s)
    # Visualize traversal on maze.
    draw_maze(maze, path)
    # PART 2
    # *******

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
