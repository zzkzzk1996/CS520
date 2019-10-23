# Import any libraries we might use!
import numpy as np    # General numerical array manipulation framework
from PIL import Image # Used to store mage as a nice image
import queue        # used for FiFO and LiFo used for BFS and DFS respectively
import heapq        # used for A* priority queue
import argparse     # Used to parse console arguments
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import sys
standardDir = np.array([[-1,0], [0,-1], [0,1], [1,0]])
# ======================================================================================
# FUNCTIONS
# **********
lt0 = lambda a, n: (a + n if a + n > 0 else 0)
gtb = lambda a, n, b: (a + n if a + n < b else b)
compOffset = lambda o,ll,hl: (1 if ( o<hl and o > ll) else 0)   # used to compute the limits for 1-neighbor search in a grid, exclusivelly
compLimits = lambda o,ll,hl: (l if ( o<hl and o > ll) else (ll if o <= ll else hl)) # more general form of the above

def create_field(dim = 32, n = 8):

    cnt = 0                     # number of mine already placed
    field = np.zeros((dim,dim)) # mine field, empty
    # While filed has less than n mines... add mines!
    while cnt < n:
        x = np.random.choice(dim, 1)
        y = np.random.choice(dim, 1)
        if field[x,y] == 0:
            field[x,y] = 1
            cnt += 1

    # return the minefield
    return field

# --------------------------------------------------------

def draw_field(maze, path = None, fMap = None, label =''):
    """ DESCRITPTION: This function draws the field to an image format for visualization purposes.
                      It gets the dimensions of thefield, and draws an image where each cell of the
                      originnal maze is displayed as a tileSize x tileSize block in the image.
                      Empty celss (field value =0) are displayed as white, occupied cells (feild value=1)
                      as black and path tiles as gray,
                
        ARGUMENTS: feild-> (numpy array) A table-like represenation of the field, where 1's represents
                          occupied space
        RETURNS: image-> () Image representation of maze
                 image-> () Image representation of traversal attempt of maze
    """
    tileSize = 16 # how large a tile is displayed on the vusalization image. reduce this to see smaller tiles!
    dim = maze.shape[0]
    image = np.empty((dim*tileSize,dim*tileSize), dtype=(np.uint8,3)) 
    if path is not None:
        label = '_' + label if label is not '' else ''
    for i in range(0, maze.shape[0]):
        for j in range(0,maze.shape[1]):
            if (i == 0 and j == 0) or (i==dim-1 and j == dim-1):
                color = [255,255,255]
            else:
                color = [255,255,255] if maze[i,j] == 0 else [0,0,0]
                color = [0,0,250] if (fMap is not None and fMap[i,j] == 1 and maze[i,j] == 1) else ([0,208,0] if (fMap is not None and fMap[i,j] == 1) else color)
            image[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color
    # Save original Maze.
    im = Image.fromarray(image.astype(np.uint8))
    im.save("MineField"+label+".jpeg")
    # If there is path, paint it on the maze!
    if path is not None:
        cnt = 0
        print("Computed traversal")
        for p in path:
            i, j = p[0], p[1]
            color = [100,100,100]
            if fMap[i,j] == 2: # 2 means I steped on mine
                color = [255,0,0] # color it red!
            image[i*tileSize:i*tileSize+tileSize, j*tileSize:j*tileSize+tileSize] = color

            im = Image.fromarray(image.astype(np.uint8))
            # im.save("gif/"+str(cnt)+"_"+"Field_traversal"+label+".jpeg")
            im.save("gif/{0:0=3d}_Field_Traversal+{1}.jpeg".format(cnt, label))
            cnt += 1
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

def query_field(field, coords, qType = 'mineQuery', verbose = False):
    """ DESCRIPTION: THis function returns wheteh a cell is mined or not. If it is the quering player explodes,
                     The game is over; if not, it returns an answer containg the amount of mines around the clear cell.
        ARGUMENTS: field (np 2D array)
        RETURNS: sum(nCells) (int): Sum of neighbors that are 1) mined in the mineQuery 2) Discovered in the discoveryQuery
        nCells (np array 2D): The local neighborhood of 1) all surrounding cells 
   """
    x = coords[0]
    y = coords[1]
    answer = None
    if field[y,x] == 1 and qType == 'mineQuery':
        if verbose:
            print("BOOOM!!! Unfortunately there was a mine at ({},{}).".format(y,x))
            print(field)
        return None
    else:
       xLrange = compOffset(x-1, 0, field.shape[1]) 
       xUrange = compOffset(x+1, 0, field.shape[1]) +1
       yLrange = compOffset(y-1, 0, field.shape[0]) 
       yUrange = compOffset(y+1, 0, field.shape[0]) +1

       nCells = field[y-yLrange:y+yUrange, x-xLrange:x+xUrange]
       # TODO: clear this up a bit when it works
       if qType is "mineQuery":
           numAnswer = np.sum(nCells)
           return numAnswer
       if qType is not 'mineQuery':
           discoveredCells=nCells[nCells>0] # HOW MANY NEIGHBORS  are discovered
           numAnswer = max(np.sum(discoveredCells) -1,0)
           # nCells = np.argwhere(field[x-xLrange:x+xUrange, y-yLrange:y+yUrange])
           # discoveredCells = np.argwhere(field[(x-xLrange:x+xUrange, y-yLrange:y+yUrange) & feild == 0]
           return numAnswer

# --------------------------------------------------------
def minesweeper(field, plan = {'strategy' : 'naive', 'manyLives': 20}, verbose = True):

    path = [] # List holding the navigation path
    success = 0
    if 'naive' in plan['strategy']:
        algoArgs = dict(manyLives = plan['manyLives'], verbose = verbose)
        path, success, fMap, livesLost = naive_navigate(field, **algoArgs)

    return path, success, fMap, livesLost
# --------------------------------------------------------
def distribute_mine_prob(field, probPerCell, verbose = False):
    x = coords[0]
    y = coords[1]
    answer = None
    if field[x,y] == 1 and qType == 'mineQuery':
        if not Verbose:
            return
        print("BOOOM!!! Unfortunately there was a mine at ({},{}).".format(y,x))
    else:
        xLrange = compOffset(x-1, 0, field.shape[1]-1) 
        xUrange = compOffset(x+1, 0, field.shape[1]) 
        yLrange = compOffset(y-1, 0, field.shape[0]-1) 
        yUrange = compOffset(y+1, 0, field.shape[0]) 

# --------------------------------------------------------
def naive_navigate(field, n = 0, safePlay = False, manyLives = 20, minedChance = 0.5, verbose = True):
    """ DESCRIPTION: This function performs a naive sweeping of the field. It simply greedily
                     picks the next cell to query, where its mine-occupancy chance is the lowest.
        ARGUMENTS: field (np 2D array:) array indicating the mine field. 1 means mine.
                   n (int): Indicates knowledge of total amount of mines. 0 means no knowledge (or a very pleasant field)
                   safePlay (Boolean): Indicates whether the minesweeper will accept a lower score in order to more probably
                            navigate the field with one life. If True and visited clear cell decreases the available rounds to
                            play by one. Resoning is cells are fixed, since we discovered a free one, the remaining free ones are
                             fixed-1. This results in some potentially free cells never being visited.
                   manyLives (int): How may attempts we have to finish the same map. If we explode-nobiggie-mark the cell as visited
                             and unfortunate and continue.

        Returns: path (list): THe sweeping path generated.
                 success (int): 0 or 1 indicating success or failure. Old school.
                 flagMap (2D array): a map that the sweeper estimates there mines underneath.
    """
    path = []
    success = 0
    rows, cols = field.shape[0], field.shape[1]
    sMap = np.ones(field.shape) * 1/(rows*cols) # minesweeper's map estimate. Inti to unofrom chance.
    flagMap = np.zeros(field.shape)
    dMap = np.zeros(field.shape)
    dQueue = queue.PriorityQueue() 
    discBy = np.ones(field.shape, dtype=(int,2)) *-1 # Hold the discoverer of the indexed node
    hasDisc = [[[] for x in range(cols)] for y in range(rows)]          # Hols a list of the node the indexed node has discovered.
    estimatedFreeCells = rows * cols# sweeper's estimate of remaing free cells. WHen 0 eploration stops!
    livesLost, visitedCells = 0,1
    cnt = 0 # Generic count variable
    sanity = 0
    # Pick a random point to start by setting its freeProb a tad lower.
    seedX, seedY = np.random.randint(0, cols), np.random.randint(0,rows)
    sMap[seedY, seedX] -= 0.001

    # while estimatedFreeCells-visitedCells >= 0:
    while (rows*cols) > visitedCells:
        # Pick the next best point
        minIdx = np.argmin(sMap)   # returns the index in 1D format. Need to be converted to 2D.
        nextY,nextX = np.unravel_index(minIdx, [rows,cols])
        # nextX = minIdx % cols      # General 2D->1D formula: idx = y * cols + x. So 1D->2D: x= i%cols
        # nextY = int((minIdx - nextX) / cols) # y = (idx-x)/cols
        if dMap[nextY,nextX] == 1: # nly way to revisit a cell is when all others have been descovered!
            print("Revisited cell {},{}. Exiting... {}".format(nextY,nextX, sMap[nextY,nextX]))
            success = -1
            break
        # Do not query a flag-as-mine-cell. Basically because we use a belief potential, when the field is close to equilibrium
        # 
        if flagMap[nextY,nextX] == 1: 
            sMap[nextY,nextX] += 1
            print("Flagged Cell {},{}, sMap: {}".format(nextY,nextX, sMap[nextY,nextX]))
            continue
        else:
            visitedCells += 1
            dMap[nextY,nextX] = 1 # this cell is discovered blow up or not; a good day?

        # Get how many mines are around this cell. Distribute the probability equally around the center cell.
        answer = query_field(field, [nextX, nextY])
        if verbose:
            print("Iter: {}. Remaining cells: {}. Visiting cell {},{}...".format(cnt, rows*cols-visitedCells, nextY,nextX))
        if answer is not None:
            if verbose:
                print(" It has {} mines around it!".format(answer))
            # Check if any of the surrounding cells are discovered. If they are, ingore them and redistribute their prob mass
            # To the remaining cells.
            numDiscNeighbors  = query_field(dMap, [nextX, nextY], qType = 'discNeighborsQuery')  # find how many neighbors around the current cell are discovered 
            # Check if all negihbors are descovered. If so, probability of mine is 0
            probDivider = (8-numDiscNeighbors) 
            probPerCell = (answer / probDivider) if probDivider > 0 else 0
            if verbose:
                print("probDiv {}, probPerCell {}".format(probDivider, probPerCell))
            # If probPerCell is 0, get all neighbors and zero out their mine probability else distribute mass.
            xLrange = compOffset(nextX-1, 0, field.shape[1]-1) 
            xUrange = compOffset(nextX+1, 0, field.shape[1]) +1
            yLrange = compOffset(nextY-1, 0, field.shape[0]-1) 
            yUrange = compOffset(nextY+1, 0, field.shape[0]) +1

            # Distribute probability to non discovered neighbors.
            # print(nextY-yLrange,nextY+yUrange, nextX-xLrange, nextX+xUrange)
            factor = 1 if probPerCell > 0 else 0 # Factor will zero each cell that is not yet discovered, if there are no mines rported around hte queried point!
            for y in range(nextY-yLrange, nextY+yUrange):
                for x in range(nextX-xLrange, nextX+xUrange):
                    
                    if dMap[y,x] == 0:
                        # if prob is 0 then resitribute this cells prop to all the other cells is original founder had also found!
                        if factor == 0:
                            discoverer = discBy[y,x]  # get original founder. Note that its [y,x]
                            if (discoverer >-1).all():  # check if there actually is a foudner yet
                                divBy = max(len(hasDisc[discoverer[0]][discoverer[1]]),1)
                                for i in hasDisc[discoverer[0]][discoverer[1]]:
                                    if dMap[i[0],i[1]] == 0:
                                        sMap[i[0],i[1]] += sMap[y,x]/divBy #- sMap[y,x]/divBy * sMap[i[0],i[1]]
                                        if sMap[i[0],i[1]] < 0:
                                            print("{},{}, s: ".format(i[0],i[1],sMap[i[0],i[1]]))
                                        if sMap[i[0],i[1]] > minedChance : # if it is too risky.... mark it as a mine!
                                            sMap[i[0],i[1]] = 1000        # set its value way too high
                                            estimatedFreeCells -= 1       # reduced the estimated free cells of the mine  
                                            flagMap[i[0],i[1]] = 1        # flag it as a mined cell.
                        # ---| End of prob redistribution
                        sMapOld = sMap[y,x]
                        sMap[y,x] = (sMap[y,x] + probPerCell - probPerCell * sMap[y,x]) * factor
                        hasDisc[nextY][nextX].append([y,x]) # log which nodes the cur node (nextY,nextX) has found
                        discBy[y,x] = nextY,nextX           # log the discoverer of the node y,x
                        # ---| End of Prob distribution.
                        if sMap[y,x] > minedChance: # if it is too risky.... dont go!
                            estimatedFreeCells -= 1 
                            flagMap[y,x] = 1
                            sMap[y,x] = 1000
                        else:
                            if sMapOld > minedChance:
                                estimatedFreeCells += 1 
                                visitedCells -= 1
                                dMap[y,x] = 0
                            flagMap[y,x] = 0
                            sMap[y,x] = sMap[y,x]
                        # ---| End of mark cell as mined!
                                
            sMap[nextY, nextX] = 100 # very large value > 1 means discovered
            path.append([nextY,nextX]) # append the recently discovered cell to traversal path
            # if safePlay:
            # estimatedFreeCells -= 1
        else:
            # estimatedFreeCells -= 1
            livesLost += 1
            if verbose:
                print("Woopsie daisies! Remaning Lives {}".format(manyLives-livesLost))
            path.append([nextX,nextY]) # append the recently discovered cell to traversal path
            sanity += (1 if flagMap[nextY,nextX] == 1 else 0)
            flagMap[nextY, nextX] = 2 # means stepped on mine...ouch!
            sMap[nextY, nextX] = 10000 # very large value > 1 means discovered
            # print(sMap)
            if livesLost >= manyLives:
                break

        # development phase eternal loo[p safeguard.
        cnt += 1
        if cnt > rows*cols*10:
            break
    if success != -1:
        success = 1
    if verbose:
        print(sMap)
        print(sanity)
        print(rows*cols, visitedCells)
    return path, success, flagMap, livesLost

# --------------------------------------------------------
def get_score(field, fMap, livesLost = None, verbose = True):

    score = 0
    score = field + fMap
    score[score !=2] = 0
    score = np.sum(score)/2
    pString1 = ''
    if livesLost is not None:
        scoreImpact = -0.5 * livesLost
        pString1 += " Lives Lost: {} = {} impact on score".format(livesLost, scoreImpact)

    pString2 = "Total Score: {}.".format(score)
   
    if verbose:
        print(pString2+pString1)

    return score

# --------------------------------------------------------

def test_random(rows=64,cols =64):

    seedX, seedY = np.random.randint(0, cols), np.random.randint(0,rows)
    print(seedX, seedY)
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
    parser.add_argument('--part', type = str,  metavar = 'dim', default='1',    help="Which part of the assignment to tackle")
    parser.add_argument('--dim', type = int,   metavar = 'dim', default=8,    help="Maze dimension. Maze is Square.")
    parser.add_argument('--n', type = int,   metavar = 'n', default=8,    help="Amount of mines in the field. If more than dimxdim, n is set to dim.")
    parser.add_argument('--trials', type = int,   metavar = 't',default=100   ,help="Number of trilas for the various monte carlo tests.")
    # Parse the input from the console. To access a specific arg-> dim = args.dim
    args = parser.parse_args()
    # Initialize
    # ----------
    np.set_printoptions(precision=4)
    trials = args.trials
    # Generate map
    mapDim     = args.dim
    mineNum   = args.n if args.n <= mapDim**2 else mapDim 
    # Visualize map, save it and hold in mazeImg to use it for drawing the path on.
    field = create_field(mapDim, mineNum)
    fieldImg = draw_field(field, label = 'original')
    # ---|

    # PART 1
    # *******
    np.random.seed(10)
    if '1' in args.part:
        success = 0
        cnt = 0
        plan = {'strategy': 'naive', 'manyLives': mapDim*mapDim}
        while not success and cnt < 50:
            path, success, flagMap, livesLost = minesweeper(field, plan=plan)
            score = get_score(field,flagMap)
            cnt += 1
        if success == 1:
            print("Successfully Sweeped mine field after {} random tries, and losing {} lives!".format(cnt, livesLost))
        elif success == -1:
            print("Exited due to revisiting a cell, after {} random tries, and losing {} lives!".format(cnt, livesLost))
        else:
            print("Failed to sweep the mine field and not explode even after {} random tries!".format(cnt))
        fieldImg = draw_field(field, path = path, fMap = flagMap)
        return 1
    if '2' in args.part:
        fieldSizes = [20]
        mineDensity = [0.1,0.2,0.3,0.4,0.5,0.6]
        mines = np.zeros((len(fieldSizes), len(mineDensity)))
        scores=np.zeros(len(mineDensity))
        plan = {'strategy': 'naive', 'manyLives':20}

        # Monte Carlo Efficiency Loop
        for i,f in enumerate(fieldSizes):
            for j, m in enumerate(mineDensity):
                mines[i,j] =  int(m*f*f)
                for k in range(trials):
                    field = create_field(f,mines[i,j])
                    path, success, flagMap, livesLost = minesweeper(field, plan=plan, verbose = False)
                    scores[j] += get_score(field,flagMap, verbose = False)
                scores[j] /= k*mines[i,j]
                print("Field: {}x{} {} mines, avg {} trial score: {}".format(f,f,mines[i,j],trials, scores[j]))
        # plot graph
        labels = ['Confidence Redistribution']
        for p, m in enumerate(fieldSizes):
            # Plot the suc vs occupanncy prob plot!
            plt.plot(mines[p], scores)              
            plt.title("{}: Score vs Density Plot, dims = {}x{}, trials = {}".format(labels[p], m,m,  trials)) # plot format
            plt.ylabel("Sweeper Score")     # name y axis
            plt.xlabel("Mine Density n")   # name x axis
            plt.savefig('{}_Score_vs_density_dim_{}_trials_{}.png'.format(labels[p], m, trials)) # save figure at the same folder as code
        plt.close()

# =======================================================================================
# MAIN
# Standard way to call a python top level. This essentially says: if this file is called
# from the console or another programm as the top level file, run the main() function
# =======================================================================================
if __name__ == "__main__":
    main()
