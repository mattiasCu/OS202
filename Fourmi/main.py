import numpy as np
import myPheromone
import myAnts
from mpi4py import MPI
import pygame as pg
import myMaze
import sys
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def loadAssets():
    #  Load patterns for maze display :
    global cases_img
    cases_img = []
    img = pg.image.load("img/cases.png").convert_alpha()
    for i in range(0, 128, 8):
        cases_img.append(pg.Surface.subsurface(img, i, 0, 8, 8))

    # Load sprites for ants display :
    global sprites 
    sprites = []
    img = pg.image.load("img/ants.png").convert_alpha()
    for i in range(0, 32, 8):
        sprites.append(pg.Surface.subsurface(img, i, 0, 8, 8))


def displayMaze(maze):
    """
    Create a picture of the maze :
    """
    maze_img = pg.Surface((8*maze.shape[1], 8*maze.shape[0]), flags=pg.SRCALPHA)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            maze_img.blit(cases_img[maze[i, j]], (j*8, i*8))

    return maze_img


def displayAnts(ants, screen):
        [screen.blit(sprites[ants.directions[i]], (8*ants.historic_path[i, ants.age[i], 1], 8*ants.historic_path[i, ants.age[i], 0])) for i in range(ants.directions.shape[0])]



def getColor(pheromon, i: int, j: int):
    val = max(min(pheromon[i, j], 1), 0)
    return [255*(val > 1.E-16), 255*val, 128.]

def displayPheromon(pheromon, screen):
    [[screen.fill(getColor(pheromon, i, j), (8*(j-1), 8*(i-1), 8, 8)) for j in range(1, pheromon.shape[1]-1)] for i in range(1, pheromon.shape[0]-1)]


if __name__ == "__main__":
    

    if rank ==0:
        pg.init()


        size_maze = 25, 25
        if len(sys.argv) > 2:
            size_maze = int(sys.argv[1]),int(sys.argv[2])

        max_life = 500
        if len(sys.argv) > 3:
            max_life = int(sys.argv[3])
        
        alpha = 0.9
        if len(sys.argv) > 4:
            alpha = float(sys.argv[4])

        beta  = 0.99
        if len(sys.argv) > 5:
            beta = float(sys.argv[5])

        #send parameters
        comm.bcast((size_maze, max_life, alpha, beta), root=0)

        # screen init
        resolution = size_maze[1]*8, size_maze[0]*8
        screen = pg.display.set_mode(resolution)

        # load assets
        loadAssets()

        # maze init
        a_maze = myMaze.Maze(size_maze, 12345)
        comm.bcast(a_maze.maze, root = 0)
        mazeImg = displayMaze(a_maze.maze)


        playing = True
        snapshop_taken = False
        ants = [0]*(size-1)
        zeros = np.zeros((size_maze[0]+2, size_maze[1]+2),  dtype=np.double)


        
        while playing:


            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    playing = False
                    
            #communication
            ants = comm.gather(None, root = 0)
            food_counter = sum([ants[k][1] for k in range(1,size)])
            pherom = comm.reduce(zeros, op = MPI.SUM, root = 0)
            comm.bcast(playing, root = 0)

            #display
            displayPheromon(pherom,screen)
            screen.blit(mazeImg, (0, 0))
            for k in range(1,size):
                displayAnts(ants[k][0], screen)
            pg.display.update()

            #save img
            if food_counter == 1 and not snapshop_taken:
                pg.image.save(screen, "img/MyFirstFood.png")
                snapshop_taken = True
            
            if not playing :
                pg.quit()
        
    else:

        if rank == size-1:
            print("\r\n")
            print("sum of the processus : ", size-1)
        
        size_maze, max_life, alpha, beta =comm.bcast(None, root=0)
    
        nb_ants = size_maze[0]*size_maze[1]//4
        nb_ants //= size

        the_food_position = size_maze[0]-1, size_maze[1]-1
        pos_nest = 0, 0

        maze = comm.bcast(None, root = 0)
        ants = myAnts.Colony(nb_ants, pos_nest, max_life)
        pheromones = myPheromone.Pheromon(size_maze, the_food_position, alpha, beta)
        
        food_counter = 0
        playing = True
        

        first_check = True
        second_check = True
        deb_main = time.time()

        FPS_avg = 0

        while playing:

            
            #compute
            deb = time.time()
            food_counter = ants.advance(maze, the_food_position, pos_nest, pheromones, food_counter)
            pheromones.do_evaporation(the_food_position)
            end = time.time()

            #communication
            comm.gather((ants, food_counter), root = 0)
            comm.reduce(pheromones.pheromon, op = MPI.SUM, root = 0)
            playing = comm.bcast(None, root = 0)


            if food_counter >= 1 and food_counter < 499:
                if first_check==True:
                    fin_first_food = time.time()
                    first_check = False
                FPS_avg += 1./(end-deb)
                print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}, temps pour première nouriture : {fin_first_food - deb_main}", end='\r')
            elif food_counter >= 500:
                if second_check==True:
                    fin_second_food = time.time()
                    second_check = False
                FPS_avg += 1./(end-deb)
                FPS_avg /= 500
                print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}, temps pour première nouriture : {fin_first_food - deb_main}, temps pour 500 nouriture : {fin_second_food - deb_main}\n")
                #print(f"FPS moyen : {FPS_avg:6.2f}")
                exit(0)
            else:
                print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", end='\r')

            
            


        
        
        
