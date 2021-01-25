
import numpy as np
import os.path
import matplotlib.pyplot as plt


def move_still_possible(S):
    return not (S[S==0].size == 0)


def move_at_random(S, p):
    xs, ys = np.where(S==0)

    i = np.random.permutation(np.arange(xs.size))[0]
    
    S[xs[i],ys[i]] = p

    return S


def move_by_probabilistic_strategy(S, p):

    sfile = open('strategy.txt','r+')
    sdata = np.zeros((3,3), dtype=float)

    # load sdata from file
    if p==1:
        for i in range(5): line = sfile.readline()
        for i in range(3):
            sdata[i] = line.split()
            for j in sdata[i]: j = float(j)
            line = sfile.readline()
    if p==-1:
        for i in range(9): line = sfile.readline()
        for i in range(3):
            sdata[i] = line.split()
            for j in sdata[i]: j = float(j)
            line = sfile.readline()

    # update and normalize sdata
    for i in range(3):
        for j in range(3):
            if S[i][j] != 0: sdata[i][j] = 0
    sdata = sdata/sdata.sum()

    # get a random move accroding to the probabilities
    index = np.random.choice([0,1,2,3,4,5,6,7,8], p=sdata.flatten())
    S[int(index/3)][index%3] = p
    return S


def move_by_heuristic_strategy(S, p):

    sfile = open('strategy.txt','r+')
    sdata = np.zeros((3,3), dtype=float)

    # load sdata from file
    if p==1:
        for i in range(5): line = sfile.readline()
        for i in range(3):
            sdata[i] = line.split()
            for j in sdata[i]: j = float(j)
            line = sfile.readline()
    if p==-1:
        for i in range(9): line = sfile.readline()
        for i in range(3):
            sdata[i] = line.split()
            for j in sdata[i]: j = float(j)
            line = sfile.readline()

    # update sdata
    for i in range(3):
        for j in range(3):
            if S[i][j] != 0: sdata[i][j] = 0

    # get the best move
    index = np.where(sdata == np.max(sdata))
    S[index[0][0]][index[1][0]] = p
    return S

def move_was_winning_move(S, p):
    if np.max((np.sum(S, axis=0)) * p) == 3:
        return True

    if np.max((np.sum(S, axis=1)) * p) == 3:
        return True

    if (np.sum(np.diag(S)) * p) == 3:
        return True

    if (np.sum(np.diag(np.rot90(S))) * p) == 3:
        return True

    return False

def draw_bar(D):
    xticks = ['X win', 'O win', 'Draw']
    plt.bar(range(3), [D.get(xtick, 0) for xtick in xticks], align='center',yerr=0.000001)
    plt.xticks(range(3), xticks)
    plt.ylabel('count')
    plt.title('winer in 10000 games')
    plt.show()

# python dictionary to map integers (1, -1, 0) to characters ('x', 'o', ' ')
symbols = { 1:'x',
           -1:'o',
            0:' '}

# print game state matrix using characters
def print_game_state(S):
    B = np.copy(S).astype(object)
    for n in [-1, 0, 1]:
        B[B==n] = symbols[n]
    print (B)

# write game state matrix using characters into log file
def write_game_state(S, f):
    for i in range (3):
        for j in range (3):
            f.write('[%c]'%symbols[S[i][j]])
        f.write('\n')
  
# check whether need to create a strategy file
def need_new_strategy_file():
    sfile = 'strategy.txt';
    # check if probabilistic strategy file exist
    if os.path.exists(sfile):
        sz = os.path.getsize(sfile)
        if sz:
            neednew = input('Strategy file exists - strategy.txt\nDo you want to create a new one? (Y: yes, else: no)')
            if neednew == 'Y' or neednew == 'y':
                print ('Generating strategy file...')
                return True
            else:
                return False
        else:
            print ('No strategy file, please create a strategy file.')
            print ('Generating strategy file, please wait...')
            return True
    else:
        print ('No strategy file, please create a strategy file.')
        print ('Generating strategy file, please wait...')
        return True

# create new strategy file
def create_new_strategy_file():

    trainLog = open('training_log.txt', 'w')
    strategyFile = open('strategy.txt', 'w')

    winCounter = {'X win':0,'O win':0,'Draw':0}# x-win, o-win, draw
    xPro = np.zeros((3,3), dtype=float)
    oPro = np.zeros((3,3), dtype=float)

    for i in range(1,10001):
        trainLog.write('game %d\n' %i)

        # initialize an empty tic tac toe board
        gameState = np.zeros((3,3), dtype=int)

        # initialize the player who moves first (either +1 or -1)
        player = 1

        # initialize a move counter
        mvcntr = 1

        # initialize a flag that indicates whether or not the game has ended
        noWinnerYet = True

    
        while move_still_possible(gameState) and noWinnerYet:
            # turn current player number into player symbol
            name = symbols[player]
            trainLog.write('%s moves\n' % name)

            # let current player move at random
            gameState = move_at_random(gameState, player)

            # write current game state to log file
            write_game_state(gameState, trainLog)
        
            # evaluate current game state
            if move_was_winning_move(gameState, player):
                trainLog.write ('player %s wins after %d moves\n' % (name, mvcntr))
                noWinnerYet = False
                # save win data
                if player==1:
                    for i in range(0,3):
                        for j in range(0,3):
                             xPro[i][j] += (gameState[i][j]==1)
                    winCounter['X win'] += 1
                if player==-1:
                    for i in range(0,3):
                        for j in range(0,3):
                             oPro[i][j] += (gameState[i][j]==-1)
                    winCounter['O win'] += 1

            # switch current player and increase move counter
            player *= -1
            mvcntr +=  1



        if noWinnerYet:
            trainLog.write ('game ended in a draw\n')
            winCounter['Draw'] += 1
    
    # save data
    strategyFile.write ('X win: %d\n' %winCounter['X win'])
    strategyFile.write ('O win: %d\n' %winCounter['O win'])
    strategyFile.write ('Draw: %d\n' %winCounter['Draw'])
    strategyFile.write ('X win probabilistic map:\n')
    for i in range(3):
        for j in range(3):
            xPro[i][j] /= 10000
            strategyFile.write ('%.4f      '%xPro[i][j])
        strategyFile.write ('\n')
    strategyFile.write ('O win probabilistic map:\n')
    for i in range(3):
        for j in range(3):
            oPro[i][j] /= 10000
            strategyFile.write ('%.4f      '%oPro[i][j])
        strategyFile.write ('\n')
    
    trainLog.close()
    strategyFile.close()

    draw_bar(winCounter)
    print ('Srategy file created, training log training_log.txt')

# X play with probabilistic strategy
def play_with_probabilistic_strategy():

    gameLog = open('probabilistic_strategy_game_log.txt', 'w')
    resultFile = open('probabilistic_strategy_result.txt', 'w')

    winCounter = {'X win':0,'O win':0,'Draw':0}    # x-win, o-win, draw

    for i in range(1,10001):
        gameLog.write('game %d\n' %i)

        # initialize an empty tic tac toe board
        gameState = np.zeros((3,3), dtype=int)

        # initialize the player who moves first (either +1 or -1)
        player = 1

        # initialize a move counter
        mvcntr = 1

        # initialize a flag that indicates whether or not the game has ended
        noWinnerYet = True

    
        while move_still_possible(gameState) and noWinnerYet:
            # turn current player number into player symbol
            name = symbols[player]
            gameLog.write('%s moves\n' % name)

            # let current player move at random
            if (player == 1):
                gameState = move_by_probabilistic_strategy(gameState, player)
            else:
                gameState = move_at_random(gameState, player)

            # write current game state to log file
            write_game_state(gameState, gameLog)
        
            # evaluate current game state
            if move_was_winning_move(gameState, player):
                gameLog.write ('player %s wins after %d moves\n' % (name, mvcntr))
                noWinnerYet = False
                # save win data
                if player==1:
                    winCounter['X win'] += 1
                if player==-1:
                    winCounter['O win'] += 1

            # switch current player and increase move counter
            player *= -1
            mvcntr +=  1

        if noWinnerYet:
            gameLog.write ('game ended in a draw\n')
            winCounter['Draw'] += 1
    

    resultFile.write ('X win: %d\n' %winCounter['X win'])
    resultFile.write ('O win: %d\n' %winCounter['O win'])
    resultFile.write ('Draw: %d\n' %winCounter['Draw'])
    gameLog.close()
    resultFile.close()
    
    draw_bar(winCounter)
    print ('Game finished, game log: probabilistic_strategy_game_log.txt')
   
# X play with heuristic strategy
def play_with_heuristic_strategy():

    gameLog = open('heuristic_strategy_game_log.txt', 'w')
    resultFile = open('heuristic_strategy_result.txt', 'w')

    winCounter = {'X win':0,'O win':0,'Draw':0}    # x-win, o-win, draw

    for i in range(1,10001):
        gameLog.write('game %d\n' %i)

        # initialize an empty tic tac toe board
        gameState = np.zeros((3,3), dtype=int)

        # initialize the player who moves first (either +1 or -1)
        player = 1

        # initialize a move counter
        mvcntr = 1

        # initialize a flag that indicates whether or not the game has ended
        noWinnerYet = True

    
        while move_still_possible(gameState) and noWinnerYet:
            # turn current player number into player symbol
            name = symbols[player]
            gameLog.write('%s moves\n' % name)

            # let current player move at random
            if (player == 1):
                gameState = move_by_heuristic_strategy(gameState, player)
            else:
                gameState = move_at_random(gameState, player)

            # write current game state to log file
            write_game_state(gameState, gameLog)
        
            # evaluate current game state
            if move_was_winning_move(gameState, player):
                gameLog.write ('player %s wins after %d moves\n' % (name, mvcntr))
                noWinnerYet = False
                # save win data
                if player==1:
                    winCounter['X win'] += 1
                if player==-1:
                    winCounter['O win'] += 1

            # switch current player and increase move counter
            player *= -1
            mvcntr +=  1

        if noWinnerYet:
            gameLog.write ('game ended in a draw\n')
            winCounter['Draw'] += 1
    

    resultFile.write ('X win: %d\n' %winCounter['X win'])
    resultFile.write ('O win: %d\n' %winCounter['O win'])
    resultFile.write ('Draw: %d\n' %winCounter['Draw'])
    gameLog.close()
    resultFile.close()
    
    draw_bar(winCounter)
    print ('Game finished, game log: heuristic_strategy_game_log.txt')

if __name__ == '__main__':

    # check if need a new strategy file
    needNew = need_new_strategy_file()
    if (needNew):
        create_new_strategy_file()

    print ('Using strategy file strategy.txt')
    
    p = input("Do you X to play with probabilistic strategy? (Y: yes, else: no)")
    if p=='y' or p=='Y':
        print ('Playing with probabilistic strategy, please wait...')
        play_with_probabilistic_strategy()
    h = input("Do you X to play with heuristic strategy? (Y: yes, else: no)")
    if h=='y' or h=='Y':
        print ('Playing with heuristic strategy, please wait...')
        play_with_heuristic_strategy()
    print ("Game finished, thank you!")

'''
    # initialize an empty tic tac toe board
    gameState = np.zeros((3,3), dtype=int)

    # initialize the player who moves first (either +1 or -1)
    player = 1

    # initialize a move counter
    mvcntr = 1

    # initialize a flag that indicates whether or not the game has ended
    noWinnerYet = True


    while move_still_possible(gameState) and noWinnerYet:
        # turn current player number into player symbol
        name = symbols[player]
        #trainLog.write('%s moves\n' % name)
        print ('%s moves' % name)

        # let current player move at random
        gameState = move_at_random(gameState, player)

        # print current game state
        print_game_state(gameState)
        
        # evaluate current game state
        if move_was_winning_move(gameState, player):
            print ('player %s wins after %d moves' % (name, mvcntr))
            noWinnerYet = False

        # switch current player and increase move counter
        player *= -1
        mvcntr +=  1



    if noWinnerYet:
        print ('game ended in a draw' )
'''
