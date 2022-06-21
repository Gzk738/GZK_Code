
import random
p=(0,1,2)
class Player:
    def __init__(self):
        self.strategy=random.choice(p)
        self.payoff=0.
        self.partner=0
        self.done=False
        self.segmented=False
        self.away=False


#parameter setting
g_size=50
n_group=1
t_round=100
mu=0.01


#initializing

player=list(range(g_size))


def matching():
    temp=list(range(g_size))
    n=g_size

    for i in range(g_size):
     player[i].done=False

    for i in range(g_size):
      if player[i].done==False:
         del temp[0]
         n=n-1
         x=random.randrange(0,n)
         player[i].partner=temp[x]
         player[temp[x]].partner=i
         player[i].done=True
         player[temp[x]].done=True
         del temp[x]
         n=n-1


# playing the game
def play():
    for i in range(g_size):
        player[i].done = False
        
    for i in range(g_size):
        if player[i].done == False:
            x = player[i].partner
            if player[i].strategy ==0:
                if player[x].strategy ==0:
                    player[i].payoff = 0
                    player[x].payoff = 0
                elif player[x].strategy ==1:
                    player[i].payoff = -1
                    player[x].payoff = 1
                else:
                    player[i].payoff = 1
                    player[x].payoff = -1
                    
            elif player[i].strategy == 1:
                if player[x].strategy == 0:
                    player[i].payoff = 1
                    player[x].payoff = -1
                elif player[x].strategy == 0:
                    player[i].payoff = 0
                    player[x].payoff = 0
                else:
                    player[i].payoff = -1
                    player[x].payoff = 1
                    
            else:
                if player[x].strategy == 0:
                    player[i].payoff = -1
                    player[x].payoff = 1
                elif player[x].strategy == 1:
                    player[i].payoff = 1
                    player[x].payoff = -1
                else:
                    player[i].payoff = 0
                    player[x].payoff = 0
                    
            player[i].done = True
            player[x].done = True
            
    for i in range(g_size):
        print(i, player[i].payoff)


def updating():

    for i in range(g_size):
        if random.random() < mu:
            player[i].strategy = random.choice(p)
        else:
            x = random.randrange(0, g_size)
            if player[i].payoff < player[x].payoff:
                player[i].strategy = player[x].strategy

    for i in range(g_size):
        print(i, player[i].strategy)


    
for i in range(g_size):
  player[i]=Player()


for r in range(t_round):
    matching()
    play()
    updating()


