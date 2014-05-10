import random
from BoilerPlate import *
d1 = [2,2,4,4,9,9]
d2 = [1,1,6,6,8,8]
d3 = [3,3,5,5,7,7]

turns = 100

gtt = list()
counts = [0.0,0.0,0.0]
for i in range(0,turns):
    
    roll1 = random.sample(d1,1)
    roll2 = random.sample(d2,1)
    roll3 = random.sample(d3,1)
    
    roll = [roll1,roll2,roll3]
    
    gt = [float(roll1>roll2), float(roll2>roll3), float(roll3> roll1)]
    print roll
    print gt
    for i in range(0,3):
        counts[i]+=gt[i]
#     gtt.append( gt)


print np.array(counts)/turns
# df = pd.DataFrame(gtt)
# print df.head()

# print df.sum(axis = 0)/turns
    
     