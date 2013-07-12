import cProfile
import re
import pstats

import GridTesterMA as gt

#cProfile.run('gt.grid_tester()','grdStats')

#Then you can use runsnakerun via terminal: runsnake grdStats

p = pstats.Stats('grdStats')
#p.strip_dirs().sort_stats(-1).print_stats()
#p.sort_stats('cumulative').print_stats(10)
p.sort_stats('time').print_stats(10)