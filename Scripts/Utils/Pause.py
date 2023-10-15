import time, argparse

'''
Simple pause function for specified number of seconds.
'''

# parse
parser = argparse.ArgumentParser()
parser.add_argument("--s", type=int, default=1, help="number of seconds")
opt = parser.parse_args()

# pause
print('Pausing for {0} seconds...'.format(opt.s))
time.sleep(opt.s)
