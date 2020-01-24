import multiprocessing as mp
from time import sleep

def func():
    count = 1
    with open('test.txt', 'a') as f:
        while True:
            f.write('{}\n'.format(count))
            count +=1

if __name__ == '__main__':
    proc = mp.Process(target=func)
    proc.start()
    sleep(1)
    proc.terminate()