import multiprocessing as mp
from multiprocessing import Process, Manager, Lock, Array, Queue
import time
import numpy as np

def test_func(i, k, output, input, lock):

    zZz = np.random.randint(10)

    lock.acquire()
    input[i] = i
    lock.release()

    time.sleep(np.random.randint(zZz))
    lock.acquire()
    output.append(k)
    lock.release()


MAX_PROCESSES = 4
processes = 0
k = 0
max_iters = 20
process_list = [None] * MAX_PROCESSES

# input_list: coordinates of points being currently evaluated. Copy to numpy when entered into function and set value
# of its own point before evaluating function value (after sample next point). Then fix list? Probably, but not
#certain that it works

# output list: all evaluated points, and maybe separate for y-values. Locked after function eval, unlocked after
# entering its value and coords in relevant list

if __name__ == '__main__':

    manager = Manager()
    lock = manager.Lock()
    input_list = manager.list()
    output_list = manager.list()

    for _ in range(4):
        input_list.append(0)



    # Initializing
    k = 0


    # MAIN CODE
    # Repeat for the set number of iterations
    while k < max_iters:

        for i in range(MAX_PROCESSES):
            process_list[i] = Process(target=test_func, args=(i, k, output_list, input_list, lock))
            process_list[i].start()
            print(f'Process {i + 1} started with output {k}')
            k += 1

        for proc in process_list:
            proc.join()








    output_ctr = 0
    for output in output_list:
        output_ctr += 1
    print('Final input:')
    print([input for input in input_list])

    print(f'Final output: {output_ctr} samples.')
    print([output for output in output_list])