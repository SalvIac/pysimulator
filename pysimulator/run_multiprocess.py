# -*- coding: utf-8 -*-
"""
@author: Salvatore
"""

import concurrent.futures
import multiprocessing as mp
import tqdm
import threading
import time
import random
import numpy as np


def run_multiprocess(func, inputs, cores):
    # run with multiprocessing
    with mp.Pool(processes=cores) as p:
        # res = p.map(run_backtest, inputs)
        res = list(tqdm.tqdm(p.imap(func, inputs), total=len(inputs)))
        p.close()
        p.join()
    return res


def rup_multithreading(func, inputs, cores):
    res = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
        res = executor.map(func, inputs, timeout=60)
    #     for inp in tqdm.tqdm(inputs):
    #         future = executor.submit(func, inp)
    #         res.append(future.result())
    return res


def run_multithreading(func, inputs, cores):
    jobs = []
    for inp in range(0, inputs):
        thread = threading.Thread(target=func, args=inp)
        jobs.append(thread)
    for j in tqdm(jobs): # Start the threads
        j.start()
    for j in jobs: # Ensure all of the threads have finished
        j.join()


    
def list_append(args):
    """
    Creates an empty list and then appends a 
    random number to the list 'count' number
    of times. A CPU-heavy operation!
    """
    size = args[0]
    icount = args[1]
    out_list = list()
    for i in range(size):
        out_list.append(random.random())
    return out_list



#%% test

if __name__ == "__main__":
    
    #%% many iterations lasting less than 0.1 set

    # no concurrency

    start_time = time.time()
    output = list()
    for i in tqdm.tqdm(range(0,10000)):
        output.append(list_append([10000, i]))
    print("test1. no concurrency: \t %s seconds" % (time.time() - start_time),
          "test1. result", np.mean(np.array(output)))

   
    # multiprocessing and multithreading
    
    inputs = list()
    for i in range(0,10000):
        inputs.append([10000, i])
    
    start_time = time.time()
    out_list1 = run_multiprocess(list_append, inputs, 4)
    print("test1. multiprocessing imap: \t %s seconds" % (time.time() - start_time),
          "test1. result", np.mean(np.array(out_list1)))
    start_time = time.time()
    out_list2 = rup_multithreading(list_append, inputs, 16)
    print("test1. concurrent ThreadPoolExecutor.submit: \t %s seconds" % (time.time() - start_time),
          "test1. result", np.mean(np.array(out_list2)))
    
    
    
    #%% fewer iterations lasting longer
    
    # no concurrency

    start_time = time.time()
    output = list()
    for i in tqdm.tqdm(range(0,100)):
        output.append(list_append([3000000, i]))
    print("test2. no concurrency: \t %s seconds" % (time.time() - start_time),
          "result", np.mean(np.array(output)))

   
    # multiprocessing and multithreading
    
    inputs = list()
    for i in range(0,100):
        inputs.append([3000000, i])
    
    start_time = time.time()
    out_list1 = run_multiprocess(list_append, inputs, 4)
    print("test2. multiprocessing imap: \t %s seconds" % (time.time() - start_time),
          "test2. result", np.mean(np.array(out_list1)))
    start_time = time.time()
    out_list2 = rup_multithreading(list_append, inputs, 16)
    print(np.mean(np.array(out_list2)))
    print("test2. concurrent ThreadPoolExecutor.submit: \t %s seconds" % (time.time() - start_time),
          "test2. result", np.mean(np.array(out_list2)))  
    
    
    
    
    
    
    
    
    
    