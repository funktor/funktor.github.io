---
layout: post
title:  "Python multiprocessing - why it works in Linux and not in MacOS"
date:   2025-07-16 18:50:11 +0530
categories: software-engineering
---

Sometimes back while I was working with the multiprocessing library in Python on my MacOS, I encountered a strange issue in my code where I was getting an error that reads something like (I have replaced the actual function names here with `producer` and `consumer`):<br/><br/>
  ```
  AttributeError: Can't pickle local object 'fn.<locals>.producer'
  ```
  <br/><br/>
The code which was producing the above error looks something like the one below:<br/><br/>
  ```python
  import multiprocessing
  import time
  
  def fn():
    def producer(q):
      for i in range(10):
         q.put(i)
      q.put(None)
    
    def consumer(inp_q, out_q):
      while True:
          try:
            item = inp_q.get(timeout=1.0)
            if item is None:
                break
            out_q.put(item*item)
          except:
            break
  
    inp_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
  
    prod_p = multiprocessing.Process(target=producer, args=(inp_queue,), daemon=True)
    prod_p.start()
  
    cons_p = []
  
    for _ in range(4):
      q = multiprocessing.Process(target=consumer, args=(inp_queue, out_queue), daemon=True)
      q.start()
      cons_p += [q]
  
    prod_p.join()
  
    for cq in cons_p:
      cq.join()
    
  if __name__ == "__main__":
    fn()
  ```
  <br/><br/>
The above code is an over-simplification of the actual code used and is just shown for demonstration purposes. The basic idea is that I have a function `fn` and inside that I am defining a `producer` and a `consumer` which are nested functions. Then I am creating a new process for the producer which takes input as the input multiprocessing queue and populates it with integers from 0 to 9. On the other hand I am creating 4 consumer processes and each one is reading an integer from the input queue and writing to an output queue the square of the input values. This is a classic producer-consumer example in multiprocessing. Note that each producer and consumers are separate processes from the main process.<br/><br/>
The above code when run on an Ubuntu server was running perfectly fine whereas when running on MacOS was giving the error shown at the top. So what is happening here ?<br/><br/>
The error implies that Python is not able to pickle the `producer` nested function defined inside the `fn` function. But why is pickle coming into the picture ?<br/><br/>
[Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html)<br/><br/>
In python multiprocessing, whenever a new child process is created by the parent process, the target functions and the arguments are passed to each child process by pickling (serializing) the function and the arguments. In each of the child process, the function and arguments are then deserialized. But note that, during pickling only the module names and the function names are serialized and not the actual function contents. Thus, when the object is deserialized by the child process, it sees only the module names and function names and imports the necessary function from the corresponding module in its path.<br/><br/>
But in the above code, the nested functions `producer` and `consumer` are not visible at the module level and is only visible from within the scope of `fn` function. Thus, serialization and deserialization of the nested functions are not possible using pickle.<br/><br/>
But then why the above code should work for Ubuntu ?<br/><br/>
The answer lies in the fact that in multiprocessing, child processes are created by 3 different mechanisms, `fork`, `spawn` and `forkserver`. We will only focus on the 1st two as these are the most widely used. In Ubuntu, the default mechanism for multiprocessing is through `fork`, whereas in Windows and MacOS, the default mechanism is `spawn`.<br/><br/>
[multiprocessing fork() vs spawn()](https://stackoverflow.com/questions/64095876/multiprocessing-fork-vs-spawn)<br/><br/>
[Python forkserver and set_forkserver_preload()](https://bnikolic.co.uk/blog/python/parallelism/2019/11/13/python-forkserver-preload.html)<br/><br/>
Whenever a child process is `forked`, the child process inherits almost all of parent process' objects and only when an object is modified inside the child process, a copy of that object is created. This is known as `copy-on-write` mechanism. Now if the original object is not modified (read-only) by child processes, then there is no need for keeping a separate copy and thus there is no need for pickling the objects by the parent process and sending them over to the child processes. When using the `fork` mechanism in the above code, the `producer` and `consumer` nested functions are not pickled and the child processes refers to the parent process' object in memory.<br/><br/>
[Copy-On-Write](https://en.wikipedia.org/wiki/Copy-on-write)<br/><br/>
On the other hand, when a child process is `spawned`, a completely new copy of the object in memory is held with the child process separate from the one held by the parent process. Thus, this requires serialization and deserialization with pickle and hence when `spawn` mechanism is used, we will get the above error. To overcome the issue in MacOS, we just need to change the start_method at the top of the file by adding a line:
  ```python
  multiprocessing.set_start_method('fork', force=True)
  ```
  <br/><br/>
This fact can be verified by printing the function id of the parent process functions and the child process functions as shown below (after adding the line above):<br/><br/>
  ```python
  def fn():
    def producer(q):
      print("Child Process Producer id = ", id(producer))
      for i in range(10):
         q.put(i)
      q.put(None)
    
    def consumer(inp_q, out_q):
      print("Child Process Consumer id = ", id(consumer))
      while True:
          try:
            item = inp_q.get(timeout=1.0)
            if item is None:
                break
            out_q.put(item*item)
          except:
            break
          
    
    print("Main Process Producer id = ", id(producer))
    print("Main Process Consumer id = ", id(consumer))
  
    inp_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
  
    prod_p = multiprocessing.Process(target=producer, args=(inp_queue,), daemon=True)
    prod_p.start()
  
    cons_p = []
  
    for _ in range(4):
      q = multiprocessing.Process(target=consumer, args=(inp_queue, out_queue), daemon=True)
      q.start()
      cons_p += [q]
  
    prod_p.join()
  
    for cq in cons_p:
      cq.join()
  ```
  <br/><br/>
If we run the above code, we will see an ouput something like:
  ```
  Main Process Producer id  =  4340068656
  Main Process Consumer id  =  4341349776
  Child Process Producer id =  4340068656
  Child Process Consumer id =  4341349776
  Child Process Consumer id =  4341349776
  Child Process Consumer id =  4341349776
  Child Process Consumer id =  4341349776
  ```
  <br/><br/>
As you can see that for both the parent and child process, the same `producer` function is used and similarly for the parent and all 4 child processes, the same `consumer` function is used.<br/><br/> 
But instead of setting the default start_method for multiprocessing to `fork`, if one wishes to run the above program using `spawn` in MacOS or Windows, they need to move the functions `producer` and `consumer` outside `fn` as shown below.<br/><br/>
 ```python
  def producer(q):
    print("Child Process Producer id = ", id(producer))
    for i in range(10):
       q.put(i)
    q.put(None)
  
  def consumer(inp_q, out_q):
    print("Child Process Consumer id = ", id(consumer))
    while True:
        try:
          item = inp_q.get(timeout=1.0)
          if item is None:
              break
          out_q.put(item*item)
        except:
          break

  def fn():
    print("Main Process Producer id = ", id(producer))
    print("Main Process Consumer id = ", id(consumer))
  
    inp_queue = multiprocessing.Queue()
    out_queue = multiprocessing.Queue()
  
    prod_p = multiprocessing.Process(target=producer, args=(inp_queue,), daemon=True)
    prod_p.start()
  
    cons_p = []
  
    for _ in range(4):
      q = multiprocessing.Process(target=consumer, args=(inp_queue, out_queue), daemon=True)
      q.start()
      cons_p += [q]
  
    prod_p.join()
  
    for cq in cons_p:
      cq.join()
  ```
  <br/><br/>
If we print the function ids now, the function ids will be different in main process and child processes.<br/><br/>
Another thing to note is that, if instead of `multiprocessing.Process`, one uses `multiprocessing.Pool` to solve the above problem as shown below:
  ```python
  def fn():
    def square(x):
      time.sleep(1.0)
      return x*x
    
    with multiprocessing.Pool(processes=4) as pool:
      results = pool.map(square, range(10))
  ```
  <br/><br/>
We will get the same error irrespective of whether we set `fork` or `spawn` in the set_start_method as shown above. This is because of how Pool is implemented. It always pickles the function and the arguments to pass to the child processes. Thus we will get the error all the time if the multiprocessing is implemented using Pool. The solution is to define the function `square` outside `fn`. <br/><br/>
But why Windows and MacOS have moved the default mechanism for multiprocessing to `spawn` from `fork` ? Nice reads from the following blogs.<br/><br/>
[Python’s multiprocessing performance problem](https://pythonspeed.com/articles/faster-multiprocessing-pickle/)<br/><br/>
[Why your multiprocessing Pool is stuck](https://pythonspeed.com/articles/python-multiprocessing/)<br/><br/>
[The Power and Danger of os.fork](https://medium.com/@tmrutherford/the-default-method-of-spawning-processes-on-linux-is-changing-in-python-3-14-heres-why-b9711df0d1b1)<br/><br/>


