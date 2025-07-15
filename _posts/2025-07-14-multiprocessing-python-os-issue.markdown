---
layout: post
title:  "Python multiprocessing - why it works in Linux and not in MacOS"
date:   2025-07-16 18:50:11 +0530
categories: software-engineering
---

Sometimes back while I was working with the multiprocessing library in Python on my MacOS, I encountered a strange issue in my code where I was getting an error that reads something like (I have replaced the function name here with something called as `square`):<br/><br/>
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
The above code is an over-simplification of the actual code used and is just shown for demonstrating purposes. The basic idea is that I have a function `fn` and inside that I am defining a `producer` and a `consumer` which are nested functions. Then I am creating a new process for the producer which takes input as the input multiprocessing queue and populates it with integers from 0 to 9. On the other hand I am creating 4 consumer processes and each one is reading an integer from the input queue and writing to an output queue the square of the input values. This is a classic producer-consumer example in multiprocessing. Note that each producer and consumers are separate processes from the main process.<br/><br/>
The above code when run on an Ubuntu server was running perfectly fine whereas when running on MacOS was giving the error show at the top. So what is happening here ?<br/><br/>
The error implies that Python is not able to pickle the `producer` nested function defined inside the `fn` function. But why is pickle coming to the picture ?
In python multiprocessing, whenever a new child process is created by the parent process, the target functions and the arguments are passed to each child process by pickling (serializing) the function and the arguments. In each of the child process, the function and arguments are deserialized. But note that, during pickling only the module names and the function names are serialized and not the actual function contents. Thus, when the object is deserialized by the child process, it sees only the module names and function names and imports the necessary function from the corresponding module in its path.
But in the above code, 


