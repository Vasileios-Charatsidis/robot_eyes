import multiprocessing as mp


def process_parallel(iter_to_process, evaluation_function,
                     num_processes=3, **kwargs):
    '''
    Given an iterable with objects to process (as tuples, lists, or
    simply elements), set up a queue for parallel processing through
    RANSAC. Each object is transformed by a given function
    'transformation', the result is then evaluated using a  function
    'evaluation'.

    Returns the outputs for all workers.

    Tried to implement this as general as possible, so that I may
    reuse it later.
    '''
    # Limit the number of processes to the number of items
    num_processes = max(num_processes, len(iter_to_process))

    # Fill the queue for all tasks
    task_queue = mp.JoinableQueue()
    for idx, elem in enumerate(iter_to_process):
        if not isinstance(elem, tuple) and \
                not isinstance(elem, list):
            elem = [elem]
        task_queue.put((idx, elem))

    # Create workers
    workers = []
    for n in xrange(num_processes):
        task_queue.put('STOP')
        p_in, p_out = mp.Pipe()
        worker = Worker(n, task_queue, p_in,
                        evaluation_function,
                        **kwargs)
        workers.append((worker, p_out))
        worker.start()

    # Wait until workers finish
    task_queue.join()
    output = {}
    for worker, pipe in workers:
        output.update(pipe.recv())
    return[v for k, v in sorted(output.iteritems())]


class Worker(mp.Process):
    def __init__(self, _id, queue, pipe,
                 evaluation_function=lambda x: x,
                 **kwargs):
        self._id = _id
        self._queue = queue
        self._pipe = pipe
        self._evaluate = evaluation_function
        # Store additional kwargs directly
        self.additional_vars = kwargs
        mp.Process.__init__(self)

    def run(self):
        output = {}
        for idx, item in iter(self._queue.get, 'STOP'):
            # Notice that this construction allows adding other data
            # without putting these in the queue itself.
            output[idx] = self._evaluate(*item,
                                         **self.additional_vars)
            self._queue.task_done()
        self._queue.task_done()
        self._pipe.send(output)
