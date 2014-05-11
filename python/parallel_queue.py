import multiprocessing as mp


def process_parallel(iter_to_process, transformation, evaluation,
                     num_processes=3):
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
        task_queue.put(elem)

    # Create workers
    workers = []
    for n in xrange(num_processes):
        task_queue.put('STOP')
        p_in, p_out = mp.Pipe()
        worker = Worker(n, task_queue, p_in,
                        transformation,
                        evaluation)
        workers.append((worker, p_out))
        worker.start()

    # Wait until workers finish
    task_queue.join()
    return [pipe.recv() for worker, pipe in workers]


class Worker(mp.Process):
    def __init__(self, _id, queue, pipe,
                 f_transform=lambda x: x,
                 f_evaluate=lambda x: x):
        self._id = _id
        self._queue = queue
        self._pipe = pipe
        self._f_transform = f_transform
        self._f_evaluate = f_evaluate
        mp.Process.__init__(self)

    def run(self):
        output = {}
        for idx, item in iter(self._queue.get, 'STOP'):
            transformed_item = self._f_transform(item)
            output[idx] = self._f_evaluate(transformed_item)
            self._queue.task_done()
        self._queue.task_done()
        self._pipe.send(output)
