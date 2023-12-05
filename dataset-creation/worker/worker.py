import logging
import multiprocessing
import os
import sys
import time
import traceback

from tqdm import tqdm

from db import DB
from task import Task
from task1_candidate import CandidateTask
from task2_text import TextTask
from task3_aug import AugTask
from utility import get_current_node_cpu_count

task_map = {
    'candidate_search': lambda: CandidateTask(),
    'text_process': lambda: TextTask(),
    'augs': lambda: AugTask()
}


def worker(task_name, input_queue, finished_queue):
    global task_map

    # Clear connection in worker
    DB.disconnect()

    logger = multiprocessing.get_logger()

    task_to_run = task_map[task_name]()  # type: Task
    while True:
        item = input_queue.get()
        if item is None:
            logger.info("Process received None from queue, ending...")
            break

        try:
            task_to_run.process(item)
            logger.info(f"Process has finished an item: {item}")
            finished_queue.put(item)
        except Exception as e:
            logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")


def run_task(task_name: str):
    global task_map
    task_to_run = task_map[task_name]()

    num_workers = get_current_node_cpu_count()
    batch_size = num_workers * 10
    input_queue = multiprocessing.Queue(maxsize=num_workers)
    finished_queue = multiprocessing.Queue()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('logfile.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    workers = []
    for _ in range(num_workers):
        worker_process = multiprocessing.Process(target=worker, args=(task_name, input_queue, finished_queue))
        worker_process.start()
        workers.append(worker_process)

    # Feed the queue until we get an empty batch
    # Feed the queue until we get an empty batch
    while True:
        batch = task_to_run.get_batch(batch_size)
        DB.disconnect()
        if batch:
            for item in batch:
                while input_queue.full():
                    time.sleep(1)  # Sleep while the queue is full
                input_queue.put(item)
                # Process finished
                while not finished_queue.empty():
                    finished = finished_queue.get()
                    if finished:
                        task_to_run.mark_complete(finished)
                        print(f'Marked complete in main: {finished}')
        else:
            print('Empty batch found, so no more items to queue..')
            break

    # Signal to workers that no more tasks will be added
    for _ in range(num_workers):
        input_queue.put(None)

    # Join worker processes (wait for them to finish)
    for worker_process in workers:
        worker_process.join()

    # Process last ones
    while not finished_queue.empty():
        finished = finished_queue.get()
        if finished:
            task_to_run.mark_complete(finished)
            print(f'Marked complete in main: {finished}')

    print(f'All workers for task {task_name} have finished.')


def main(test: bool):
    global task_map

    # Database
    DB.connect()

    # Run available tasks
    task_row = DB.fetch_one("SELECT * FROM tasks WHERE active=1 LIMIT 1")
    if task_row:
        if not test:
            run_task(task_row['name'])
        else:
            # Test a batch
            task_to_run = task_map[task_row['name']]()

            # Get batch of 1
            batch = task_to_run.get_batch(1)
            print(f'Batch: {batch}')
            print()

            DB.connect()

            # Process it
            print(f'Processing..')
            task_to_run.process(batch[0])
            task_to_run.mark_complete(batch[0])
            print(f'Done with test..')
            exit()


if __name__ == "__main__":
    try:
        main('--test' in sys.argv)
        print('Completed task')
    except Exception as _:
        print("Exception in worker:")
        print(traceback.format_exc())

    print("Restarting in 30s...")
    time.sleep(30.0)
    os.execl(sys.executable, sys.executable, *sys.argv)
