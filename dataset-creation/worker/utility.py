import multiprocessing
import os


def get_current_node_cpu_count():
    try:
        nodes = int(os.getenv('SLURM_CPUS_ON_NODE'))
        print(f'SLURM CPU NODES: {nodes}')
        return nodes
    except:
        return multiprocessing.cpu_count()
    # # Get the hostname
    # try:
    #     hostname = subprocess.check_output('hostname', shell=True).decode().strip()
    #     node_name = hostname.split('.')[0]
    #
    #     # Run the squeue command and get its output
    #     squeue_output = subprocess.check_output('squeue --me --format %N-%C', shell=True).decode().strip()
    #
    #     # Split the output into lines
    #     lines = squeue_output.split('\n')
    #
    #     # Iterate over the lines to find the one for the current node
    #     for line in lines:
    #         if line.startswith(node_name):
    #             # The line for the current node has been found,
    #             # extract the CPU count and return it
    #             cpu_count = line.split('-')[1]
    #             return int(cpu_count)
    # except Exception as e:
    #     # Log the exception for debugging
    #     print(f'Failed to get CPU count from SLURM, falling back to multiprocessing: {e}')
    #
    # # If the function hasn't returned yet, it means the current node wasn't found in the squeue output
    # # or some other error occurred, so return the multiprocessing CPU count
    # return multiprocessing.cpu_count()
