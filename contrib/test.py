import argparse
import os
parser = argparse.ArgumentParser("test.py")

parser.add_argument('-v', '--version', nargs='?', type=str, choices=['ebbrt', 'linux'], default='ebbrt', help='version')
parser.add_argument('-d', '--dir', nargs='?', type=str, required=True, help='results directory')
parser.add_argument('-r', '--repetitions', nargs='?', type=int, default=1, help='number of repetitions')
parser.add_argument('-i', '--iterations', nargs='*', required=True, default=[], type=int, help='number of iterations')
parser.add_argument('-t', '--threads', nargs='*', required=True, default=[], type=int, help='number of threads')
parser.add_argument('-n', '--nodes', nargs='*', required=True, default=[], type=int, help='number of backend nodes')
parser.add_argument('-s', '--datasets', nargs='*', required=True, default=[], type=str, choices=['small', 'large'], help='datasets')
parser.add_argument('-c', '--complete',nargs='?', const=True, default=False, type=bool, help='complete missing repetitions')
parser.add_argument('--test',nargs='?', const=True, default=False, type=bool, help='debug test without actually running it')

args = parser.parse_args()
params = vars(args)

DIR = params['dir']
NODES = params['nodes']
ITERATIONS = params['iterations']
THREADS = params['threads']
REPETITIONS = params['repetitions']
DATASETS = params['datasets']
VERSION = params['version']


def parse_tmp(dataset, threads, iterations, nodes, filename):
    tmp = open('tmp', 'r')
    if VERSION == 'ebbrt':
        csv = '{},{},{},{},'.format(dataset, threads,iterations,nodes)
    else:
        csv = '{},{},{},'.format(dataset, threads,iterations)
    result_file = open(filename, 'a')
    for line in tmp:
        line = line.rstrip()
        if line:
            s = line.split();
            csv = csv + s[-1] + ","
    csv = csv[:-1]
    result_file.write(csv + "\n")
    result_file.close()


def run_test(dataset, threads, iterations, nodes, filename):
    if VERSION == 'ebbrt':
        root_dir = './'
        cmd = 'ROOT_DIR={} {}/contrib/{}.sh {} {} {} {}'.format(root_dir, root_dir,dataset,threads+1, iterations, nodes, nodes+1)
    else:
        root_dir = 'ext/fetalReconstruction/'
        cmd = 'ROOT_DIR={} {}/contrib/{}.sh {} {}'.format(root_dir, root_dir, dataset,threads, iterations)
        
    out = os.system(cmd)
    if out == 0:
        parse_tmp(dataset, threads, iterations, nodes, filename)
    return "done" if out == 0 else "error" 


def get_repetitions(filename):
    repetitions = 0
    if not params['complete']:
        repetitions = REPETITIONS
    elif not os.path.exists(filename):
        repetitions = REPETITIONS
    else:
        num_lines = sum(1 for line in open(filename))
        repetitions =  REPETITIONS - num_lines

    return repetitions if repetitions > 0 else 0

# Open log file
log = open('log', 'a')

# Main
for dataset in DATASETS:
    for nodes in NODES:
        for iterations in ITERATIONS:
            for threads in THREADS:
                filename = '{}/{}-{}-{}-{}-{}.csv'.format(DIR, VERSION, dataset, threads, iterations, nodes)
                repetitions = get_repetitions(filename)

                run_str = 'Running {}: threads:{}, iterations:{}, nodes:{}, repetitions:{}/{}\n'.format(filename, threads, iterations, nodes, repetitions, REPETITIONS)
                print(run_str)
                
                log.write(run_str)

                if params['test']:
                    continue
                
                log.write('{}:\n'.format(filename))
                for i in range(repetitions):
                    status = run_test(dataset, threads, iterations, nodes, filename)
                    log.write('\t {}: iterations {}/{} {}\n'.format(filename, i+1,repetitions, status))


# Close log
log.close()

