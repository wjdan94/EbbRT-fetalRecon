import os

THREADS=[1,2,3,4,5]
ITERATIONS=[1,2,4]

def parse_tmp(dataset, threads, iterations, nodes):
    tmp = open('tmp', 'r')
    csv = dataset + "," + str(threads) + "," + str(iterations) + ","+ str(nodes) +","
    log = open('results.csv', 'a')
    for line in tmp:
        line = line.rstrip()
        if line:
            s = line.split();
            csv = csv + s[-1] + "," 
    csv = csv[:-1]
    print(csv)
    log.write(csv + "\n")

def run_test(dataset, threads, iterations, nodes):
    
    if dataset == "small":
        out = os.system("./small.sh " + str(threads+1) + " " +  str(iterations) +
                " " + str(nodes) + " " + str(nodes+1))
    else:
        out = os.system("./large.sh " + str(threads+1) + " " +  str(iterations) +
                " " + str(nodes) + " " + str(nodes+1))

    parse_tmp(dataset, threads, iterations, nodes)

#for iteration in ITERATIONS:
#    for thread in THREADS:
#        run_test("small", thread, iteration, 1)

#for iteration in ITERATIONS:
#    for thread in THREADS:
#        run_test("large", thread, iteration, 1)

