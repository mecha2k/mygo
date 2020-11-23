import time
import multiprocessing


def count(name):
    for i in range(50000):
        print(name, " : ", i)


if __name__ == "__main__":
    name_list = ["name1", "name2", "name3"]

    # start = time.time()
    # for name in name_list:
    #     count(name)
    # print("elapsed time: %s sec." % (time.time() - start))

    start = time.time()
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    pool.imap(count, name_list)
    pool.close()
    pool.join()
    print("elapsed time: %s sec." % (time.time() - start))
