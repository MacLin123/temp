// Copyright 2019 Kurakin Mikhail
#include "../../../modules/task_3/kurakin_m_batcher_sort/batcher_sort.h"
#include <mpi.h>
#include <algorithm>
#include <ctime>
#include <random>
#include <utility>
#include <vector>
#include <cstdlib>

struct TPair {
    int a;
    int b;
};

std::vector<TPair> comparators;

int compare_int(const void *a, const void *b) {
    if (*static_cast<const int*>(a) < *static_cast< const int*>(b)) {
        return -1;
    } else if (*static_cast<const int*>(a) == *static_cast<const int*>(b)) {
        return 0;
    } else {
        return 1;
    }
}

void GatComparators(std::vector<int> procs_up, std::vector<int> procs_down) {
    int procCount = procs_up.size() + procs_down.size();
    if (procCount == 1) return;
    if (procCount == 2) {
        TPair tmpPair{procs_up[0], procs_down[0]};
        comparators.push_back(tmpPair);
        return;
    }
    std::vector<int> procs_up_odd;
    std::vector<int> procs_down_odd;

    std::vector<int> procs_up_even;
    std::vector<int> procs_down_even;
    std::vector<int> procsAll(procCount);
    for (uint32_t i = 0; i < procs_up.size(); i++) {
        if (i % 2) {
            procs_up_even.push_back(procs_up[i]);
        } else {
            procs_up_odd.push_back(procs_up[i]);
        }
    }
    for (uint32_t i = 0; i < procs_down.size(); i++) {
        if (i % 2) {
            procs_down_even.push_back(procs_down[i]);
        } else {
            procs_down_odd.push_back(procs_down[i]);
        }
    }
    GatComparators(procs_up_odd, procs_down_odd);
    GatComparators(procs_up_even, procs_down_even);

    std::copy(procs_up.begin(), procs_up.end(), procsAll.begin());
    std::copy(procs_down.begin(), procs_down.end(),
              procsAll.begin() + procs_up.size());

    for (uint32_t i = 1; i < procsAll.size() - 1; i += 2) {
        TPair tmpPair{procsAll[i], procsAll[i + 1]};
        comparators.push_back(tmpPair);
    }
}

void getOddEvenSortNet(std::vector<int> procs) {
    if (procs.size() < 2) return;
    std::vector<int> procs_up(procs.size() / 2);
    std::vector<int> procs_down(procs.size() / 2 + procs.size() % 2);

    std::copy(procs.begin(), procs.begin() + procs_up.size(), procs_up.begin());
    std::copy(procs.begin() + procs_up.size(), procs.end(), procs_down.begin());

    getOddEvenSortNet(procs_up);
    getOddEvenSortNet(procs_down);

    GatComparators(procs_up, procs_down);
}

void CreateSortNet(int numProcs) {
    std::vector<int> procs(numProcs);
    for (uint32_t i = 0; i < procs.size(); i++) {
        procs[i] = i;
    }
    getOddEvenSortNet(procs);
}

int *CreateArray(int size) {
    if (size < 1) return NULL;
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    int *array = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = gen() % 64001 - 32000;
    }
    return array;
}

int *BatcherSort(int *arrIn, int size) {
    MPI_Status status;

    int rank, proc_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (size <= 0) {
        throw "wrong size";
    }
    if (arrIn == nullptr) {
        throw "arr can't be null";
    }

    CreateSortNet(proc_count);

    int sizeNew =
        size + ((size % proc_count) ? (proc_count - (size % proc_count)) : 0);
    int elems_per_proc_size = sizeNew / proc_count;
    int *arrRes = new int[sizeNew];
    for (int i = 0; i < size; i++) {
        arrRes[i] = arrIn[i];
    }
    for (int i = size; i < sizeNew; i++) {
        arrRes[i] = INT32_MIN;
    }

    int *elems_res = new int[elems_per_proc_size];
    int *elems_cur = new int[elems_per_proc_size];
    int *elems_tmp = new int[elems_per_proc_size];

    MPI_Scatter(arrRes, elems_per_proc_size, MPI_INT, elems_res,
                elems_per_proc_size, MPI_INT, 0, MPI_COMM_WORLD);

    std::qsort(elems_res, elems_per_proc_size, sizeof(int), compare_int);
    for (uint32_t i = 0; i < comparators.size(); i++) {
        TPair comparator = comparators[i];
        if (rank == comparator.a) {
            MPI_Send(elems_res, elems_per_proc_size, MPI_INT, comparator.b, 0,
                     MPI_COMM_WORLD);
            MPI_Recv(elems_cur, elems_per_proc_size, MPI_INT, comparator.b, 0,
                     MPI_COMM_WORLD, &status);

            for (int resInd = 0, curInd = 0, tmpInd = 0;
                 tmpInd < elems_per_proc_size; tmpInd++) {
                int res = elems_res[resInd];
                int cur = elems_cur[curInd];
                if (res < cur) {
                    elems_tmp[tmpInd] = res;
                    resInd++;
                } else {
                    elems_tmp[tmpInd] = cur;
                    curInd++;
                }
            }

            std::swap(elems_res, elems_tmp);
        } else if (rank == comparator.b) {
            MPI_Recv(elems_cur, elems_per_proc_size, MPI_INT, comparator.a, 0,
                     MPI_COMM_WORLD, &status);
            MPI_Send(elems_res, elems_per_proc_size, MPI_INT, comparator.a, 0,
                     MPI_COMM_WORLD);
            int start = elems_per_proc_size - 1;
            for (int resInd = start, curInd = start, tmpInd = start;
                 tmpInd >= 0; tmpInd--) {
                int res = elems_res[resInd];
                int cur = elems_cur[curInd];
                if (res > cur) {
                    elems_tmp[tmpInd] = res;
                    resInd--;
                } else {
                    elems_tmp[tmpInd] = cur;
                    curInd--;
                }
            }
            std::swap(elems_res, elems_tmp);
        }
    }
    // union
    MPI_Gather(elems_res, elems_per_proc_size, MPI_INT, arrRes,
               elems_per_proc_size, MPI_INT, 0, MPI_COMM_WORLD);

    int elDiff = sizeNew - size;
    if (rank == 0) {
        if (elDiff) {  // repacking
            int *arrTemp = new int[size];
            for (int32_t i = 0; i < size; i++) {
                arrTemp[i] = arrRes[i + elDiff];
            }
            delete[] arrRes;
            arrRes = arrTemp;
            arrTemp = nullptr;
        }
    }

    delete[] elems_res;
    delete[] elems_tmp;
    delete[] elems_cur;
    return arrRes;
}
