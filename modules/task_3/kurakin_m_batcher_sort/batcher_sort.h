// Copyright 2019 Kurakin Mikhail
#ifndef MODULES_TASK_3_KURAKIN_M_BATCHER_SORT_BATCHER_SORT_H_
#define MODULES_TASK_3_KURAKIN_M_BATCHER_SORT_BATCHER_SORT_H_

#include <mpi.h>
#include <vector>

int compare_int(const void *a, const void *b);
void GatComparators(std::vector<int> procs_up, std::vector<int> procs_down);
void getOddEvenSortNet(std::vector<int> procs);
void CreateSortNet(int numProcs);
int* CreateArray(int size);
int* BatcherSort(int*arrIn, int size);

#endif  // MODULES_TASK_3_KURAKIN_M_BATCHER_SORT_BATCHER_SORT_H_
