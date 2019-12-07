// Copyright 2019 Kurakin Mikhail
#include <gtest/gtest.h>
#include <gtest-mpi-listener.hpp>
#include "./batcher_sort.h"

TEST(Batcher_Sort_MPI, Output_Arr_Not_Null) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 10;
    int arr[] = {1, 8, 5, 3, 2, 9, 0, 4, 7, 6};
    int* resArr = BatcherSort(arr, size);
    if (rank == 0) {
        EXPECT_NE(nullptr, resArr);
    }
}

TEST(Batcher_Sort_MPI, Throw_When_Input_Arr_Is_Null) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 10;
    ASSERT_ANY_THROW(BatcherSort(nullptr, size));
}

TEST(Batcher_Sort_MPI, Throw_Exception_When_Size_Is_Wrong) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 0;
    int arr[] = {1, 8, 5, 3, 2, 9, 0, 4, 7, 6};
    ASSERT_ANY_THROW(BatcherSort(arr, size));
}

TEST(Batcher_Sort_MPI, Array_sorted_properly) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 10;
    int* arr;
    if (rank == 0) {
        arr = CreateArray(size);
    }
    int* arrBatcherSort = BatcherSort(arr, size);
    if (rank == 0) {
        qsort(arr, size, sizeof(int), compare_int);
        bool AreEq = true;
        for (int i = 0; i < size; i++) {
            if (arr[i] != arrBatcherSort[i]) {
                AreEq = false;
                break;
            }
        }
        EXPECT_EQ(true, AreEq);
    }
}

TEST(Batcher_Sort_MPI, Array_sorted_Properly_With_Same_Num) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 10;
    int arr[] = {1, 1, 1, 2, 2, 2, 0, 0, 0, 9};
    int* arrBatcherSort = BatcherSort(arr, size);

    if (rank == 0) {
        qsort(arr, size, sizeof(int), compare_int);
        bool AreEq = true;
        for (int i = 0; i < size; i++) {
            if (arr[i] != arrBatcherSort[i]) {
                AreEq = false;
                break;
            }
        }
        EXPECT_EQ(true, AreEq);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
