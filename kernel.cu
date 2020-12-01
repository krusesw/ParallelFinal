
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "NumCpp.hpp"

#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

nc::NdArray<double> generateDataTestingForCLass(int datapoints, int constraintX, int constraintY, bool garunteedGoodClustering, int numclusters = 3) {
    nc::NdArray<double> generatedDataset = nc::empty<double>(nc::Shape(1, 2));
    for (int i = 0; i < datapoints; i++) {
        if (garunteedGoodClustering) {
            //       nc::NdArray<double> randomPointX = nc::random::uniform<double>(nc::Shape(1), 0, constraintX);
            //       nc::NdArray<double> randomPointY = nc::random::uniform<double>(nc::Shape(1), 0, constraintY);
            //       nc::NdArray<double> randomPoint = nc::append<double>(randomPointX, randomPointY, nc::Axis::NONE);
            //       nc::take()
            //       if
            //       generatedDataset = nc::append<double>(generatedDataset, randomPoint, nc::Axis::ROW);
            //       nc::norm();
        }
        else {
            nc::NdArray<double> randomPointX = nc::random::uniform<double>(nc::Shape(1), 0, constraintX);
            nc::NdArray<double> randomPointY = nc::random::uniform<double>(nc::Shape(1), 0, constraintY);
            nc::NdArray<double> randomPoint = nc::append<double>(randomPointX, randomPointY, nc::Axis::NONE);
            generatedDataset = nc::append<double>(generatedDataset, randomPoint, nc::Axis::ROW);
        }
    }
    generatedDataset = nc::deleteIndices(generatedDataset, 0, nc::Axis::ROW);
    return generatedDataset;
}

nc::NdArray<double> euclidianDistanceMatrix(nc::NdArray<double> dataset) {
    nc::NdArray<double> xPoints = nc::deleteIndices(dataset, 1, nc::Axis::COL);
    nc::NdArray<double> yPoints = nc::deleteIndices(dataset, 0, nc::Axis::COL);

    std::pair<nc::NdArray<double>, nc::NdArray<double>> meshpairX = nc::meshgrid(xPoints, xPoints);
    std::pair<nc::NdArray<double>, nc::NdArray<double>> meshpairY = nc::meshgrid(yPoints, yPoints);

    nc::NdArray<double> xDistances = nc::abs(std::get<0>(meshpairX) - std::get<1>(meshpairX));
    nc::NdArray<double> yDistances = nc::abs(std::get<0>(meshpairY) - std::get<1>(meshpairY));

    nc::NdArray<double> euclidianDistances = nc::sqrt(nc::power(xDistances, 2) + nc::power(yDistances, 2));
    euclidianDistances = nc::replace(euclidianDistances, 0.0, (-1.0));

    return euclidianDistances;
}

nc::NdArray<int> initialClusterAssignment(int datapoints, bool garunteedGoodClustering, int numclusters = 3) {
    nc::NdArray<int> clusterAssignment = nc::arange<int>(0, datapoints);
    clusterAssignment = clusterAssignment.reshape(datapoints, 1);
    nc::NdArray<int> clusterZeros = nc::zeros<int>(datapoints, datapoints - 1);
    clusterZeros = nc::where(clusterZeros == 0, -1, -1);
    clusterAssignment = nc::append<int>(clusterAssignment, clusterZeros, nc::Axis::COL);
    return clusterAssignment;
}

void agglomerativeShortestLink(int datapoints, int numClusters, nc::NdArray<double> distances, nc::NdArray<int> clusterAssignments) {
    nc::NdArray<double> distanceAssessment = nc::where(distances > 0.0, distances, 99999999999999999999.9);
    nc::NdArray<double> min = nc::min(distanceAssessment);
    double minValue = min(0, 0);
    nc::NdArray<nc::uint32> minIndicies = nc::argmin(distanceAssessment, nc::Axis::NONE);
    int minInt = int(minIndicies(0, 0));

    //Always cluster left
    int row = minInt / datapoints;
    int column = minInt % datapoints;
    int removal = 0;
    int rewrite = 0;
    if (row >= column) {
        removal = row;
        rewrite = column;
    }
    else {
        removal = column;
        rewrite = row;
    }

    nc::NdArray<double> firstMergePointDistances = distances(distances.rSlice(), removal);
    nc::NdArray<double> secondMergePointDistances = distances(distances.rSlice(), rewrite);
    nc::NdArray<double> mergeSet = nc::stack({ firstMergePointDistances, secondMergePointDistances }, nc::Axis::COL);
    mergeSet = nc::amin(mergeSet, nc::Axis::COL);
    nc::NdArray<double> mergeSetRow = nc::deleteIndices(mergeSet, removal, nc::Axis::COL);
    mergeSetRow = nc::deleteIndices(mergeSetRow, rewrite, nc::Axis::COL);
    nc::NdArray<double> negitiveOne = { -1.0 };
    mergeSetRow = nc::append<double>(negitiveOne, mergeSetRow, nc::Axis::NONE);
    nc::NdArray<double> mergeSetCol = nc::deleteIndices(mergeSetRow, 0, nc::Axis::COL);

    int clustersOG = clusterAssignments.shape().cols;
    nc::NdArray<int> clusterZeros = nc::zeros<int>(1, clustersOG);
    clusterZeros = nc::where(clusterZeros == 0, -1, -1);
    nc::NdArray<int> mergeInClusterOne = clusterAssignments.row(removal);
    for (int value : mergeInClusterOne) {
        if (value > -1) {
            nc::NdArray<int> valueint = { value };
            clusterZeros = nc::append<int>(valueint, clusterZeros, nc::Axis::COL);
        }
    }
    nc::NdArray<int> mergeInClusterTwo = clusterAssignments.row(removal);
    for (int value : mergeInClusterTwo) {
        if (value > -1) {
            nc::NdArray<int> valueint = { value };
            clusterZeros = nc::append<int>(valueint, clusterZeros, nc::Axis::COL);
        }
    }

    clusterAssignments = nc::deleteIndices(clusterAssignments, removal, nc::Axis::ROW);
    clusterAssignments = nc::deleteIndices(clusterAssignments, rewrite, nc::Axis::ROW);
    clusterAssignments = nc::append<int>(clusterZeros(nc::Slice(0, 0), nc::Slice(0, clustersOG)), clusterAssignments, nc::Axis::ROW);

    distances = nc::deleteIndices(distances, removal, nc::Axis::ROW);
    distances = nc::deleteIndices(distances, removal, nc::Axis::COL);
    distances = nc::deleteIndices(distances, rewrite, nc::Axis::ROW);
    distances = nc::deleteIndices(distances, rewrite, nc::Axis::COL);

    distances = nc::stack({ mergeSetCol.reshape(datapoints - 2,1), distances }, nc::Axis::COL);
    distances = nc::stack({ mergeSetRow, distances }, nc::Axis::ROW);

    printf("%s", "\n");
    printf("%s", "\n");
    printf("%s", "\n");
    printf("%s", "\n");
    mergeSetRow.print();
    mergeSetCol.print();
    printf("%s", "\n");
    distances.print();
    printf("%s", "\n");
    clusterAssignments.print();
    if (datapoints - 1 > numClusters) {
        datapoints = datapoints - 1;
        agglomerativeShortestLink(datapoints, numClusters, distances, clusterAssignments);
    }
}





int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };

    nc::NdArray<double> dataSet = generateDataTestingForCLass(10, 100, 100, false);
    nc::NdArray<double> euclidianDistances = euclidianDistanceMatrix(dataSet);
    nc::NdArray<int> clusterAssignments = initialClusterAssignment(10, false);
    //   if (garunteedGoodClustering) {
    //       datapoints = datapoints + numclusters;
    //   }
    euclidianDistances.print();
    agglomerativeShortestLink(10, 3, euclidianDistances, clusterAssignments);

    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
