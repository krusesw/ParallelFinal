
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include "NumCpp.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <math.h> 
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <stdio.h>

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

nc::NdArray<float> generateDataTestingForCLass(int datapoints, int constraintX, int constraintY, bool garunteedGoodClustering, int numclusters = 3) {
    nc::NdArray<float> generatedDataset = nc::empty<float>(nc::Shape(1, 2));
    for (int i = 0; i < datapoints; i++) {
        if (garunteedGoodClustering) {
            //       nc::NdArray<float> randomPointX = nc::random::uniform<float>(nc::Shape(1), 0, constraintX);
            //       nc::NdArray<float> randomPointY = nc::random::uniform<float>(nc::Shape(1), 0, constraintY);
            //       nc::NdArray<float> randomPoint = nc::append<float>(randomPointX, randomPointY, nc::Axis::NONE);
            //       nc::take()
            //       if
            //       generatedDataset = nc::append<float>(generatedDataset, randomPoint, nc::Axis::ROW);
            //       nc::norm();
        }
        else {
            nc::NdArray<float> randomPointX = nc::random::uniform<float>(nc::Shape(1), (float)0.0, (float)constraintX);
            nc::NdArray<float> randomPointY = nc::random::uniform<float>(nc::Shape(1), (float)0.0, (float)constraintY);
            nc::NdArray<float> randomPoint = nc::append<float>(randomPointX, randomPointY, nc::Axis::NONE);
            generatedDataset = nc::append<float>(generatedDataset, randomPoint, nc::Axis::ROW);
        }
    }
    generatedDataset = nc::deleteIndices(generatedDataset, 0, nc::Axis::ROW);
    return generatedDataset;
}

nc::NdArray<float> euclidianDistanceMatrix(nc::NdArray<float> dataset) {
    nc::NdArray<float> xPoints = nc::deleteIndices(dataset, 1, nc::Axis::COL);
    nc::NdArray<float> yPoints = nc::deleteIndices(dataset, 0, nc::Axis::COL);

    std::pair<nc::NdArray<float>, nc::NdArray<float>> meshpairX = nc::meshgrid(xPoints, xPoints);
    std::pair<nc::NdArray<float>, nc::NdArray<float>> meshpairY = nc::meshgrid(yPoints, yPoints);

    nc::NdArray<float> xDistances = nc::abs(std::get<0>(meshpairX) - std::get<1>(meshpairX));
    nc::NdArray<float> yDistances = nc::abs(std::get<0>(meshpairY) - std::get<1>(meshpairY));

    nc::NdArray<float> euclidianDistances = nc::sqrt(nc::power(xDistances, 2) + nc::power(yDistances, 2));
    euclidianDistances = nc::replace(euclidianDistances, (float)0.0, (float)-1.0);

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


void agglomerativeShortestLink(int datapoints, int numClusters, nc::NdArray<float> distances, nc::NdArray<int> clusterAssignments) {
    nc::NdArray<float> distanceAssessment = nc::where(distances > (float) 0.0, distances, (float) 999999.9);
    nc::NdArray<float> min = nc::min(distanceAssessment);
    float minValue = min(0, 0);
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

    nc::NdArray<float> firstMergePointDistances = distances(distances.rSlice(), removal);
    nc::NdArray<float> secondMergePointDistances = distances(distances.rSlice(), rewrite);
    nc::NdArray<float> mergeSet = nc::stack({ firstMergePointDistances, secondMergePointDistances }, nc::Axis::COL);
    mergeSet = nc::amin(mergeSet, nc::Axis::COL);
    nc::NdArray<float> mergeSetRow = nc::deleteIndices(mergeSet, removal, nc::Axis::COL);
    mergeSetRow = nc::deleteIndices(mergeSetRow, rewrite, nc::Axis::COL);
    nc::NdArray<float> negitiveOne = { -1.0 };
    mergeSetRow = nc::append<float>(negitiveOne, mergeSetRow, nc::Axis::NONE);
    nc::NdArray<float> mergeSetCol = nc::deleteIndices(mergeSetRow, 0, nc::Axis::COL);

    int clustersOG = clusterAssignments.shape().cols;
    nc::NdArray<int> clusterZeros = nc::zeros<int>(1, clustersOG);
    clusterZeros = nc::where(clusterZeros == 0, -1, -1);
    nc::NdArray<int> mergeInClusterOne = clusterAssignments.row(removal);
    for (int value : mergeInClusterOne) {
        printf("%i", value);
        if (value > -1) {
            nc::NdArray<int> valueint = { value };
            clusterZeros = nc::deleteIndices(clusterZeros, clustersOG - 1, nc::Axis::COL);
            clusterZeros = nc::append<int>(valueint, clusterZeros, nc::Axis::COL);
        }
    }
    nc::NdArray<int> mergeInClusterTwo = clusterAssignments.row(rewrite);
    for (int value : mergeInClusterTwo) {
        printf("%i", value);
        if (value > -1) {
            nc::NdArray<int> valueint = { value };
            clusterZeros = nc::deleteIndices(clusterZeros, clustersOG - 1, nc::Axis::COL);
            clusterZeros = nc::append<int>(valueint, clusterZeros, nc::Axis::COL);
        }
    }

    clusterAssignments = nc::deleteIndices(clusterAssignments, removal, nc::Axis::ROW);
    clusterAssignments = nc::deleteIndices(clusterAssignments, rewrite, nc::Axis::ROW);
    clusterAssignments = nc::append<int>(clusterZeros, clusterAssignments, nc::Axis::ROW);

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

__global__ void agglomerativeShortestLinkCUDA(int datapoints, int numClusters, float* distances, int* clusterAssignments) {

}

struct gtz {
    __host__ __device__ bool operator() (double x) { return x > 0.; }
};

struct gtzVec {
    __host__ __device__ bool operator() (float x, float y) { return x > 0.; }
};

void sequentialRun() {
    nc::NdArray<float> dataSet = generateDataTestingForCLass(10, 100, 100, false);
    nc::NdArray<float> euclidianDistances = euclidianDistanceMatrix(dataSet);
    nc::NdArray<int> clusterAssignments = initialClusterAssignment(10, false);

    std::vector<float> distanceVector = euclidianDistances.toStlVector();
    std::vector<int> clusterVector = clusterAssignments.toStlVector();

    float* distancePointer = distanceVector.data();
    int* clusterPointer = clusterVector.data();

    //Calling this appearently makes cuda start faster for loading thrust
    cudaFree(0);

    //Convert Distance Vector to Thrust Vector for parallel compuation
    //https://github.com/NVIDIA/thrust/
    //WARNING: ACTIVELY BUGGED IN NEWEST VERSION OF CUDA
    //IF YOU HAVE CUDA 11.0 OR 11.1, THIS WILL NOT WORK
    //FOLLOW WORKAROUND HERE: 
    printf("%s", "cuda starting");
    thrust::device_vector<float> d_vec = distanceVector;
    //Sort distance vector, reminder that -1 values will be at front

    float *minElement = thrust::min_element(thrust::device, d_vec.begin(), d_vec.begin()+100, );

    for (int i = 0; i < d_vec.size(); i++)
        std::cout << "D[" << i << "] = " << d_vec[i] << std::endl;

    //Find minimum non negative value
    thrust::device_vector<float>::iterator iter1 = d_vec.begin();
    thrust::device_vector<float>::iterator iter2 = thrust::find_if(d_vec.begin(), d_vec.begin() + 100, gtz());
    int d = thrust::distance(iter1, iter2);

    printf("%i", d);

    int row = d / 10;
    int col = d % 10;
    int removal = 0;
    int rewrite = 0;
    if (row >= col) {
        removal = row;
        rewrite = col;
    }
    else {
        removal = col;
        rewrite = row;
    }

    printf("%i", row);
    printf("%i", col);


     //  thrust::host_vector<int> h_vec(32 << 20);
     //  std::generate(h_vec.begin(), h_vec.end(), rand);
    //   thrust::device_vector<int> d_vec = h_vec;
    //   thrust::sort(d_vec.begin(), d_vec.end());
    //   thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

       //    float* cudaDistancePointer;
       //    int* cudaClusterPointer;

       //    int numbytes = 10 * 10 * sizeof(float);

       //    cudaMalloc(&cudaDistancePointer, numbytes);
       //    cudaMalloc(&cudaClusterPointer, numbytes);
       //    cudaMemcpy(cudaDistancePointer, distancePointer, numbytes, cudaMemcpyHostToDevice);
       //    cudaMemcpy(cudaClusterPointer, clusterPointer, numbytes, cudaMemcpyHostToDevice);

       //    agglomerativeShortestLinkCUDA << <1, 10 >> > (10, 3, cudaDistancePointer, cudaClusterPointer);
           //agglomerativeShortestLinkCUDA << <1, datapoints >> > (datapoints, numClusters, ffd.data(), ffd.data());

  //  euclidianDistances.print();
   // agglomerativeShortestLink(10, 3, euclidianDistances, clusterAssignments);

}







int main()
{
    bool exitLoop = false;
    while (!exitLoop) {
        std::cout << "Enter 1 to run sequential, Enter 2 to run parallel, Other key to exit: ";
        int option = -1;
        std::cin >> option;
        if (std::cin.good()) {
            if (option == 1) {
                sequentialRun();
            }
            else if (option == 2) {
                // cudaRun();
            }
            else {
                return 0;
            }
        }
        else {
            return 0;
        }

    }
    cudaDeviceSynchronize();

    return 0;
}