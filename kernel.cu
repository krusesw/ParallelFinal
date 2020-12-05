
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
#include <thrust/iterator/transform_iterator.h>
#include <math.h> 
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>

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
    euclidianDistances = nc::replace(euclidianDistances, (float)0.0, (float)999999.9);

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


void agglomerativeShortestLinkSequential(int datapoints, int numClusters, nc::NdArray<float> distances, nc::NdArray<int> clusterAssignments) {
    //Find minimum distance and record index and value
    nc::NdArray<float> distanceAssessment = nc::where(distances > (float)0.0, distances, (float)999999.9);
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

    //Merges removed columns
    nc::NdArray<float> firstMergePointDistances = distances(distances.rSlice(), removal);
    nc::NdArray<float> secondMergePointDistances = distances(distances.rSlice(), rewrite);
    nc::NdArray<float> mergeSet = nc::stack({ firstMergePointDistances, secondMergePointDistances }, nc::Axis::COL);
    mergeSet = nc::amin(mergeSet, nc::Axis::COL);
    nc::NdArray<float> mergeSetRow = nc::deleteIndices(mergeSet, removal, nc::Axis::COL);
    mergeSetRow = nc::deleteIndices(mergeSetRow, rewrite, nc::Axis::COL);
    nc::NdArray<float> negitiveOne = { -1.0 };
    mergeSetRow = nc::append<float>(negitiveOne, mergeSetRow, nc::Axis::NONE);
    nc::NdArray<float> mergeSetCol = nc::deleteIndices(mergeSetRow, 0, nc::Axis::COL);

    //Clusters points together based on min distance calculated
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

    //Remove all values we no longer need because they were in a row or col with min distance
    //Replace 2 rows and 2 cols removed with 1 row and col for new cluster
    clusterAssignments = nc::deleteIndices(clusterAssignments, removal, nc::Axis::ROW);
    clusterAssignments = nc::deleteIndices(clusterAssignments, rewrite, nc::Axis::ROW);
    clusterAssignments = nc::append<int>(clusterZeros, clusterAssignments, nc::Axis::ROW);

    distances = nc::deleteIndices(distances, removal, nc::Axis::ROW);
    distances = nc::deleteIndices(distances, removal, nc::Axis::COL);
    distances = nc::deleteIndices(distances, rewrite, nc::Axis::ROW);
    distances = nc::deleteIndices(distances, rewrite, nc::Axis::COL);

    distances = nc::stack({ mergeSetCol.reshape(datapoints - 2,1), distances }, nc::Axis::COL);
    distances = nc::stack({ mergeSetRow, distances }, nc::Axis::ROW);

    if (datapoints - 1 > numClusters) {
        datapoints = datapoints - 1;
        agglomerativeShortestLinkSequential(datapoints, numClusters, distances, clusterAssignments);
    }
    else {
        clusterAssignments.print();
    }
}

struct gtz {
    __device__ bool operator() (double x) { return x > 0.; }
};

struct delPoint {
    __device__ bool operator() (int x) { return (x == 1); }
};

typedef thrust::tuple<int, float> argMinType;



void agglomerativeShortestLinkCuda(int numPoints, int originalNumPoints, int numCluster, float* distancePointer, int* clusterPointer) {
    //Convert Distance Vector to Thrust Vector for parallel compuation
    //https://github.com/NVIDIA/thrust/
    //WARNING: ACTIVELY BUGGED IN NEWEST VERSION OF CUDA
    //IF YOU HAVE CUDA 11.0 OR 11.1, THIS WILL NOT WORK
    //FOLLOW WORKAROUND HERE: 
    thrust::device_vector<float> cudaDistanceVector(distancePointer, distancePointer + numPoints*numPoints);

    //Find min distance using thrust min element divide and conqour approach on device  
    thrust::device_ptr<float> CDVPtr = cudaDistanceVector.data();
    thrust::device_vector<float>::iterator minIterator = thrust::min_element(thrust::device, CDVPtr, CDVPtr + cudaDistanceVector.size());

    //Get value for index of vector
    unsigned int index = minIterator - cudaDistanceVector.begin();

    //Transform index into row cloumn data using divide and modulo
    //No need for cuda since these are 1 step
    unsigned int row = index / numPoints;
    unsigned int col = index % numPoints;

    //To avoid indexing issues, always remove the rightmost column and downmost row first
    //Rename closest index between row and column to 0, named leftIndex
    //Rename farthest index between row and column to 0, named rightIndex
    //No need for cuda since these are O(1)
    unsigned int rightIndex = 0;
    unsigned int leftIndex = 0;
    if (row >= col) {
        rightIndex = row;
        leftIndex = col;
    }
    else {
        rightIndex = col;
        leftIndex = row;
    }

    //Declaring keys to delete from distance vector
    //Could not find a way to do this more efficiently using thrust in time
    //Issue could potentially be solved by setting two thrust sequences and combining them, but order matters
    thrust::device_vector<int> deleteKeys(numPoints* numPoints);
    for (int i = 0; i < (numPoints* numPoints); i++) {
        if (i % numPoints == leftIndex || i / numPoints == leftIndex || i % numPoints == rightIndex || i / numPoints == rightIndex) {
            deleteKeys[i] = 1;
        }
    }

    //Get columns to merge together
    thrust::device_vector<float> mergeRowOne(numPoints);
    thrust::copy(thrust::device, CDVPtr + rightIndex * numPoints, CDVPtr + (rightIndex * numPoints) + numPoints, mergeRowOne.begin());
    thrust::device_vector<float> mergeRowTwo(numPoints);
    thrust::copy(thrust::device, CDVPtr + leftIndex * numPoints, CDVPtr + (leftIndex * numPoints) + numPoints, mergeRowTwo.begin());

    //Create new vector containing those two columns
    mergeRowOne.insert(mergeRowOne.begin() + numPoints, mergeRowTwo.begin(), mergeRowTwo.begin() + numPoints);

    //Get min from each column of mergeRowOne, merge into new vector of minimums
    //With help from advice on this thread:
    //https://stackoverflow.com/questions/17698969/determining-the-least-element-and-its-position-in-each-matrix-column-with-cuda-t/29841094#29841094
    thrust::device_vector<float>    distanceMinVector(numPoints);
    thrust::device_vector<int>      distanceMinIndicies(numPoints);

    thrust::reduce_by_key(
        thrust::make_transform_iterator(
            thrust::make_counting_iterator((int)0),
            thrust::placeholders::_1 / 2),
        thrust::make_transform_iterator(
            thrust::make_counting_iterator((int)0),
            thrust::placeholders::_1 / 2) + 2 * numPoints,
        thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::make_permutation_iterator(
                    mergeRowOne.begin(),
                    thrust::make_transform_iterator(
                        thrust::make_counting_iterator((int)0), (thrust::device,thrust::placeholders::_1 % 2) * numPoints + thrust::placeholders::_1 / 2)),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator((int)0), thrust::placeholders::_1 % 2))),
        thrust::make_discard_iterator(),
        thrust::make_zip_iterator(
            thrust::make_tuple(
                distanceMinVector.begin(),
                distanceMinIndicies.begin())),
        thrust::equal_to<int>(),
        thrust::minimum<thrust::tuple<float, int> >()
    );

    //Get clusters to merge together
    thrust::device_vector<int> mergeCRowOne(clusterPointer+(rightIndex*originalNumPoints), clusterPointer+ (rightIndex * originalNumPoints)+originalNumPoints);
    int *uniquePtrOne = thrust::unique(thrust::device, mergeCRowOne.begin(), mergeCRowOne.begin() + originalNumPoints);

    thrust::device_vector<int> mergeCRowTwo(clusterPointer + (leftIndex * originalNumPoints), clusterPointer + (leftIndex * originalNumPoints) + originalNumPoints);
    int* uniquePtrTwo = thrust::unique(thrust::device, mergeCRowTwo.begin(), mergeCRowTwo.begin() + originalNumPoints);

    //Remove the minvalue from minarray of new cluster, top left most value will always be 999999.9 once inserted.
    distanceMinVector.erase(distanceMinVector.begin() + rightIndex);
    distanceMinVector.erase(distanceMinVector.begin() + leftIndex);

    //Delete old clusters from distance vector
    thrust::device_vector<float>::iterator delIterator = thrust::remove_if(cudaDistanceVector.begin(), cudaDistanceVector.begin() + cudaDistanceVector.size(), deleteKeys.begin(), delPoint());

    //Insert new min row for new cluster into distance matrix
    distanceMinVector.insert(distanceMinVector.begin() + numPoints-2, cudaDistanceVector.begin(), cudaDistanceVector.begin() + cudaDistanceVector.size());

    //Creating new vector with clustered row
    thrust::device_vector<float> cudaDistanceVectorNew((numPoints-1) * (numPoints-1));

    thrust::copy(distanceMinVector.begin(), distanceMinVector.end(), std::ostream_iterator<float>(std::cout, " "));

    std::cout << "\n\n";

    

    //Fill new vector with data (currently no way to insert column to left that I know of in cuda without a more time complexive method),
    //Have to use sequential for loop here, time constraints


    
    thrust::copy(cudaDistanceVectorNew.begin(), cudaDistanceVectorNew.end(), std::ostream_iterator<float>(std::cout, " "));
}

//Prompts users for dataset generation and then starts clustering on service specified
void setup(bool cuda) {
    std::cout << "Enter number of datapoints to generate: ";
    int numPoints = -1;
    std::cin >> numPoints;

    std::cout << "Enter cluster number to stop generating at: ";
    int numCluster = -1;
    std::cin >> numCluster;

    std::cout << "Enter x max (less than 500000): ";
    int xMax = -1;
    std::cin >> xMax;

    std::cout << "Enter y max (less than 500000): ";
    int yMax = -1;
    std::cin >> yMax;

    if (numPoints < 0 || yMax > 500000 || yMax < 0 || xMax > 500000 || xMax < 0) {
        std::cout << "Unacceptable Values, try again \n";
        return;
    }
    //Setup data
    nc::NdArray<float> dataSet = generateDataTestingForCLass(numPoints, xMax, yMax, false);
    nc::NdArray<float> euclidianDistances = euclidianDistanceMatrix(dataSet);
    nc::NdArray<int> clusterAssignments = initialClusterAssignment(10, false);
    clock_t timer;
    std::cout << "\nStarting with euclidian distance matrix: \n";
    euclidianDistances.print();
    std::cout << "\nStarting with each point in seperate clustering. \n";

    if (!cuda) {
        std::cout << "\nStarting sequential. \n";
        timer = clock();
        agglomerativeShortestLinkSequential(numPoints, numCluster, euclidianDistances, clusterAssignments);
        float dt = clock() - timer;
        std::cout << "took " << dt << " ms \n";
    }
    else {
        //Prepare data for cuda
        std::vector<float> distanceVector = euclidianDistances.toStlVector();
        std::vector<int> clusterVector = clusterAssignments.toStlVector();
        float* distancePointer = distanceVector.data();
        int* clusterPointer = clusterVector.data();
        //Calling this appearently makes cuda start faster for loading thrust
        cudaFree(0);
        std::cout << "\nStarting CUDA. \n";
        timer = clock();
        agglomerativeShortestLinkCuda(numPoints, numPoints, numCluster, distancePointer, clusterPointer);
        float dt = clock() - timer;
        std::cout << "took " << dt << " ms \n";
    }
}

//Main GUI loop
int main()
{
    bool exitLoop = false;
    while (!exitLoop) {
        std::cout << "Enter 1 to run sequential, Enter 2 to run parallel, Other key to exit: ";
        int option = -1;
        std::cin >> option;
        if (std::cin.good()) {
            if (option == 1) {
                setup(false);
            }
            else if (option == 2) {
                setup(true);
            }
            else {
                return 0;
            }
        }
        else {
            return 0;
        }

    }

    return 0;
}