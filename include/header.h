#include <iostream>
#include <string>
#include <bitset>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <math.h>
#include <time.h>
#include <boost/version.hpp>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <valarray>
#include <sys/time.h>
#include <dirent.h> // for looping over the files in the directory
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl/console/parse.h>
#include <pcl/octree/octree.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/boundary.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/usc.h>
#include <pcl/features/rsd.h>
#include <pcl/features/impl/rsd.hpp>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <pcl/features/3dsc.h>



//#include <generate_random.h>


using namespace std;

using namespace Eigen;

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

//#define PI 3.14159265



#include <utility>
//#include <super_duper.h>

#include <pcl/features/rops_estimation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>


typedef std::pair<float,int> mypair;

bool comparator_here ( const mypair& l, const mypair& r)
{ return l.first > r.first; }

bool comparator ( const mypair& l, const mypair& r)
{ return l.first > r.first; }

inline double getMs()
{
    struct timeval t0;
    gettimeofday(&t0, NULL);
    double ret = t0.tv_sec * 1000.0;
    ret += ((double) t0.tv_usec)*0.001;
    return ret;
}

template< typename T >
T minVect(const T *v, int n, int *ind=NULL)
{
    assert(n > 0);

    T min = v[0];
    if (ind != NULL) *ind = 0;
    for (int i=1; i<n; i++)
        if (v[i] < min) {
            min = v[i];
            if (ind != NULL) *ind=i;
        }

    return min;
}


class cbshot_descriptor
{
public:
    int bins[352];
    std::bitset< 352 > bits;
    std::bitset< 483 > overlap_bits;
};

class cbrops_descriptor
{
public:
    int bins[135];
    std::bitset< 135 > bits;
    std::bitset< 483 > overlap_bits;
};


class cbfpfh_descriptor
{
public:
    int bins[36];
    std::bitset< 33 > bits;
    //std::bitset< 36 > bits;
};

struct desc
{
    std::vector<int> values;
};

class binary_desc
{
public:
    std::bitset<64> bits;
};


class cbshot
{

public :

    pcl::PointCloud<pcl::PointXYZ> cloud1, cloud2;
    pcl::PointCloud<pcl::Normal> cloud1_normals, cloud2_normals;
    pcl::PointCloud<pcl::PointXYZ> cloud1_keypoints, cloud2_keypoints;

    pcl::PointCloud<pcl::SHOT352> cloud1_shot, cloud2_shot;
    pcl::PointCloud<pcl::FPFHSignature33> cloud1_fpfh, cloud2_fpfh;
    pcl::PointCloud<pcl::Histogram<153> > cloud1_spin, cloud2_spin;
    pcl::PointCloud<pcl::UniqueShapeContext<pcl::PointXYZ> > cloud1_usc, cloud2_usc;
    pcl::PointCloud<pcl::ShapeContext1980> cloud1_sc, cloud2_sc;
    pcl::PointCloud<pcl::PrincipalRadiiRSD> cloud1_rsd, cloud2_rsd;
    pcl::PointCloud<pcl::Histogram <135> > histograms1;
    pcl::PointCloud<pcl::Histogram <135> > histograms2;

    std::vector<desc> compressed_shot1, compressed_shot2;
    std::vector<desc> double_compressed_shot1, double_compressed_shot2;
    std::vector<desc> reconstructed_compressed_shot1, reconstructed_compressed_shot2;

    std::vector<binary_desc> binary_compressed_shot1, binary_compressed_shot2;

    std::vector<cbshot_descriptor> cloud1_cbshot, cloud2_cbshot;
    std::vector<cbrops_descriptor> cloud1_cbrops, cloud2_cbrops;

    std::vector<cbfpfh_descriptor> cloud1_cbfpfh, cloud2_cbfpfh;

    std::vector<int> cloud1_keypoints_indices, cloud2_keypoints_indices;

    pcl::IndicesConstPtr indices;

    pcl::PolygonMesh mesh1, mesh2;







    void calculate_normals ( float radius )
    {
        // Estimate the normals.
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setRadiusSearch(radius);
        normalEstimation.setNumberOfThreads(12);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);

        normalEstimation.setInputCloud(cloud1.makeShared());
        normalEstimation.compute(cloud1_normals);

        normalEstimation.setInputCloud(cloud2.makeShared());
        normalEstimation.compute(cloud2_normals);
    }


    void  calculate_iss_keypoints ( float leaf_size)
    {

        //
        //  ISS3D parameters
        //
        double iss_salient_radius_;
        double iss_non_max_radius_;
        double iss_normal_radius_;
        double iss_border_radius_;
        double iss_gamma_21_ (0.975);
        double iss_gamma_32_ (0.975);
        double iss_min_neighbors_ (5);
        int iss_threads_ (8);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());

        // Fill in the model cloud

        double model_resolution = leaf_size;

        // Compute model_resolution

        iss_salient_radius_ = 6 * model_resolution;
        iss_non_max_radius_ = 4 * model_resolution;
        iss_normal_radius_ = 4 * model_resolution;
        iss_border_radius_ = 1 * model_resolution;

        //
        // Compute keypoints
        //
        pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;

        iss_detector.setSearchMethod (tree);
        iss_detector.setSalientRadius (iss_salient_radius_);
        iss_detector.setNonMaxRadius (iss_non_max_radius_);

        iss_detector.setNormalRadius (iss_normal_radius_);
        iss_detector.setBorderRadius (iss_border_radius_);

        iss_detector.setThreshold21 (iss_gamma_21_);
        iss_detector.setThreshold32 (iss_gamma_32_);
        iss_detector.setMinNeighbors (iss_min_neighbors_);
        iss_detector.setNumberOfThreads (iss_threads_);


        iss_detector.setInputCloud (cloud1.makeShared());
        iss_detector.compute (cloud1_keypoints);

        iss_detector.setInputCloud (cloud2.makeShared());
        iss_detector.compute (cloud2_keypoints);


    }


    void  calculate_voxel_grid_keypoints ( float leaf_size )
    {
        // Find Keypoints on the input cloud
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);

        voxel_grid.setInputCloud(cloud1.makeShared());
        voxel_grid.filter(cloud1_keypoints);

        voxel_grid.setInputCloud(cloud2.makeShared());
        voxel_grid.filter(cloud2_keypoints);

        //        Eigen::Matrix4f Tx;
        //        Tx << 0.995665,	-0.00287336,	-0.0929723,	-0.0966733,
        //        0.00810565,	0.998401,	0.0559452,	-0.0522716,
        //        0.0926635,	-0.0564565,	0.994095,	-0.0271486,
        //        0,	0,	0,	1;

        //        pcl::transformPointCloud(cloud1_keypoints,cloud2_keypoints,Tx);


    }


    void get_keypoint_indices()
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud (cloud1.makeShared());
        pcl::PointXYZ searchPoint;

        for (int i = 0; i < cloud1_keypoints.size(); i++)
        {
            searchPoint = cloud1_keypoints[i];

            int K = 1;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                cloud1_keypoints_indices.push_back(pointIdxNKNSearch[0]);
            }

        }

        kdtree.setInputCloud (cloud2.makeShared());

        for (int i = 0; i < cloud2_keypoints.size(); i++)
        {
            searchPoint = cloud2_keypoints[i];

            int K = 1;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                cloud2_keypoints_indices.push_back(pointIdxNKNSearch[0]);
            }

        }

    }


    void get_keypoint_indices_for_evaluation(float squared_distance, pcl::Correspondences &ground_truth)
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud (cloud1.makeShared());
        pcl::PointXYZ searchPoint;

        for (int i = 0; i < cloud1_keypoints.size(); i++)
        {
            searchPoint = cloud1_keypoints[i];

            int K = 1;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                cloud1_keypoints_indices.push_back(pointIdxNKNSearch[0]);
            }

        }

        kdtree.setInputCloud (cloud2.makeShared());

        int counter_index_j; counter_index_j = 0;
        for (int i = 0; i < cloud2_keypoints.size(); i++)
        {
            searchPoint = cloud2_keypoints[i];

            int K = 1;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);

            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
            {
                if (pointNKNSquaredDistance[0] < squared_distance)
                {

                    cloud2_keypoints_indices.push_back(pointIdxNKNSearch[0]);

                    pcl::Correspondence c;
                    c.index_query = i;
                    c.index_match = counter_index_j;

                    ground_truth.push_back(c);

                    counter_index_j++;
                }
            }

        }

        cloud2_keypoints.clear();
        for (int i = 0; i < cloud2_keypoints_indices.size(); i++)
            cloud2_keypoints.push_back(cloud2[cloud2_keypoints_indices[i]]);

    }

    void  calculate_voxel_grid_keypoints_for_evaluation ( float leaf_size, Eigen::Matrix4f Tx )
    {
        // Find Keypoints on the input cloud
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);

        voxel_grid.setInputCloud(cloud1.makeShared());
        voxel_grid.filter(cloud1_keypoints);

        //voxel_grid.getIndices()

        //voxel_grid.setInputCloud(cloud2.makeShared());
        //voxel_grid.filter(cloud2_keypoints);


        pcl::transformPointCloud(cloud1_keypoints, cloud2_keypoints,Tx);


    }

    /*
    void calculate_curvature_keypoints(float leaf_size, Eigen::Matrix4f Tx)
    {
        KPD kp;
        kp.cloud = cloud1;
        kp.normal_estimation_pcl(0.02);
        kp.remove_boundary_points(0.01);
        kp.non_maxima_suppression(0.02);
        pcl::copyPointCloud( kp.fifth_keypoints_cloud, cloud1_keypoints);



        pcl::transformPointCloud(cloud1_keypoints, cloud2_keypoints,Tx);

    }

*/
    void calculate_SHOT ( float radius )
    {

        // SHOT estimation object.
        pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
        shot.setRadiusSearch(radius);

        shot.setNumberOfThreads(12);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        shot.setSearchMethod(kdtree);

        shot.setInputCloud(cloud1_keypoints.makeShared());
        shot.setSearchSurface(cloud1.makeShared());
        shot.setInputNormals(cloud1_normals.makeShared());
        shot.compute(cloud1_shot);

        shot.setInputCloud(cloud2_keypoints.makeShared());
        shot.setSearchSurface(cloud2.makeShared());
        shot.setInputNormals(cloud2_normals.makeShared());
        shot.compute(cloud2_shot);

    }


    void calculate_FPFH ( float radius )
    {

        // SHOT estimation object.
        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        fpfh.setRadiusSearch(radius);

        fpfh.setNumberOfThreads(12);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh.setSearchMethod(kdtree);

        fpfh.setInputCloud(cloud1_keypoints.makeShared());
        fpfh.setSearchSurface(cloud1.makeShared());
        fpfh.setInputNormals(cloud1_normals.makeShared());
        fpfh.setNumberOfThreads(12);
        fpfh.compute(cloud1_fpfh);


        fpfh.setInputCloud(cloud2_keypoints.makeShared());
        fpfh.setSearchSurface(cloud2.makeShared());
        fpfh.setInputNormals(cloud2_normals.makeShared());
        fpfh.setNumberOfThreads(12);
        fpfh.compute(cloud2_fpfh);


    }



    void calculate_SpinImage ( float radius )
    {

        pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> > spin_image_descriptor;

        pcl::PointCloud<pcl::Normal>::Ptr normals2_spin(new pcl::PointCloud<pcl::Normal>);


        // Estimate the normals.
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(cloud2_keypoints.makeShared());
        normalEstimation.setSearchSurface(cloud2.makeShared());
        normalEstimation.setRadiusSearch(0.02);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);
        normalEstimation.compute(*normals2_spin);




        spin_image_descriptor.setInputCloud (cloud2_keypoints.makeShared());
        spin_image_descriptor.setInputNormals (normals2_spin);

        spin_image_descriptor.setRadiusSearch (radius);
        spin_image_descriptor.setImageWidth(8);
        // Actually compute the spin images
        spin_image_descriptor.compute (cloud2_spin);



    }




    void calculate_usc ( float radius )
    {

        // USC estimation object.
        pcl::UniqueShapeContext<pcl::PointXYZ> usc;
        usc.setInputCloud(cloud1_keypoints.makeShared());
        usc.setSearchSurface(cloud1.makeShared());

        // Search radius, to look for neighbors. It will also be the radius of the support sphere.
        usc.setRadiusSearch(radius);
        // The minimal radius value for the search sphere, to avoid being too sensitive
        // in bins close to the center of the sphere.
        usc.setMinimalRadius(radius / 20.0);
        // Radius used to compute the local point density for the neighbors
        // (the density is the number of points within that radius).
        usc.setPointDensityRadius(radius / 10.0);
        // Set the radius to compute the Local Reference Frame.
        usc.setLocalRadius(radius);
        //usc.compute(cloud1_usc);
        usc.compute(cloud1_sc);

        usc.setInputCloud(cloud2_keypoints.makeShared());
        usc.setSearchSurface(cloud2.makeShared());
        usc.compute(cloud2_sc);



    }





    void calculate_3dsc ( float radius )
    {

        // USC estimation object.
        pcl::ShapeContext3DEstimation<pcl::PointXYZ, pcl::Normal, pcl::ShapeContext1980> sc3d;
        sc3d.setInputCloud(cloud1_keypoints.makeShared());
        sc3d.setSearchSurface(cloud1.makeShared());

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_here1(new pcl::search::KdTree<pcl::PointXYZ>);
        sc3d.setSearchMethod(kdtree_here1);


        sc3d.setInputNormals(cloud1_normals.makeShared());

        // Search radius, to look for neighbors. It will also be the radius of the support sphere.
        sc3d.setRadiusSearch(radius);
        // The minimal radius value for the search sphere, to avoid being too sensitive
        // in bins close to the center of the sphere.
        sc3d.setMinimalRadius(radius / 20.0);
        // Radius used to compute the local point density for the neighbors
        // (the density is the number of points within that radius).
        sc3d.setPointDensityRadius(radius / 10.0);
        // Set the radius to compute the Local Reference Frame.
        sc3d.setRadiusSearch(radius);
        //usc.compute(cloud1_usc);
        sc3d.compute(cloud1_sc);

        sc3d.setInputCloud(cloud2_keypoints.makeShared());
        sc3d.setSearchSurface(cloud2.makeShared());
        sc3d.setInputNormals(cloud2_normals.makeShared());
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_here2(new pcl::search::KdTree<pcl::PointXYZ>);
        sc3d.setSearchMethod(kdtree_here2);
        sc3d.compute(cloud2_sc);



    }




    void calculate_rsd ( float radius )
    {

        // RSD estimation object.
        RSDEstimation<PointXYZ, Normal, PrincipalRadiiRSD> rsd;
        rsd.setInputCloud(cloud1_keypoints.makeShared());
        rsd.setSearchSurface(cloud1.makeShared());
        rsd.setInputNormals(cloud1_normals.makeShared());
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        rsd.setSearchMethod(kdtree);
        // Search radius, to look for neighbors. Note: the value given here has to be
        // larger than the radius used to estimate the normals.
        rsd.setRadiusSearch(radius);
        // Plane radius. Any radius larger than this is considered infinite (a plane).
        rsd.setPlaneRadius(2*radius);
        // Do we want to save the full distance-angle histograms?
        rsd.setSaveHistograms(false);

        rsd.compute(cloud1_rsd);

        //std::cout << "Done cloud1 " << endl;

        rsd.setInputCloud(cloud2_keypoints.makeShared());
        rsd.setInputNormals(cloud2_normals.makeShared());
        rsd.setSearchSurface(cloud2.makeShared());
        rsd.compute(cloud2_rsd);

        //std::cout << "Done cloud2" <<  endl;



    }



    void calculate_rops(float support_radius)
    {

        unsigned int number_of_partition_bins = 5;
        unsigned int number_of_rotations = 3;

        pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method (new pcl::search::KdTree<pcl::PointXYZ>);
        search_method->setInputCloud (cloud1.makeShared());

        std::vector <pcl::Vertices> triangles1;
        triangles1 = mesh1.polygons;

        pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram <135> > feature_estimator;
        feature_estimator.setSearchMethod (search_method);
        feature_estimator.setSearchSurface (cloud1.makeShared());
        feature_estimator.setInputCloud (cloud1_keypoints.makeShared());
        //feature_estimator.setIndices (indices);
        feature_estimator.setTriangles (triangles1);// changed to triangles.polygons for consistency :)
        feature_estimator.setRadiusSearch (support_radius);
        feature_estimator.setNumberOfPartitionBins (number_of_partition_bins);
        feature_estimator.setNumberOfRotations (number_of_rotations);
        feature_estimator.setSupportRadius (support_radius);

        feature_estimator.compute (histograms1);

        std::vector <pcl::Vertices> triangles2;
        triangles2 = mesh2.polygons;

        search_method->setInputCloud (cloud2.makeShared());

        feature_estimator.setSearchMethod (search_method);
        feature_estimator.setSearchSurface (cloud2.makeShared());
        feature_estimator.setInputCloud (cloud2_keypoints.makeShared());
        //feature_estimator.setIndices (indices);
        feature_estimator.setTriangles (triangles2);// changed to triangles.polygons for consistency :)

        feature_estimator.compute (histograms2);


    }













};


class two_indices
{
public:
    int one;
    int two;
};



