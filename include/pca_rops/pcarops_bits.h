#include <pca_rops/pcarops_headers_bits.h>
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

        fpfh.setNumberOfThreads(8);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh.setSearchMethod(kdtree);

        fpfh.setInputCloud(cloud1_keypoints.makeShared());
        fpfh.setSearchSurface(cloud1.makeShared());
        fpfh.setInputNormals(cloud1_normals.makeShared());
        fpfh.setNumberOfThreads(8);
        fpfh.compute(cloud1_fpfh);


        fpfh.setInputCloud(cloud2_keypoints.makeShared());
        fpfh.setSearchSurface(cloud2.makeShared());
        fpfh.setInputNormals(cloud2_normals.makeShared());
        fpfh.setNumberOfThreads(8);
        fpfh.compute(cloud2_fpfh);


    }



    void calculate_SpinImage ( float radius )
    {

        pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> > spin_image_descriptor(8, 0.5, 16);
        spin_image_descriptor.setInputCloud (cloud1_keypoints.makeShared());
        spin_image_descriptor.setInputNormals (cloud1_normals.makeShared());

        // Use the same KdTree from the normal estimation
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ>);
        spin_image_descriptor.setSearchMethod (kdtree);
        spin_image_descriptor.setRadiusSearch (radius);
        spin_image_descriptor.setSearchSurface(cloud1.makeShared());

        // Actually compute the spin images
        spin_image_descriptor.compute (cloud1_spin);

        spin_image_descriptor.setInputCloud (cloud2_keypoints.makeShared());
        spin_image_descriptor.setInputNormals (cloud2_normals.makeShared());
        spin_image_descriptor.setSearchSurface(cloud2.makeShared());

        // Use the same KdTree from the normal estimation
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree1 (new pcl::search::KdTree<pcl::PointXYZ>);
        spin_image_descriptor.setSearchMethod (kdtree1);
        spin_image_descriptor.setRadiusSearch (radius);

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



    void compute_cbshot()
    {
        compute_cbshot_from_SHOT( cloud1_shot, cloud1_cbshot);
        compute_cbshot_from_SHOT( cloud2_shot, cloud2_cbshot);
    }

    void compute_cbshot_from_SHOT(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here, std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            //int compact_shot[352] = { 0 };

            std::bitset < 352 > temp;
            temp.reset();

            for (int j = 0 ; j < 88 ; j++)
            {
                float vec[4] = { 0 };
                for (int k = 0 ; k < 4 ; k++)
                {
                    vec[k] = shot_descriptors_here[i].descriptor[ j*4 + k ];

                }

                // first bit -> if all are zeros , then set to invalid
                // TODO and or less than 0.002, then---then set to invalid---Must see--possiblity of outliers
                //int bin[4] = { 0 };

                std::bitset< 4 > bit;
                bit.reset();

                float sum = vec[0]+vec[1]+vec[2]+vec[3];

                //default ratio = 0.9
                float ratio = 0.9;

                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0)// is float and int comparision in IF --OK ?
                {
                    //bin[0] = bin[1] = bin[2] = bin[3] = 0;
                    // by default , they are all ZEROS
                }
                else if ( vec[0] > (ratio * (sum) ) )
                {
                    //bin[0] = 1;
                    bit.set(0);
                }
                else if ( vec[1] > (ratio * (sum) ) )
                {
                    //bin[1] = 1;
                    bit.set(1);
                }
                else if ( vec[2] > (ratio * (sum) ) )
                {
                    //bin[2] = 1;
                    bit.set(2);
                }
                else if ( vec[3] > (ratio * (sum) ) )
                {
                    //bin[3] = 1;
                    bit.set(3);
                }
                else if ( (vec[0]+vec[1]) > (ratio * (sum))  )
                {
                    //bin[0] = 1;
                    //bin[1] = 1;
                    bit.set(0);
                    bit.set(1);
                }
                else if ( (vec[1]+vec[2]) > (ratio * (sum)) )
                {
                    //bin[1] = 1;
                    //bin[2] = 1;
                    bit.set(1);
                    bit.set(2);
                }

                else if ( (vec[2]+vec[3]) > (ratio * (sum)) )
                {
                    //bin[2] = 1;
                    //bin[3] = 1;
                    bit.set(2);
                    bit.set(3);
                }
                else if ( (vec[0]+vec[3]) > (ratio * (sum)) )
                {
                    //bin[0] = 1;
                    //bin[3] = 1;
                    bit.set(0);
                    bit.set(3);
                }
                else if ( (vec[1]+vec[3]) > (ratio * (sum)) )
                {
                    //bin[1] = 1;
                    //bin[3] = 1;
                    bit.set(1);
                    bit.set(3);
                }
                else if ( (vec[0]+vec[2]) > (ratio * (sum)) )
                {
                    //bin[0] = 1;
                    //bin[2] = 1;
                    bit.set(0);
                    bit.set(2);
                }
                else if ( (vec[0]+ vec[1] +vec[2]) > (ratio * (sum)) )
                {
                    //bin[0] = 1;
                    //bin[1] = 1;
                    //bin[2] = 1;
                    bit.set(0);
                    bit.set(1);
                    bit.set(2);
                }
                else if ( (vec[1]+ vec[2] +vec[3]) > (ratio * (sum)) )
                {
                    //bin[1] = 1;
                    //bin[2] = 1;
                    //bin[3] = 1;
                    bit.set(1);
                    bit.set(2);
                    bit.set(3);
                }
                else if ( (vec[0]+ vec[2] +vec[3]) > (ratio * (sum)) )
                {
                    //bin[0] = 1;
                    //bin[2] = 1;
                    //bin[3] = 1;
                    bit.set(0);
                    bit.set(2);
                    bit.set(3);
                }
                else if ( (vec[0]+ vec[1] +vec[3]) > (ratio * (sum)) )
                {
                    //bin[0] = 1;
                    //bin[1] = 1;
                    //bin[3] = 1;
                    bit.set(0);
                    bit.set(1);
                    bit.set(3);
                }
                else
                {
                    //bin[0] = 1;
                    //bin[1] = 1;
                    //bin[2] = 1;
                    //bin[3] = 1;
                    bit.set(0);
                    bit.set(1);
                    bit.set(2);
                    bit.set(3);
                }

                //compact_shot[j*4] = bin[0];
                //compact_shot[(j*4) + 1] = bin[1];
                //compact_shot[(j*4) + 2] = bin[2];
                //compact_shot[(j*4) + 3] = bin[3];

                if (bit.test(0))
                    temp.set(j*4);

                if (bit.test(1))
                    temp.set(j*4 + 1);

                if (bit.test(2))
                    temp.set(j*4 + 2);

                if (bit.test(3))
                    temp.set(j*4 + 3);


            }

            CBSHOT_descriptors[i].bits = temp;

            //            for (int j = 0; j < 352; j++)
            //            {
            //                CBSHOT_descriptors[i].bins[j] = compact_shot[j];
            //                if (compact_shot[j] == 1)
            //                CBSHOT_descriptors[i].bits.set(j);
            //            }

        }

    }






    void compute_adaptive_cbshot()
    {
        //adaptive_compute_cbshot_from_SHOT4( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT4( cloud2_shot, cloud2_cbshot);

        //adaptive_compute_cbshot_from_SHOT5( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT5( cloud2_shot, cloud2_cbshot);

        //adaptive_compute_cbshot_from_SHOT6( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT6( cloud2_shot, cloud2_cbshot);

        //adaptive_compute_cbshot_from_SHOT7( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT7( cloud2_shot, cloud2_cbshot);

        // best best best
        adaptive_compute_cbshot_from_SHOT_pair( cloud1_shot, cloud1_cbshot);
        adaptive_compute_cbshot_from_SHOT_pair( cloud2_shot, cloud2_cbshot);

        // Overlap is not good as it is failing in some cases by producing false correspondences!
        //adaptive_CBSHOT_OVERLAP( cloud1_shot, cloud1_cbshot);
        //cout << "here" <<endl;
        //adaptive_CBSHOT_OVERLAP( cloud2_shot, cloud2_cbshot);

    }



    void compute_adaptive_rops()
    {
        //adaptive_compute_cbshot_from_SHOT4( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT4( cloud2_shot, cloud2_cbshot);

        //adaptive_compute_cbshot_from_SHOT5( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT5( cloud2_shot, cloud2_cbshot);

        //adaptive_compute_cbshot_from_SHOT6( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT6( cloud2_shot, cloud2_cbshot);

        //adaptive_compute_cbshot_from_SHOT7( cloud1_shot, cloud1_cbshot);
        //adaptive_compute_cbshot_from_SHOT7( cloud2_shot, cloud2_cbshot);

        // best best best
        //adaptive_compute_cbrops_from_ROPS_pair( histograms1, cloud1_cbrops);
        //adaptive_compute_cbrops_from_ROPS_pair( histograms2, cloud2_cbrops);

        // Overlap is not good as it is failing in some cases by producing false correspondences!
        //adaptive_CBSHOT_OVERLAP( cloud1_shot, cloud1_cbshot);
        //cout << "here" <<endl;
        //adaptive_CBSHOT_OVERLAP( cloud2_shot, cloud2_cbshot);

    }



    /**************************************
 */
    void adaptive_compute_cbshot_from_SHOT4(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here,
                                            std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        //cout << " \n\n adaptive CBSHOT 4 \n\n" << endl;
        int n = 4;

        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 352 > temp;
            temp.reset();

            for (int j = 0 ; j < 352/n ; j++)
            {
                //float vec[n] = { 0 };
                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);
                for (int k = 0 ; k < n ; k++)
                {
                    vec[k] = shot_descriptors_here[i].descriptor[ j*n + k ];

                }
                std::bitset< 4 > bit;// fixed---
                bit.reset();

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;

                int level0 = 0;

                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0)
                {
                    level0 = 1;
                }


                int level1 = 0;
                if (level0 == 0)
                {
                    for (int l = 0; l < n and level1 == 0; l++)
                    {
                        //cout << l << endl;
                        if (vec[l] > part_sum)
                        {
                            level1 = 1;
                            bit.set(l);
                            //cout << "ia m here" << endl;
                        }
                    }
                }

                int level2 = 0;
                if (!level1 and !level0)
                {
                    for (int l = 0; l < n and level2 == 0; l++)
                    {
                        for (int m = 0; m < l and level2 == 0; m++)
                        {
                            //cout << l << m << endl;
                            if (vec[l] + vec[m] > part_sum)
                            {
                                level2 = 1;
                                bit.set(l);
                                bit.set(m);
                                //cout << "i am here here" << endl;
                            }
                        }
                    }
                }



                int level3 = 0;
                if (!level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level3 == 0; l++)
                    {
                        for (int m = 0; m < l and level3 == 0; m++)
                        {
                            for (int o = 0; o < m and level3 == 0; o++)
                            {
                                //cout << l << m << n << endl;
                                if (vec[l] + vec[m] + vec[o] > part_sum)
                                {
                                    level3 = 1;
                                    bit.set(l);
                                    bit.set(m);
                                    bit.set(o);
                                    //cout << "i am here here here" << endl;
                                }
                            }
                        }
                    }
                }



                if (!level3 and !level2 and !level1 and !level0)
                {
                    bit.set();
                    //cout << " i am here here here here " << endl;
                }

                //cout << bit << endl;

                for (int l = 0; l < n; l++)
                {
                    if (bit.test(l))
                    {
                        temp.set(j*n + l);
                    }
                }
            }

            CBSHOT_descriptors[i].bits = temp;

        }

    }



    /***********************************************/




    /*********************************************************************************
 */
    void adaptive_compute_cbshot_from_SHOT5(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here,
                                            std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        cout << " \n\n adaptive CBSHOT 5 \n\n" << endl;
        int n = 5;

        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 352 > temp;
            temp.reset();

            int count;
            if (352 % n == 0)
            {
                count = 352/n;
            }
            else count = (352/n) -1;

            for (int j = 0 ; j < count ; j++)
            {
                //float vec[n] = { 0 };
                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);
                for (int k = 0 ; k < n ; k++)
                {
                    vec[k] = shot_descriptors_here[i].descriptor[ j*n + k ];

                }
                std::bitset< 5 > bit;// fixed---
                bit.reset();

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;

                int level0 = 0;

                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0 and vec[4] == 0)
                {
                    level0 = 1;
                }


                int level1 = 0;
                if (level0 == 0)
                {
                    for (int l = 0; l < n and level1 == 0; l++)
                    {
                        //cout << l << endl;
                        if (vec[l] > part_sum)
                        {
                            level1 = 1;
                            bit.set(l);
                            //cout << "ia m here" << endl;
                        }
                    }
                }

                int level2 = 0;
                if (!level1 and !level0)
                {
                    for (int l = 0; l < n and level2 == 0; l++)
                    {
                        for (int m = 0; m < l and level2 == 0; m++)
                        {
                            //cout << l << m << endl;
                            if (vec[l] + vec[m] > part_sum)
                            {
                                level2 = 1;
                                bit.set(l);
                                bit.set(m);
                                //cout << "i am here here" << endl;
                            }
                        }
                    }
                }



                int level3 = 0;
                if (!level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level3 == 0; l++)
                    {
                        for (int m = 0; m < l and level3 == 0; m++)
                        {
                            for (int o = 0; o < m and level3 == 0; o++)
                            {
                                //cout << l << m << n << endl;
                                if (vec[l] + vec[m] + vec[o] > part_sum)
                                {
                                    level3 = 1;
                                    bit.set(l);
                                    bit.set(m);
                                    bit.set(o);
                                    //cout << "i am here here here" << endl;
                                }
                            }
                        }
                    }
                }


                int level4 = 0;
                if (!level3 and !level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level4 == 0; l++)
                    {
                        for (int m = 0; m < l and level4 == 0; m++)
                        {
                            for (int o = 0; o < m and level4 == 0; o++)
                            {
                                for (int p = 0; p < o and level4 == 0; p++)
                                {
                                    //cout << l << m << n << endl;
                                    if (vec[l] + vec[m] + vec[o] + vec[p] > part_sum)
                                    {
                                        level4 = 1;
                                        bit.set(l);
                                        bit.set(m);
                                        bit.set(o);
                                        bit.set(p);
                                        //cout << "i am here here here" << endl;
                                    }
                                }
                            }
                        }
                    }
                }



                if (!level4 and !level3 and !level2 and !level1 and !level0)
                {
                    bit.set();
                    //cout << " i am here here here here " << endl;
                }

                //cout << bit << endl;

                for (int l = 0; l < n; l++)
                {
                    if (bit.test(l))
                    {
                        temp.set(j*n + l);
                    }
                }
            }

            //            if (352%n != 0)
            //            {
            //                std::bitset<2> bit1;
            //                bit1.reset();

            //            }

            CBSHOT_descriptors[i].bits = temp;

        }

    }



    /***************************************************************************/






    /*********************************************************************************
 */
    void adaptive_compute_cbshot_from_SHOT6(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here,
                                            std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        cout << " \n\n adaptive CBSHOT 6 \n\n" << endl;
        int n = 6;

        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 352 > temp;
            temp.reset();

            int count;
            if (352 % n == 0)
            {
                count = 352/n;
            }
            else count = (352/n) -1;

            for (int j = 0 ; j < count ; j++)
            {
                //float vec[n] = { 0 };
                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);
                for (int k = 0 ; k < n ; k++)
                {
                    vec[k] = shot_descriptors_here[i].descriptor[ j*n + k ];

                }
                std::bitset< 6 > bit;// fixed---
                bit.reset();

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;

                int level0 = 0;

                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0 and vec[4] == 0 and vec[5] == 0)
                {
                    level0 = 1;
                }


                int level1 = 0;
                if (level0 == 0)
                {
                    for (int l = 0; l < n and level1 == 0; l++)
                    {
                        //cout << l << endl;
                        if (vec[l] > part_sum)
                        {
                            level1 = 1;
                            bit.set(l);
                            //cout << "ia m here" << endl;
                        }
                    }
                }

                int level2 = 0;
                if (!level1 and !level0)
                {
                    for (int l = 0; l < n and level2 == 0; l++)
                    {
                        for (int m = 0; m < l and level2 == 0; m++)
                        {
                            //cout << l << m << endl;
                            if (vec[l] + vec[m] > part_sum)
                            {
                                level2 = 1;
                                bit.set(l);
                                bit.set(m);
                                //cout << "i am here here" << endl;
                            }
                        }
                    }
                }



                int level3 = 0;
                if (!level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level3 == 0; l++)
                    {
                        for (int m = 0; m < l and level3 == 0; m++)
                        {
                            for (int o = 0; o < m and level3 == 0; o++)
                            {
                                //cout << l << m << n << endl;
                                if (vec[l] + vec[m] + vec[o] > part_sum)
                                {
                                    level3 = 1;
                                    bit.set(l);
                                    bit.set(m);
                                    bit.set(o);
                                    //cout << "i am here here here" << endl;
                                }
                            }
                        }
                    }
                }


                int level4 = 0;
                if (!level3 and !level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level4 == 0; l++)
                    {
                        for (int m = 0; m < l and level4 == 0; m++)
                        {
                            for (int o = 0; o < m and level4 == 0; o++)
                            {
                                for (int p = 0; p < o and level4 == 0; p++)
                                {
                                    //cout << l << m << n << endl;
                                    if (vec[l] + vec[m] + vec[o] + vec[p] > part_sum)
                                    {
                                        level4 = 1;
                                        bit.set(l);
                                        bit.set(m);
                                        bit.set(o);
                                        bit.set(p);
                                        //cout << "i am here here here" << endl;
                                    }
                                }
                            }
                        }
                    }
                }


                int level5 = 0;
                if (!level4 and !level3 and !level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level5 == 0; l++)
                    {
                        for (int m = 0; m < l and level5 == 0; m++)
                        {
                            for (int o = 0; o < m and level5 == 0; o++)
                            {
                                for (int p = 0; p < o and level5 == 0; p++)
                                {
                                    for (int q = 0; q < p and level5 == 0; q++)
                                    {
                                        //cout << l << m << n << endl;
                                        if (vec[l] + vec[m] + vec[o] + vec[p] + vec[q] > part_sum)
                                        {
                                            level5 = 1;
                                            bit.set(l);
                                            bit.set(m);
                                            bit.set(o);
                                            bit.set(p);
                                            bit.set(q);
                                            //cout << "i am here here here" << endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }


                if (!level5 and !level4 and !level3 and !level2 and !level1 and !level0)
                {
                    bit.set();
                    //cout << " i am here here here here " << endl;
                }

                //cout << bit << endl;

                for (int l = 0; l < n; l++)
                {
                    if (bit.test(l))
                    {
                        temp.set(j*n + l);
                    }
                }
            }

            //            if (352%n != 0)
            //            {
            //                std::bitset<2> bit1;
            //                bit1.reset();

            //            }

            CBSHOT_descriptors[i].bits = temp;

        }

    }



    /***************************************************************************/





    /*********************************************************************************
 */
    void adaptive_compute_cbshot_from_SHOT7(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here,
                                            std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        cout << " \n\n adaptive CBSHOT 7 \n\n" << endl;
        int n = 7;

        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 352 > temp;
            temp.reset();

            int count;
            if (352 % n == 0)
            {
                count = 352/n;
            }
            else count = (352/n) -1;

            for (int j = 0 ; j < count ; j++)
            {
                //float vec[n] = { 0 };
                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);
                for (int k = 0 ; k < n ; k++)
                {
                    vec[k] = shot_descriptors_here[i].descriptor[ j*n + k ];

                }
                std::bitset< 7 > bit;// fixed---
                bit.reset();

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;

                int level0 = 0;

                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0 and vec[4] == 0 and vec[5] == 0 and vec[6] == 0)
                {
                    level0 = 1;
                }


                int level1 = 0;
                if (level0 == 0)
                {
                    for (int l = 0; l < n and level1 == 0; l++)
                    {
                        //cout << l << endl;
                        if (vec[l] > part_sum)
                        {
                            level1 = 1;
                            bit.set(l);
                            //cout << "ia m here" << endl;
                        }
                    }
                }

                int level2 = 0;
                if (!level1 and !level0)
                {
                    for (int l = 0; l < n and level2 == 0; l++)
                    {
                        for (int m = 0; m < l and level2 == 0; m++)
                        {
                            //cout << l << m << endl;
                            if (vec[l] + vec[m] > part_sum)
                            {
                                level2 = 1;
                                bit.set(l);
                                bit.set(m);
                                //cout << "i am here here" << endl;
                            }
                        }
                    }
                }



                int level3 = 0;
                if (!level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level3 == 0; l++)
                    {
                        for (int m = 0; m < l and level3 == 0; m++)
                        {
                            for (int o = 0; o < m and level3 == 0; o++)
                            {
                                //cout << l << m << n << endl;
                                if (vec[l] + vec[m] + vec[o] > part_sum)
                                {
                                    level3 = 1;
                                    bit.set(l);
                                    bit.set(m);
                                    bit.set(o);
                                    //cout << "i am here here here" << endl;
                                }
                            }
                        }
                    }
                }


                int level4 = 0;
                if (!level3 and !level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level4 == 0; l++)
                    {
                        for (int m = 0; m < l and level4 == 0; m++)
                        {
                            for (int o = 0; o < m and level4 == 0; o++)
                            {
                                for (int p = 0; p < o and level4 == 0; p++)
                                {
                                    //cout << l << m << n << endl;
                                    if (vec[l] + vec[m] + vec[o] + vec[p] > part_sum)
                                    {
                                        level4 = 1;
                                        bit.set(l);
                                        bit.set(m);
                                        bit.set(o);
                                        bit.set(p);
                                        //cout << "i am here here here" << endl;
                                    }
                                }
                            }
                        }
                    }
                }


                int level5 = 0;
                if (!level4 and !level3 and !level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level5 == 0; l++)
                    {
                        for (int m = 0; m < l and level5 == 0; m++)
                        {
                            for (int o = 0; o < m and level5 == 0; o++)
                            {
                                for (int p = 0; p < o and level5 == 0; p++)
                                {
                                    for (int q = 0; q < p and level5 == 0; q++)
                                    {
                                        //cout << l << m << n << endl;
                                        if (vec[l] + vec[m] + vec[o] + vec[p] + vec[q] > part_sum)
                                        {
                                            level5 = 1;
                                            bit.set(l);
                                            bit.set(m);
                                            bit.set(o);
                                            bit.set(p);
                                            bit.set(q);
                                            //cout << "i am here here here" << endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }


                int level6 = 0;
                if (!level5 and !level4 and !level3 and !level2 and !level1 and !level0)
                {
                    for (int l = 0; l < n and level6 == 0; l++)
                    {
                        for (int m = 0; m < l and level6 == 0; m++)
                        {
                            for (int o = 0; o < m and level6 == 0; o++)
                            {
                                for (int p = 0; p < o and level6 == 0; p++)
                                {
                                    for (int q = 0; q < p and level6 == 0; q++)
                                    {
                                        for (int r = 0; r < q and level6 == 0; r++)
                                        {
                                            //cout << l << m << n << endl;
                                            if (vec[l] + vec[m] + vec[o] + vec[p] + vec[q] + vec[r] > part_sum)
                                            {
                                                level6 = 1;
                                                bit.set(l);
                                                bit.set(m);
                                                bit.set(o);
                                                bit.set(p);
                                                bit.set(q);
                                                bit.set(r);
                                                //cout << "i am here here here" << endl;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }


                if (!level6 and !level5 and !level4 and !level3 and !level2 and !level1 and !level0)
                {
                    bit.set();
                    //cout << " i am here here here here " << endl;
                }

                //cout << bit << endl;

                for (int l = 0; l < n; l++)
                {
                    if (bit.test(l))
                    {
                        temp.set(j*n + l);
                    }
                }
            }

            //            if (352%n != 0)
            //            {
            //                std::bitset<2> bit1;
            //                bit1.reset();

            //            }

            CBSHOT_descriptors[i].bits = temp;

        }

    }



    /***************************************************************************/






    /**************************************
 */
    void adaptive_compute_cbshot_from_SHOT_pair(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here,
                                                std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        //cout << " \n\n aaaaaadaptive CBSHOT 4 \n\n" << endl;

        int n = 4;// CHANGE CHANGE CHANGE CHANGE CHANGE

        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 352 > temp;
            temp.reset();

            for (int j = 0 ; j < (int)352/n ; j++)
            {
                std::vector<mypair> vector_of_pairs;
                vector_of_pairs.resize(n);

                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);

                for (int k = 0 ; k < n ; k++)
                {

                    vec[k] = shot_descriptors_here[i].descriptor[ j*n + k ];

                    vector_of_pairs[k].first = shot_descriptors_here[i].descriptor[ j*n + k ];
                    vector_of_pairs[k].second = k;
                    //cout << "j : " << j << " k : "<<k<<endl;
                    //if ((j*n+k) > 352)
                    //{
                    //    cout << vec[k] << endl;
                    //}
                }

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;// CHANGE CHANGE CHANGE CHANGE CHANGE

                std::bitset< 4 > bit;// CHANGE CHANGE CHANGE CHANGE CHANGE
                bit.reset();

                int level = 0;
                for (int m = 0; m < n and level == 0 ; m++)
                {
                    if(vector_of_pairs[m].first == 0)
                    {
                        if (m == n-1)
                        {
                            //bit.reset();// do nothing!!!
                        }
                        continue;
                    }
                    else level = 1;
                }



                // using function as comp
                if(level)
                {
                    std::sort (vector_of_pairs.begin(), vector_of_pairs.end(), comparator_here);
                    //for (int l = 0; l < n; l++)
                    //cout << vector_of_pairs[l].first << " " << vector_of_pairs[l].second << endl;
                }


                int check = 0;
                int bit_count;
                if(level)
                {


                    for (int l = 0; l < n and check == 0; l++ )
                    {
                        float temp_sum = 0;

                        for (int m = 0; m <= l; m++)
                        {
                            temp_sum = temp_sum + vector_of_pairs[m].first;
                            bit_count = m;
                        }
                        if (temp_sum > part_sum)
                        {
                            check = 1;
                            for (int o = 0; o <= bit_count; o++)
                            {
                                bit.set(vector_of_pairs[o].second);
                            }
                        }
                    }



                    for (int l = 0; l < n; l++)
                    {
                        if (bit.test(l))
                        {
                            temp.set(j*n + l);
                        }
                    }
                }
            }

            CBSHOT_descriptors[i].bits = temp;
            //for (int g = 0; g < 352; g++)
            //cout << "temp" << temp << endl;


        }

    }



    /***********************************************/




    /**************************************
 */
    void adaptive_compute_cbrops_from_ROPS_pair(pcl::PointCloud<pcl::Histogram <135> >& histograms,
                                                std::vector<cbrops_descriptor>& cloud_cbrops)
    {
        //cout << " \n\n aaaaaadaptive CBSHOT 4 \n\n" << endl;

        int n = 4;// CHANGE CHANGE CHANGE CHANGE CHANGE

        cloud_cbrops.resize(histograms.size());
        for (int i = 0; i < (int)histograms.size(); i++)
        {
            std::bitset < 135 > temp;
            temp.reset();

            for (int j = 0 ; j < (int)135/n ; j++)
            {
                std::vector<mypair> vector_of_pairs;
                vector_of_pairs.resize(n);

                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);

                for (int k = 0 ; k < n ; k++)
                {

                    vec[k] = histograms[i].histogram[ j*n + k ];

                    vector_of_pairs[k].first = histograms[i].histogram[ j*n + k ];
                    vector_of_pairs[k].second = k;
                    //cout << "j : " << j << " k : "<<k<<endl;
                    //if ((j*n+k) > 352)
                    //{
                    //    cout << vec[k] << endl;
                    //}
                }

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;// CHANGE CHANGE CHANGE CHANGE CHANGE

                std::bitset< 4 > bit;// CHANGE CHANGE CHANGE CHANGE CHANGE
                bit.reset();

                int level = 0;
                for (int m = 0; m < n and level == 0 ; m++)
                {
                    if(vector_of_pairs[m].first == 0)
                    {
                        if (m == n-1)
                        {
                            //bit.reset();// do nothing!!!
                        }
                        continue;
                    }
                    else level = 1;
                }



                // using function as comp
                if(level)
                {
                    std::sort (vector_of_pairs.begin(), vector_of_pairs.end(), comparator_here);
                    //for (int l = 0; l < n; l++)
                    //cout << vector_of_pairs[l].first << " " << vector_of_pairs[l].second << endl;
                }


                int check = 0;
                int bit_count;
                if(level)
                {


                    for (int l = 0; l < n and check == 0; l++ )
                    {
                        float temp_sum = 0;

                        for (int m = 0; m <= l; m++)
                        {
                            temp_sum = temp_sum + vector_of_pairs[m].first;
                            bit_count = m;
                        }
                        if (temp_sum > part_sum)
                        {
                            check = 1;
                            for (int o = 0; o <= bit_count; o++)
                            {
                                bit.set(vector_of_pairs[o].second);
                            }
                        }
                    }



                    for (int l = 0; l < n; l++)
                    {
                        if (bit.test(l))
                        {
                            temp.set(j*n + l);
                        }
                    }
                }
            }

            cloud_cbrops[i].bits = temp;
            //for (int g = 0; g < 352; g++)
            //cout << "temp" << temp << endl;


        }

    }



    /***********************************************/









    /**************************************
 */
    void adaptive_CBSHOT_OVERLAP(pcl::PointCloud<pcl::SHOT352>& shot_descriptors_here,
                                 std::vector<cbshot_descriptor>& CBSHOT_descriptors)
    {
        cout << " \n\n aaaaaadaptive CBSHOT 4 \n\n" << endl;

        int n = 7;// CHANGE CHANGE CHANGE CHANGE CHANGE
        int z = 2;
        int y = 5;

        CBSHOT_descriptors.resize(shot_descriptors_here.size());
        for (int i = 0; i < (int)shot_descriptors_here.size(); i++)
        {
            std::bitset < 483 > temp;
            temp.reset();

            for (int j = 0 ; j < ((int)((352-n)/y)) /*+ 1*/ ; j++)
            {
                std::vector<mypair> vector_of_pairs;
                vector_of_pairs.resize(n);

                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);

                if (j == 0)// important ...dont put '=', you should put "=="
                {
                    for (int k = 0 ; k < n ; k++)
                    {

                        vec[k] = shot_descriptors_here[i].descriptor[ (j*n + k) ];

                        vector_of_pairs[k].first = shot_descriptors_here[i].descriptor[ (j*n + k) ];
                        vector_of_pairs[k].second = k;
                    }
                }

                else
                {

                    for (int k = 0 ; k < n ; k++)
                    {

                        vec[k] = shot_descriptors_here[i].descriptor[ (j*n + k) - z];

                        vector_of_pairs[k].first = shot_descriptors_here[i].descriptor[ (j*n + k) - z ];
                        vector_of_pairs[k].second = k;
                    }
                }

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;// CHANGE CHANGE CHANGE CHANGE CHANGE

                std::bitset< 7 > bit;// CHANGE CHANGE CHANGE CHANGE CHANGE
                bit.reset();

                int level = 0;
                for (int m = 0; m < n and level == 0 ; m++)
                {
                    if(vector_of_pairs[m].first == 0)
                    {
                        if (m == n-1)
                        {
                            //bit.reset();// do nothing!!!
                        }
                        continue;
                    }
                    else level = 1;
                }



                // using function as comp
                if(level)
                {
                    std::sort (vector_of_pairs.begin(), vector_of_pairs.end(), comparator_here);
                    //for (int l = 0; l < n; l++)
                    //cout << vector_of_pairs[l].first << " " << vector_of_pairs[l].second << endl;
                }


                int check = 0;
                int bit_count;
                if(level)
                {


                    for (int l = 0; l < n and check == 0; l++ )
                    {
                        float temp_sum = 0;

                        for (int m = 0; m <= l; m++)
                        {
                            temp_sum = temp_sum + vector_of_pairs[m].first;
                            bit_count = m;
                        }
                        if (temp_sum > part_sum)
                        {
                            check = 1;
                            for (int o = 0; o <= bit_count; o++)
                            {
                                bit.set(vector_of_pairs[o].second);
                            }
                        }
                    }



                    for (int l = 0; l < n; l++)
                    {
                        if (bit.test(l))
                        {
                            temp.set(j*n + l);
                        }
                    }
                }
            }

            CBSHOT_descriptors[i].overlap_bits = temp;


        }

    }



    /***********************************************/








    void compute_cbfpfh()
    {
        //compute_cbfpfh_from_FPFH( cloud1_fpfh, cloud1_cbfpfh);
        //compute_cbfpfh_from_FPFH( cloud2_fpfh, cloud2_cbfpfh);

        compute_cbfpfh_from_FPFH_adaptive( cloud1_fpfh, cloud1_cbfpfh);
        compute_cbfpfh_from_FPFH_adaptive( cloud2_fpfh, cloud2_cbfpfh);
    }






    void compute_cbfpfh_from_FPFH_adaptive(pcl::PointCloud<pcl::FPFHSignature33>& fpfh_descriptors_here, std::vector<cbfpfh_descriptor>& CBFPFH_descriptors)
    {
        cout << " \n\n aaaaaadaptive CB FPFH 33 \n\n" << endl;

        int n = 4;// CHANGE CHANGE CHANGE CHANGE CHANGE

        CBFPFH_descriptors.resize(fpfh_descriptors_here.size());
        for (int i = 0; i < (int)fpfh_descriptors_here.size(); i++)
        {
            std::bitset < 33 > temp;
            temp.reset();

            for (int j = 0 ; j < (int)33/n ; j++)
            {
                std::vector<mypair> vector_of_pairs;
                vector_of_pairs.resize(n);

                std::vector<float> vec;// every element is properly set in vec[n]
                vec.resize(n);

                for (int k = 0 ; k < n ; k++)
                {

                    vec[k] = fpfh_descriptors_here[i].histogram[ j*n + k ];

                    vector_of_pairs[k].first = fpfh_descriptors_here[i].histogram[ j*n + k ];
                    vector_of_pairs[k].second = k;
                    //cout << "j : " << j << " k : "<<k<<endl;
                    //if ((j*n+k) > 352)
                    //{
                    //    cout << vec[k] << endl;
                    //}
                }

                float sum = 0;
                for(int l = 0; l < n; l++)
                    sum = sum + vec[l];

                float part_sum = 0.9 * sum;// CHANGE CHANGE CHANGE CHANGE CHANGE

                std::bitset< 4 > bit;// CHANGE CHANGE CHANGE CHANGE CHANGE
                bit.reset();

                int level = 0;
                for (int m = 0; m < n and level == 0 ; m++)
                {
                    if(vector_of_pairs[m].first == 0)
                    {
                        if (m == n-1)
                        {
                            //bit.reset();// do nothing!!!
                        }
                        continue;
                    }
                    else level = 1;
                }



                // using function as comp
                if(level)
                {
                    std::sort (vector_of_pairs.begin(), vector_of_pairs.end(), comparator_here);
                    //for (int l = 0; l < n; l++)
                    //cout << vector_of_pairs[l].first << " " << vector_of_pairs[l].second << endl;
                }


                int check = 0;
                int bit_count;
                if(level)
                {


                    for (int l = 0; l < n and check == 0; l++ )
                    {
                        float temp_sum = 0;

                        for (int m = 0; m <= l; m++)
                        {
                            temp_sum = temp_sum + vector_of_pairs[m].first;
                            bit_count = m;
                        }
                        if (temp_sum > part_sum)
                        {
                            check = 1;
                            for (int o = 0; o <= bit_count; o++)
                            {
                                bit.set(vector_of_pairs[o].second);
                            }
                        }
                    }



                    for (int l = 0; l < n; l++)
                    {
                        if (bit.test(l))
                        {
                            temp.set(j*n + l);
                        }
                    }
                }
            }

            CBFPFH_descriptors[i].bits = temp;
            //for (int g = 0; g < 352; g++)
            //cout << "temp" << temp << endl;


        }

    }











    //    void compute_cbfpfh_from_FPFH(pcl::PointCloud<pcl::FPFHSignature33>& fpfh_descriptors_here, std::vector<cbfpfh_descriptor>& CBFPFH_descriptors)
    //    {
    //        CBFPFH_descriptors.resize(fpfh_descriptors_here.size());
    //        for (int i = 0; i < (int)fpfh_descriptors_here.size(); i++)
    //        {
    //            //int compact_shot[352] = { 0 };

    //            std::bitset < 36 > temp;
    //            temp.reset();

    //            for (int j = 0 ; j < 9 ; j++)
    //            {
    //                float vec[4] = { 0 };
    //                for (int k = 0 ; k < 4 ; k++)
    //                {
    //                    vec[k] = fpfh_descriptors_here[i].histogram[ j*4 + k ];

    //                }

    //                // first bit -> if all are zeros , then set to invalid
    //                // TODO and or less than 0.002, then---then set to invalid---Must see--possiblity of outliers
    //                //int bin[4] = { 0 };

    //                std::bitset< 4 > bit;
    //                bit.reset();

    //                float sum = vec[0]+vec[1]+vec[2]+vec[3];

    //                if (vec[0] == 0 and vec [1] == 0 and vec[2] == 0 and vec[3] == 0)// is float and int comparision in IF --OK ?
    //                {
    //                    //bin[0] = bin[1] = bin[2] = bin[3] = 0;
    //                    // by default , they are all ZEROS
    //                }
    //                else if ( vec[0] > (0.9 * (sum) ) )
    //                {
    //                    //bin[0] = 1;
    //                    bit.set(0);
    //                }
    //                else if ( vec[1] > (0.9 * (sum) ) )
    //                {
    //                    //bin[1] = 1;
    //                    bit.set(1);
    //                }
    //                else if ( vec[2] > (0.9 * (sum) ) )
    //                {
    //                    //bin[2] = 1;
    //                    bit.set(2);
    //                }
    //                else if ( vec[3] > (0.9 * (sum) ) )
    //                {
    //                    //bin[3] = 1;
    //                    bit.set(3);
    //                }
    //                else if ( (vec[0]+vec[1]) > (0.9 * (sum))  )
    //                {
    //                    //bin[0] = 1;
    //                    //bin[1] = 1;
    //                    bit.set(0);
    //                    bit.set(1);
    //                }
    //                else if ( (vec[1]+vec[2]) > (0.9 * (sum)) )
    //                {
    //                    //bin[1] = 1;
    //                    //bin[2] = 1;
    //                    bit.set(1);
    //                    bit.set(2);
    //                }

    //                else if ( (vec[2]+vec[3]) > (0.9 * (sum)) )
    //                {
    //                    //bin[2] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(2);
    //                    bit.set(3);
    //                }
    //                else if ( (vec[0]+vec[3]) > (0.9 * (sum)) )
    //                {
    //                    //bin[0] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(0);
    //                    bit.set(3);
    //                }
    //                else if ( (vec[1]+vec[3]) > (0.9 * (sum)) )
    //                {
    //                    //bin[1] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(1);
    //                    bit.set(3);
    //                }
    //                else if ( (vec[0]+vec[2]) > (0.9 * (sum)) )
    //                {
    //                    //bin[0] = 1;
    //                    //bin[2] = 1;
    //                    bit.set(0);
    //                    bit.set(2);
    //                }
    //                else if ( (vec[0]+ vec[1] +vec[2]) > (0.9 * (sum)) )
    //                {
    //                    //bin[0] = 1;
    //                    //bin[1] = 1;
    //                    //bin[2] = 1;
    //                    bit.set(0);
    //                    bit.set(1);
    //                    bit.set(2);
    //                }
    //                else if ( (vec[1]+ vec[2] +vec[3]) > (0.9 * (sum)) )
    //                {
    //                    //bin[1] = 1;
    //                    //bin[2] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(1);
    //                    bit.set(2);
    //                    bit.set(3);
    //                }
    //                else if ( (vec[0]+ vec[2] +vec[3]) > (0.9 * (sum)) )
    //                {
    //                    //bin[0] = 1;
    //                    //bin[2] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(0);
    //                    bit.set(2);
    //                    bit.set(3);
    //                }
    //                else if ( (vec[0]+ vec[1] +vec[3]) > (0.9 * (sum)) )
    //                {
    //                    //bin[0] = 1;
    //                    //bin[1] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(0);
    //                    bit.set(1);
    //                    bit.set(3);
    //                }
    //                else
    //                {
    //                    //bin[0] = 1;
    //                    //bin[1] = 1;
    //                    //bin[2] = 1;
    //                    //bin[3] = 1;
    //                    bit.set(0);
    //                    bit.set(1);
    //                    bit.set(2);
    //                    bit.set(3);
    //                }

    //                //compact_shot[j*4] = bin[0];
    //                //compact_shot[(j*4) + 1] = bin[1];
    //                //compact_shot[(j*4) + 2] = bin[2];
    //                //compact_shot[(j*4) + 3] = bin[3];

    //                if (bit.test(0))
    //                    temp.set(j*4);

    //                if (bit.test(1))
    //                    temp.set(j*4 + 1);

    //                if (bit.test(2))
    //                    temp.set(j*4 + 2);

    //                if (bit.test(3))
    //                    temp.set(j*4 + 3);

    //            }

    //            CBFPFH_descriptors[i].bits = temp;

    //        }

    //    }
















    //    double hamming_distance_fix(int* a, int* b, int n)
    //    {
    //        int disti = 0;
    //        for(int i=0; i<n; i++)
    //        {
    //           disti += (a[i] != b[i]);
    //        }
    //        return 1.0*disti/n;
    //    }



    int hamming_distance_fix(int* a, int* b, int n)
    {
        int disti = 0;
        for(int i=0; i<n; i++)
        {
            disti += (a[i] != b[i]);
        }
        return disti;
    }

    int hamming_distance_fix_omp(int* a, int* b, int n)
    {
        int disti = 0;
#pragma omp parallel for reduction(+:disti)
        for(int i=0; i<n; i++)
        {
            disti += (a[i] != b[i]);
        }
        return disti;
    }




};


class two_indices
{
public:
    int one;
    int two;
};



