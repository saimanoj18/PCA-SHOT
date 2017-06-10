#include<pca_shot/pcashot_bits.h>

#define MAXBUFSIZE  ((int) 1e6)

MatrixXf readMatrix(const char *filename)
{
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
    {
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];

        if (temp_cols == 0)
            continue;

        if (cols == 0)
            cols = temp_cols;

        rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    MatrixXf result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
}



int main()
{
    cbshot cb;
    // Read a PCD file from disk.

    pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_files/scene005_0.pcd", cb.cloud2);
    //pcl::io::loadPolygonFilePLY("../sample_files/scene005_0.ply", cb.mesh2);

    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_files/PeterRabbit001_0.pcd", cb.cloud1);
    //pcl::io::loadPolygonFilePLY("../sample_files/PeterRabbit001_0.ply", cb.mesh1);

    //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_files/Doll018_0.pcd", cb.cloud1);
    //pcl::io::loadPolygonFilePLY("../sample_files/Doll018_0.ply", cb.mesh1);

    pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_files/mario000_0.pcd", cb.cloud1);
    //pcl::io::loadPolygonFilePLY("../sample_files/mario000_0.ply", cb.mesh1);


    //pcl::io::loadPCDFile<pcl::PointXYZ>("/home/sai/Downloads/cloud0077.pcd", cb.cloud2);


    cb.calculate_normals (0.02);

    cb.calculate_voxel_grid_keypoints (0.02);//0.01

    cb.get_keypoint_indices();

    cb.calculate_SHOT(0.05);//0.10


/*
    ofstream shot_distribution;
    shot_distribution.open ("/home/sai/pcl_testing/pca_shot/PCA_matrices/shot_distribution_cloud77_0.02_10.txt", ios::out | ios::app);

    for (int i = 0; i < cb.cloud2_shot.size(); i++)
    {
        for (int j = 0; j < 352; j++)
        {
            shot_distribution << cb.cloud2_shot[i].descriptor[j] << "  ";
        }
        shot_distribution << "\n";
    }

*/


    /**************************************************/

    pcl::Correspondences corrs;

    Eigen::MatrixXf dist;
    dist.resize(cb.cloud1_shot.size(), cb.cloud2_shot.size());
    dist.setZero();

    Eigen::MatrixXf T;
    T = readMatrix("/home/sai/pcl_testing/pca_shot/PCA_matrices/PCA_for_SHOT_cloud77.txt");
    //T = readMatrix("/home/sai/pcl_testing/pca_shot/PCA_matrices/PCA_for_SHOT_80_clouds.txt");

    //T = readMatrix("/home/sai/pcl_testing/pca_shot/PCA_matrices/SparsePCA_for_SHOT_cloud77.txt");


    clock_t start_, end_;
    double cpu_time_used_;
    start_ = clock();

    for (int i = 0; i < cb.cloud1_shot.size(); i++)
    {
        Eigen::VectorXf Y,X; Y.resize(352); X.resize(352);
        for(int id1 = 0; id1 < 352; id1++)
            X(id1) = cb.cloud1_shot[i].descriptor[id1];

        Y = T * X;

        for (int j = 0; j < 352; j++)
            cb.cloud1_shot[i].descriptor[j] = Y(j);

    }

    for (int i = 0; i < cb.cloud2_shot.size(); i++)
    {
        Eigen::VectorXf Y,X; Y.resize(352); X.resize(352);
        for(int id1 = 0; id1 < 352; id1++)
            X(id1) = cb.cloud2_shot[i].descriptor[id1];

        Y = T * X;

        for (int j = 0; j < 352; j++)
            cb.cloud2_shot[i].descriptor[j] = Y(j);

    }

    end_ = clock();
    cpu_time_used_ = ((double) (end_ - start_)) / CLOCKS_PER_SEC;
    std::cout << "Time taken for creating PCA-SHOT descriptors : " << (double)cpu_time_used_ << std::endl;



    for (int i = 0; i < cb.cloud1_shot.size(); i++)
    {
        for (int j = 0; j < cb.cloud2_shot.size(); j++)
        {
            //dist(i,j) = (sh1.signature_descriptors[i] - sh2.signature_descriptors[j]).norm();

            Eigen::VectorXf one, two;
            one.resize(30);
            two.resize(30);

            for(int k = 0; k < 30; k++)
            {
                one[k] = cb.cloud1_shot[i].descriptor[k];
            }

            for(int k = 0; k < 30; k++)
            {
                two[k] = cb.cloud2_shot[j].descriptor[k];
            }

            dist(i,j) = (one-two).norm();


        }
    }



    for (int i = 0; i < dist.rows(); i++)
    {
        float small = dist(i,0);
        int index_i = i;
        int index_j = 0;
        for(int j = 0; j < dist.cols(); j++)
        {
            if (dist(i,j) < small)
            {
                small = dist(i,j);
                index_i = i;
                index_j = j;
            }
        }
        int temp = 0;
        for (int j = 0; j < dist.rows(); j++)
        {
            if (dist(j,index_j) < small)
                temp = 1;
        }
        if (temp == 0)
        {
            pcl::Correspondence corr;
            //cout << "here " << endl;
            //cout << cb.cloud1_keypoints_indices[index_i] << endl;
            corr.index_query = cb.cloud1_keypoints_indices[index_i];// vulnerable
            corr.index_match = cb.cloud2_keypoints_indices[index_j];// vulnerable
            corr.distance = dist(index_i,index_j);

            corrs.push_back(corr);
        }

    }


    /************************************************************/



    cout << "No. of Reciprocal Correspondences : " << corrs.size() << endl;



    pcl::CorrespondencesConstPtr corrs_const_ptr = boost::make_shared< pcl::Correspondences >(corrs);

    pcl::Correspondences corr_shot;
    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot;
    Ransac_based_Rejection_shot.setInputSource(cb.cloud1.makeShared());
    Ransac_based_Rejection_shot.setInputTarget(cb.cloud2.makeShared());
    Ransac_based_Rejection_shot.setInlierThreshold(0.02);
    Ransac_based_Rejection_shot.setInputCorrespondences(corrs_const_ptr);
    Ransac_based_Rejection_shot.getCorrespondences(corr_shot);

    cout << "Mat : \n" << Ransac_based_Rejection_shot.getBestTransformation()<< endl;

    cout << "True correspondences after RANSAC : " << corr_shot.size() << endl;


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (255, 255, 255);



    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(cb.cloud1.makeShared(), 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cb.cloud1.makeShared(), single_color1, "sample cloud1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud1");
    //viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    Eigen::Matrix4f t;
    t<<1,0,0,0.6,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1;

    //cloudNext is my target cloud
    pcl::transformPointCloud(cb.cloud2,cb.cloud2,t);

    //int v2(1);
    //viewer->createViewPort (0.5,0.0,0.1,1.0,1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cb.cloud2.makeShared(), 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ> (cb.cloud2.makeShared(), single_color2, "sample cloud2");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "sample cloud2");



    viewer->addCorrespondences<pcl::PointXYZ>(cb.cloud1.makeShared(), cb.cloud2.makeShared(), /*corrs*/ corr_shot, "correspondences"/*,v1*/);



    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }




    return 0;
}

