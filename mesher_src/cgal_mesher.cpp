#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_constant_domain_field_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>

// g++ -O3 -std=c++17 -DCGAL_DISABLE_GMP=1 -DCMAKE_OVERRIDDEN_DEFAULT_ENT_BACKEND=3 -I/home/jiaming/Documents/CGAL-6.0.1/include -I/home/jiaming/Documents/boost_1_76_0 cgal_mesher.cpp -o cgalmesher
// cl.exe /O2 /std:c++17 /I"C:\Users\hamid\OneDrive - University of Birmingham\Documents\Jiaming\boost_1_76_0" /I"C:\Users\hamid\OneDrive - University of Birmingham\Documents\CGAL-6.0.1\include" /D CGAL_DISABLE_GMP=1 /D CMAKE_OVERRIDDEN_DEFAULT_ENT_BACKEND=3 /EHsc cgal_mesher.cpp /Fecgalmesher.exe

// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Labeled_mesh_domain_3<K> Mesh_domain;
typedef CGAL::Sequential_tag Concurrency_tag;

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
typedef CGAL::Mesh_constant_domain_field_3<Mesh_domain::R, Mesh_domain::Index> Sizing_field;

using namespace std;
using namespace CGAL::parameters;

int main(int argc, char *argv[]){
	if(argc!=3 && argc!=4){
		cerr << "Input should be: inrfile.inr outmesh.mesh [criteria.txt]" << endl;
		return 1;
	}
	string inrfile, outfile, criteriafile;
	inrfile = argv[1];
	outfile = argv[2];

	CGAL::Image_3 image;
    if(!image.read(inrfile.c_str())){
      cerr << "Error: Cannot read file " <<  inrfile << endl;
      return 1;
    }
    // Domain
    Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(image);

    double facet_angle_=25, facet_size_=2, facet_distance_=1.5, cell_radius_edge_ratio_=2, general_cell_size_=2;
    bool smooth = true;
    int n_subdomain = 0;
    vector<int> labels;
    vector<double> cell_sizes;

    // Read the criteria file
    if(argc==4){
      criteriafile = argv[3];
    	ifstream cfs(criteriafile.c_str());
  		if (!cfs) {
  			cerr << " Can not read mesh criteria file!" << endl;
  			return 1;
  		}
  		cfs >> facet_angle_;
  		cfs >> facet_size_;
  		cfs >> facet_distance_;
  		cfs >> cell_radius_edge_ratio_;
  		cfs >> general_cell_size_;
  		cfs >> smooth;
      cfs >> n_subdomain;
      if(n_subdomain>0){
        labels.resize(n_subdomain);
        cell_sizes.resize(n_subdomain);
        for(int i=0; i<n_subdomain; i++){
          cfs >> labels[i];
          cfs >> cell_sizes[i];
        }
        // for(int i=0; i<n_subdomain; i++){
        //   cout << labels[i] << ", " << cell_sizes[i]<<endl;
        // }
      }
    }
    // cout << facet_angle_ << endl<< facet_size_ << endl<< facet_distance_ << endl<< cell_radius_edge_ratio_ << endl<< general_cell_size_ << endl<< smooth << endl;
    
	  // Sizing field: default and special subdomain 
    Sizing_field size(general_cell_size_);
    if(n_subdomain>0){
      for(int i=0; i<n_subdomain; i++){
        size.set_size(cell_sizes[i], 3, domain.index_from_subdomain_index(labels[i]));
      }
    }

    // Mesh criteria
    Mesh_criteria criteria(facet_angle(facet_angle_).facet_size(facet_size_).facet_distance(facet_distance_).cell_radius_edge_ratio(cell_radius_edge_ratio_).cell_size(size));

    // Meshing
    C3t3 c3t3;
    cout << "Meshing..." << endl;
    try{
      if(smooth){
        c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(),no_exude());
        cout << "Running Lloyd smoothing... (up to 120s)" << endl;
        CGAL::lloyd_optimize_mesh_3(c3t3, domain, time_limit=120);
        cout << "Running local optimization..." << endl;
        CGAL::perturb_mesh_3(c3t3, domain);
        CGAL::exude_mesh_3(c3t3, sliver_bound=10, time_limit=60);
      }
      else{
        c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());
        cout << "Running local optimization..." << endl;
        CGAL::perturb_mesh_3(c3t3, domain);
        CGAL::exude_mesh_3(c3t3, sliver_bound=10, time_limit=60);
      }
    }
    catch(...){
      cerr << "CGAL mesher failed." << endl;
      return 1;
    }

    // Output
    ofstream medit_file(outfile.c_str());
    c3t3.output_to_medit(medit_file);
    medit_file.close();

    return 0;
}
