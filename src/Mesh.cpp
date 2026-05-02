#include <algorithm>
#include <assert.h>
#include <chrono>
#include <gmsh.h>
#include <iostream>
#include <omp.h>
#include <string>
#include <unordered_map>

#include "Mesh.h"
#include "Parallel.h"
#include "configParser.h"
#include "utils.h"

#ifdef DG_USE_MPI
// Forward declarations: implementations live at the bottom of this file,
// next to buildPartitionLayout() (the partition-aware helpers section).
static void loadElementsByPartition(const PartitionLayout &layout,
                                    int elementType,
                                    const std::vector<double> &integrationParamCoords,
                                    std::vector<std::size_t> &elTags,
                                    std::vector<std::size_t> &elNodeTags,
                                    std::vector<double> &jacobians,
                                    std::vector<double> &jacobianDets,
                                    std::vector<double> &intPtCoords,
                                    std::vector<int> &haloOwnerRank,
                                    int &nLocal,
                                    int &nHalo);

// Concatenated barycenters (owned then halo) — same ordering as elTags.
static void loadBarycentersByPartition(const PartitionLayout &layout,
                                       int elementType,
                                       std::vector<double> &barycenters);

// Concatenated element face/edge node tags (owned then halo).
// faceNumNodes = 4 for hex faces, 3 for tet faces, 2 for line edges.
// Pass -1 in faceNumNodes to use getElementEdgeNodes (1D edges).
static void loadFaceNodesByPartition(const PartitionLayout &layout,
                                     int elementType,
                                     int faceNumNodes,
                                     std::vector<std::size_t> &fNodeTags);
#endif

/**
 * Mesh constructor: load the mesh data and parameters thanks to
 * Gmsh api. Create the elements mapping and set the boundary conditions.
 *
 * @name string File name
 * @config config Configuration object (content of the config parsed and load in memory)
 */
Mesh::Mesh(Config config) : config(config)
{
    // ----------------------------------------------------------------
    // Phase 7 profiler — cumulative timing of major preprocessing blocks.
    // Set DG_PROFILE_MESH=1 to enable; output is one CSV line per rank
    // on stdout at end of constructor (ingestible by pandas/awk).
    // ----------------------------------------------------------------
    using prof_clk = std::chrono::steady_clock;
    const bool profileMesh = (std::getenv("DG_PROFILE_MESH") != nullptr);
    auto profTotalStart = prof_clk::now();
    std::map<std::string, long long> profUs;
    auto profMark = [&](const std::string &label, prof_clk::time_point t0) {
        if (!profileMesh) return;
        auto t1 = prof_clk::now();
        profUs[label] += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    };

    /******************************
     *          Elements          *
     ******************************/

    // meshFileName = config.meshFileName;

    screen_display::write_string("Load data", GREEN);

    auto pp = [](const std::string &label, const std::vector<double> &v, int mult)
    {
        std::cout << " * " << v.size() / mult << " " << label << ": ";
        for (auto c : v)
            std::cout << c << " ";
        std::cout << "\n";
    };

    auto start = std::chrono::system_clock::now();
    auto pt = prof_clk::now();
    m_elDim = gmsh::model::getDimension();
    gmsh::model::mesh::getElementTypes(m_elType, m_elDim);
    int _numPrimaryNodes = 0;

    gmsh::model::mesh::getElementProperties(m_elType[0], m_elName, m_elDim,
                                            m_elOrder, m_elNumNodes, m_elParamCoord, _numPrimaryNodes);

    m_elIntType = "Gauss" + std::to_string(2 * m_elOrder);

    // Integration points (mesh-independent; depends only on element type)
    gmsh::model::mesh::getIntegrationPoints(m_elType[0], m_elIntType, m_elParamCoord, m_elWeight);
    profMark("01_el_props_intpts", pt);

    screen_display::write_string("Elements - Compute Jacobian", GREEN);

    pt = prof_clk::now();
    int _numOrientations;
    int _numComponents;
    gmsh::model::mesh::getBasisFunctions(m_elType[0], m_elParamCoord, config.elementType,
                                         _numComponents, m_elBasisFcts, _numOrientations);
    gmsh::model::mesh::getBasisFunctions(m_elType[0], m_elParamCoord, "Grad" + config.elementType,
                                         _numComponents, m_elUGradBasisFcts, _numOrientations);
    profMark("02_el_basis_fcts", pt);

#ifdef DG_USE_MPI
    bool usePartition = (Parallel::size() > 1);
    std::vector<int> haloOwnerRank;
    int nLocal = 0, nHalo = 0;
    if (usePartition)
    {
        pt = prof_clk::now();
        // Partitioned load: each rank only allocates [owned | halo].
        PartitionLayout layout = buildPartitionLayout();
        profMark("03a_buildPartitionLayout", pt);
        pt = prof_clk::now();
        loadElementsByPartition(layout, m_elType[0], m_elParamCoord,
                                m_elTags, m_elNodeTags,
                                m_elJacobians, m_elJacobianDets, m_elIntPtCoords,
                                haloOwnerRank, nLocal, nHalo);
        profMark("03b_loadElementsByPartition", pt);
        m_elNum = nLocal + nHalo;

        // Member partition state (replaces the old global m_elOwnerRank/range)
        m_localElStart = 0;
        m_localElEnd   = nLocal;
        const int myRank = Parallel::rank();
        m_elOwnerRank.assign(m_elNum, myRank);
        for (int i = 0; i < nHalo; ++i)
            m_elOwnerRank[nLocal + i] = haloOwnerRank[i];

        std::cout << "[MPI rank " << myRank << "] m_elNum=" << m_elNum
                  << " (" << nLocal << " local + " << nHalo << " halo)\n"
                  << std::flush;
    }
    else
#endif
    {
        pt = prof_clk::now();
        // Non-MPI / single-rank path: full global mesh on this process.
        gmsh::model::mesh::getElementsByType(m_elType[0], m_elTags, m_elNodeTags);
        m_elNum = (int)m_elTags.size();
        profMark("03c_getElementsByType_seq", pt);
        pt = prof_clk::now();
        gmsh::model::mesh::getJacobians(m_elType[0], m_elParamCoord, m_elJacobians,
                                        m_elJacobianDets, m_elIntPtCoords);
        profMark("03d_getJacobians_seq", pt);
    }

    // std::ofstream _outfile_("m_elJacobians.txt");
    // _outfile_ << "size=" << m_elJacobians.size() << std::endl;
    // for (size_t i = 0; i < m_elJacobians.size(); i++)
    //     _outfile_ << m_elJacobians[i] << std::endl;
    // _outfile_.close();

    // pp("Jacobian determinants at integration points", m_elJacobianDets, 1);

    m_elNumIntPts = (int)m_elJacobianDets.size() / m_elNum;

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);
    /**
     * Gmsh provides the derivative of the shape functions along
     * the parametric directions. We therefore compute their derivative
     * along the physical directions thanks to composed derivative.
     * The system can be expressed as J^T * df/dx = df/du
     *
     * |dx/du dx/dv dx/dw|^T  |df/dx|   |df/du|
     * |dy/du dy/dv dy/dw|  * |df/dy| = |df/dv|
     * |dz/du dz/dv dz/dw|    |df/dz|   |df/dw|
     *
     * (x,y,z) are the physical coordinates
     * (u,v,w) are the parametric coordinates
     *
     * NB: Instead of transposing, we take advantages of the fact
     * Lapack/Blas use column major while Gmsh provides row major.
     */

    start = std::chrono::system_clock::now();
    pt = prof_clk::now();
    std::vector<double> jacobian(m_elDim * m_elDim);
    m_elGradBasisFcts.resize(m_elNum * m_elNumNodes * m_elNumIntPts * 3);

    // #pragma omp parallel for
    for (size_t el = 0; el < m_elNum; ++el)
    {
        for (int g = 0; g < m_elNumIntPts; ++g)
        {
            for (int f = 0; f < m_elNumNodes; ++f)
            {
                // The copy operations are not required. They're simply enforced
                // to ensure that the inputs (jacobian, grad) remains unchanged.
                for (int i = 0; i < m_elDim; ++i)
                {
                    for (int j = 0; j < m_elDim; ++j)
                    {
                        // screen_display::write_value("elJacobian(el, g, i, j)",elJacobian(el, g, i, j),"");
                        jacobian[i * m_elDim + j] = elJacobian(el, g, i, j);
                    }
                }

                // std::cout<<"el = "<<el<<" - g = "<<g<<" - f = "<<f<<std::endl;
                // screen_display::write_value("elUGradBasisFct(g, f)",elUGradBasisFct(g, f),"",BLUE);
                // screen_display::write_value("elGradBasisFct(el, g, f)",elGradBasisFct(el, g, f),"",BLUE);
                std::copy(&elUGradBasisFct(g, f), &elUGradBasisFct(g, f) + m_elDim, &elGradBasisFct(el, g, f));
                eigen::solve(jacobian.data(), &elGradBasisFct(el, g, f), m_elDim);
                // screen_display::write_string("flag 1", RED);
            }
        }
    }

    profMark("04_el_phys_grad_basis", pt);

    // pp("Element jacobian", jacobian, 1);

    assert(m_elType.size() == 1);
    assert(m_elNodeTags.size() == m_elNum * m_elNumNodes);
    assert(m_elJacobianDets.size() == m_elNum * m_elNumIntPts);
    assert(m_elBasisFcts.size() == m_elNumNodes * m_elNumIntPts);
    assert(m_elGradBasisFcts.size() == m_elNum * m_elNumIntPts * m_elNumNodes * 3);

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    gmsh::logger::write("==================================================");
    gmsh::logger::write("Number of Elements : " + std::to_string(m_elNum));
    gmsh::logger::write("Element dimension : " + std::to_string(m_elDim));
    gmsh::logger::write("Element Type : " + m_elName);
    gmsh::logger::write("Element Order : " + std::to_string(m_elOrder));
    gmsh::logger::write("Element Nbr Nodes : " + std::to_string(m_elNumNodes));
    gmsh::logger::write("Integration type : " + m_elIntType);
    gmsh::logger::write("Integration Nbr points : " + std::to_string(m_elNumIntPts));

    /******************************
     *            Faces           *
     ******************************/
    screen_display::write_string("Faces treatment", GREEN);
    start = std::chrono::system_clock::now();
    m_fDim = m_elDim - 1;
    const bool hasQuadrilateralFaces = m_elName.find("Hexahedron") != std::string::npos;
    m_fName = m_fDim == 0 ? "point" : m_fDim == 1 ? "line"
                                  : m_fDim == 2   ? (hasQuadrilateralFaces ? "quadrangle" : "triangle")
                                                  : "None";
    m_fNumNodes = m_fDim == 0 ? 1 : m_fDim == 1 ? 1 + m_elOrder
                                : m_fDim == 2   ? (hasQuadrilateralFaces ? (m_elOrder + 1) * (m_elOrder + 1)
                                                                           : (m_elOrder + 1) * (m_elOrder + 2) / 2)
                                                : 0;

    m_fType = gmsh::model::mesh::getElementType(m_fName, m_elOrder);

    /**
     * [1] Get Faces for all elements (local + halo in MPI mode)
     */
    pt = prof_clk::now();
#ifdef DG_USE_MPI
    if (Parallel::size() > 1)
    {
        // Re-classify: we need the layout again to know which entities to walk.
        // (Cheap: it just queries gmsh, no re-partitioning.)
        PartitionLayout layout = buildPartitionLayout();
        const int faceNumNodes = (m_fDim < 2) ? -1 : (hasQuadrilateralFaces ? 4 : 3);
        loadFaceNodesByPartition(layout, m_elType[0], faceNumNodes, m_elFNodeTags);
    }
    else
#endif
    {
        if (m_fDim < 2)
            gmsh::model::mesh::getElementEdgeNodes(m_elType[0], m_elFNodeTags, -1);
        else
            gmsh::model::mesh::getElementFaceNodes(m_elType[0], hasQuadrilateralFaces ? 4 : 3, m_elFNodeTags, -1);
    }
    profMark("05_get_face_nodes", pt);

    m_fNumPerEl = m_elFNodeTags.size() / (m_elNum * m_fNumNodes);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);
    /**
     * [2] Remove the faces counted two times
     *     i.e. common face between two elements.
     */
    screen_display::write_string("Remove the faces counted two times", GREEN);
    start = std::chrono::system_clock::now();

    pt = prof_clk::now();
    //! ////////////////////////////////
    getUniqueFaceNodeTags();
    //! ////////////////////////////////
    profMark("06_uniqueFaceNodeTags", pt);

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    /**
     * [3] Finally, we create a single entity containing all the
     *     unique faces. We call Gmsh with empty face tags and
     *     retrieve directly after the auto-generated tags.
     */
    screen_display::write_string("Create a single entity");
    start = std::chrono::system_clock::now();
    pt = prof_clk::now();
    m_fEntity = gmsh::model::addDiscreteEntity(m_fDim);

    gmsh::model::mesh::addElementsByType(m_fEntity, m_fType, {}, m_fNodeTags);

    m_fIntType = m_elIntType;

    gmsh::model::mesh::getIntegrationPoints(m_fType, m_fIntType, m_fIntParamCoords, m_fWeight);

    m_fNum = m_fNodeTags.size() / m_fNumNodes;
    profMark("07_face_entity_creation", pt);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    /**
     * A priori the same integration type and order is applied
     * to the surface and to the volume integrals.
     */
    screen_display::write_string("Faces - Compute Jacobian");
    start = std::chrono::system_clock::now();
    pt = prof_clk::now();
    m_fIntType = m_elIntType;

    gmsh::model::mesh::getBasisFunctions(m_fType, m_fIntParamCoords, config.elementType, *new int, m_fBasisFcts, _numOrientations);

    gmsh::model::mesh::getBasisFunctions(m_fType, m_fIntParamCoords, "Grad" + config.elementType, *new int, m_fUGradBasisFcts, _numOrientations);

    gmsh::model::mesh::getJacobians(m_fType, m_fIntParamCoords, m_fJacobians, m_fJacobianDets, m_fIntPtCoords, m_fEntity);

    m_fNumIntPts = (int)m_fJacobianDets.size() / m_fNum;
    profMark("08_face_basis_jacobians", pt);

    /**
     * See element part for explanation. (line 40)
     */
    pt = prof_clk::now();
    m_fGradBasisFcts.resize(m_fNum * m_fNumNodes * m_fNumIntPts * 3);
    // #pragma omp parallel for
    for (int f = 0; f < m_fNum; ++f)
    {
        for (int g = 0; g < m_fNumIntPts; ++g)
        {
            for (int n = 0; n < m_fNumNodes; ++n)
            {
                for (int i = 0; i < m_elDim; ++i)
                {
                    for (int j = 0; j < m_elDim; ++j)
                    {
                        jacobian[i * m_elDim + j] = fJacobian(f, g, i, j);
                    }
                }
                std::copy(/*std::execution::par,*/ &fUGradBasisFct(g, n), &fUGradBasisFct(g, n) + m_elDim, &fGradBasisFct(f, g, n));
                eigen::solve(jacobian.data(), &fGradBasisFct(f, g, n), m_elDim);
            }
        }
    }
    profMark("09_face_phys_grad_basis", pt);

    // pp("m_fJacobians", m_fJacobians, 3);

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);
    /**
     * Define a normal associated to each surface.
     */

    screen_display::write_string("Define a normal/tangent/bitangent (1D and 2D only, 3D in progress) associated to each surface.", GREEN);
    start = std::chrono::system_clock::now();
    pt = prof_clk::now();
    std::vector<double> normal(m_Dim);
    std::vector<double> tangent(m_Dim);
    std::vector<double> bitangent(m_Dim);
    // #pragma omp parallel for
    for (int f = 0; f < m_fNum; ++f)
    {
        // compute here tangent and bitangent for 3D cases

        std::vector<double> T = {0, 0, 0};
        std::vector<double> B = {0, 0, 0};

        if (m_fDim == 2)
        {
            // Build T and B from the first two edges of the face (Gram-Schmidt).
            // Using node positions is robust for any integration rule (avoids
            // collinear integration-point triplets that caused delta=0 / NaN).
            std::vector<double> n0coord, n1coord, n2coord, _param;
            int _dim, _tag;
            gmsh::model::mesh::getNode(fNodeTag(f, 0), n0coord, _param, _dim, _tag);
            gmsh::model::mesh::getNode(fNodeTag(f, 1), n1coord, _param, _dim, _tag);
            gmsh::model::mesh::getNode(fNodeTag(f, 2), n2coord, _param, _dim, _tag);

            // First edge: T = normalize(n1 - n0)
            T[0] = n1coord[0] - n0coord[0];
            T[1] = n1coord[1] - n0coord[1];
            T[2] = n1coord[2] - n0coord[2];
            int dim3 = 3;
            eigen::normalize(T.data(), dim3);

            // Second edge projected out of T: B = normalize((n2-n0) - ((n2-n0)·T) T)
            double e2x = n2coord[0] - n0coord[0];
            double e2y = n2coord[1] - n0coord[1];
            double e2z = n2coord[2] - n0coord[2];
            double proj = e2x * T[0] + e2y * T[1] + e2z * T[2];
            B[0] = e2x - proj * T[0];
            B[1] = e2y - proj * T[1];
            B[2] = e2z - proj * T[2];
            eigen::normalize(B.data(), dim3);
        }

        // screen_display::write_value("delta", delta);
        // screen_display::write_value("|T|", sqrt(pow(T[0], 2) + pow(T[1], 2) + pow(T[2], 2)));
        // screen_display::write_value("|B|", sqrt(pow(B[0], 2) + pow(B[1], 2) + pow(B[2], 2)));
        // getchar();

        //! ////////////////////////////////////////////////////
        for (int g = 0; g < m_fNumIntPts; ++g)
        {

            switch (m_fDim)
            {
            case 0:
            {
                normal = {1, 0, 0};
                tangent = {0, 0, 0};
                bitangent = {0, 0, 0};
                break;
            }
            case 1:
            {
                std::vector<double> normalPlane = {0, 0, -1};
                eigen::cross(&fGradBasisFct(f, g, 0), normalPlane.data(), normal.data());
                if (eigen::dot(&fGradBasisFct(f, g), &fGradBasisFct(f, 0), m_Dim) < 0)
                {
                    for (int x = 0; x < m_Dim; ++x)
                        // #pragma omp atomic
                        normal[x] *= -1.0;
                }
                tangent = {-normal[1], normal[0], 0};
                bitangent = {0, 0, 0};
                break;
            }
            case 2:
            {
                eigen::cross(&fGradBasisFct(f, g, 0), &fGradBasisFct(f, g, 1), normal.data());
                if (g != 0 && eigen::dot(&fNormal(f, 0), normal.data(), m_Dim) < 0)
                {
                    for (int x = 0; x < m_Dim; ++x)
                        // #pragma omp atomic
                        normal[x] *= -1.0;
                    // normal[x] = -normal[x];
                }
                tangent = {T[0], T[1], T[2]};
                bitangent = {B[0], B[1], B[2]};
                break;
            }
            }
            eigen::normalize(normal.data(), m_Dim);
            eigen::normalize(tangent.data(), m_Dim);
            eigen::normalize(bitangent.data(), m_Dim);
            m_fNormals.insert(m_fNormals.end(), normal.begin(), normal.end());
            m_fTangents.insert(m_fTangents.end(), tangent.begin(), tangent.end());
            m_fBiTangents.insert(m_fBiTangents.end(), bitangent.begin(), bitangent.end());
        }
    }

    if (m_elDim == 3 && m_elOrder != 1)
        fc = -1;

    profMark("10_face_normals_tangents", pt);

    screen_display::write_if_false(m_elFNodeTags.size() == m_elNum * m_fNumPerEl * m_fNumNodes, "m_elFNodeTags size error");
    screen_display::write_if_false(m_fJacobianDets.size() == m_fNum * m_fNumIntPts, "m_fJacobianDets size error");
    screen_display::write_if_false(m_fBasisFcts.size() == m_fNumNodes * m_fNumIntPts, "m_fBasisFcts size error");
    screen_display::write_if_false(m_fNormals.size() == m_Dim * m_fNum * m_fNumIntPts, "m_fNormals size error");
    screen_display::write_if_false(m_fTangents.size() == m_Dim * m_fNum * m_fNumIntPts, "m_fTangents size error");
    screen_display::write_if_false(m_fBiTangents.size() == m_Dim * m_fNum * m_fNumIntPts, "m_fBiTangents size error");

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    gmsh::logger::write("==================================================");
    gmsh::logger::write("Number of Faces : " + std::to_string(m_fNum));
    gmsh::logger::write("Faces per Element : " + std::to_string(m_fNumPerEl));
    gmsh::logger::write("Face dimension : " + std::to_string(m_fDim));
    gmsh::logger::write("Face Type : " + m_fName);
    gmsh::logger::write("Face Nbr Nodes : " + std::to_string(m_fNumNodes));
    gmsh::logger::write("Integration type : " + m_fIntType);
    gmsh::logger::write("Integration Nbr points : " + std::to_string(m_fNumIntPts));

    /******************************
     *       Connectivity         *
     ******************************/

    /**
     * Assign corresponding faces to each element, we use the
     * fact that the node tags per face has already been ordered
     */
    screen_display::write_string("Connectivity: Assign corresponding faces to each element", GREEN);
    start = std::chrono::system_clock::now();

    pt = prof_clk::now();
    //! ////////////////////////////////
    getConnectivityFaceToElement();
    //! ////////////////////////////////
    profMark("11_getConnectivityFaceToElement", pt);

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    /**
     * For efficiency purposes we also directly store the mapping
     * between face node id and element node id. For example, the
     * 3rd node of the face correspond to the 7th of the element.
     */
    screen_display::write_string("Store the mapping between face node id and element node id", GREEN);
    start = std::chrono::system_clock::now();

    pt = prof_clk::now();
    m_fNToElNIds.resize(m_fNum);
    // #pragma omp parallel for
    for (int f = 0; f < m_fNum; ++f)
    {
        for (int nf = 0; nf < m_fNumNodes; ++nf)
        {
            for (size_t el : m_fNbrElIds[f])
            {
                for (int nel = 0; nel < m_elNumNodes; ++nel)
                {
                    if (fNodeTag(f, nf) == elNodeTag(el, nel))
                        m_fNToElNIds[f].push_back(nel);
                }
            }
        }
    }
    profMark("12_face_to_el_node_map", pt);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);
    /**
     * Up to now, the normals are associated to the faces.
     * We still need to know how the normal is oriented
     * with respect to its neighbouring elements.
     *
     * For instance, element1 normal has the same orientation, we therefore assign
     * the orientation +1 and reciprocally we set the orientation to -1 for e2.
     *
     *  ____     f1       ____
     * |    |     |      |    |
     * | e1 |->   |->  <-| e2 |
     * |____|     |      |____|
     *
     */
    screen_display::write_string("Define normals orientation", GREEN);
    start = std::chrono::system_clock::now();

    pt = prof_clk::now();
    double dotProduct;
    std::vector<double> m_elBarycenters, fNodeCoord(3), elOuterDir(3), paramCoords;
#ifdef DG_USE_MPI
    if (Parallel::size() > 1)
    {
        PartitionLayout layout = buildPartitionLayout();
        loadBarycentersByPartition(layout, m_elType[0], m_elBarycenters);
    }
    else
#endif
    {
        gmsh::model::mesh::getBarycenters(m_elType[0], -1, false, true, m_elBarycenters);
    }

    m_elFOrientation.clear();

    for (size_t el = 0; el < m_elNum; ++el)
    {
        for (int f = 0; f < m_fNumPerEl; ++f)
        {
            dotProduct = 0.0;

            int _dim, _tag;

            gmsh::model::mesh::getNode(elFNodeTag(el, f), fNodeCoord, paramCoords, _dim, _tag);

            for (int x = 0; x < m_Dim; x++)
            {
                elOuterDir[x] = fNodeCoord[x] - m_elBarycenters[el * 3 + x];
                dotProduct += elOuterDir[x] * fNormal(elFId(el, f), 0, x);
            }

            size_t value = (dotProduct >= 0) ? 1 : -1;
            m_elFOrientation.push_back(value);
        }
    }

    profMark("13_el_face_orientation", pt);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    /**
     * Once the orientation known, we reclassify the neighbouring
     * elements by imposing the first one to be oriented in the
     * same direction than the corresponding face.
     */
    screen_display::write_string("Reclassification of the neighbouring elements", GREEN);
    start = std::chrono::system_clock::now();

    pt = prof_clk::now();
    size_t elf;
    // #pragma omp parallel for
    for (int f = 0; f < m_fNum; ++f)
    {
        for (int lf = 0; lf < m_fNumPerEl; ++lf)
        {
            if (elFId(fNbrElId(f, 0), lf) == f)
                elf = lf;
        }
        if (m_fNbrElIds.size() == 2)
        {
            if (elFOrientation(fNbrElId(f, 0), elf) <= 0)
            {
                std::swap(m_fNbrElIds[f][0], m_fNbrElIds[f][1]);
                for (int nf = 0; nf < m_fNumNodes; ++nf)
                    std::swap(fNToElNId(f, nf, 0), fNToElNId(f, nf, 1));
            }
        }
    }

    assert(m_elFIds.size() == m_elNum * m_fNumPerEl);
    assert(m_elFOrientation.size() == m_elNum * m_fNumPerEl);

    profMark("14_nbr_reclassification", pt);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    gmsh::logger::write("==================================================");
    gmsh::logger::write("Element-Face connectivity retrieved.");
    start = std::chrono::system_clock::now();
    pt = prof_clk::now();
    //---------------------------------------------------------------------
    // Boundary conditions
    //---------------------------------------------------------------------

    /******************************
     *     Boundary conditions    *
     ******************************/

    /**
     * Check if a face is a boundary or not and orientate
     * the normal at boundaries in the outward direction.
     * This convention is particularly useful for BCs.
     */
    screen_display::write_string("Boundary conditions", GREEN);
    // #pragma omp parallel for
    for (int f = 0; f < m_fNum; ++f)
    {
        if (m_fNbrElIds[f].size() < 2)
        {
            m_fIsBoundary.push_back(true);
            for (int lf = 0; lf < m_fNumPerEl; ++lf)
            {
                if (elFId(fNbrElId(f, 0), lf) == f)
                {
                    for (int g = 0; g < m_fNumIntPts; ++g)
                    {
                        // #pragma omp atomic
                        fNormal(f, g, 0) *= elFOrientation(fNbrElId(f, 0), lf);
                        // fTangent(f, g, 0) *= elFOrientation(fNbrElId(f, 0), lf);
                        // fBiTangent(f, g, 0) *= elFOrientation(fNbrElId(f, 0), lf);
                        // #pragma omp atomic
                        fNormal(f, g, 1) *= elFOrientation(fNbrElId(f, 0), lf);
                        // fTangent(f, g, 1) *= elFOrientation(fNbrElId(f, 0), lf);
                        // fBiTangent(f, g, 1) *= elFOrientation(fNbrElId(f, 0), lf);
                        // #pragma omp atomic
                        fNormal(f, g, 2) *= elFOrientation(fNbrElId(f, 0), lf);
                        // fTangent(f, g, 2) *= elFOrientation(fNbrElId(f, 0), lf);
                        // fBiTangent(f, g, 2) *= elFOrientation(fNbrElId(f, 0), lf);
                    }
                    elFOrientation(fNbrElId(f, 0), lf) = 1;
                }
            }
        }
        else
        {
            m_fIsBoundary.push_back(false);
        }
    }

    /**
     * Iterate over the physical boundaries and over each nodes
     * belonging to that boundary. Retrieve the associated face and assign
     * it an unique integer representing the BC type.
     *
     * 1        : Reflecting
     * 2        : Absorbing
     * Default  : Absorbing (!= 1 or 2)
     */
    m_fBC.resize(m_fNum);
    std::vector<size_t> nodeTags;
    std::vector<double> coord;
    for (auto const &physBC : config.physBCs)
    {
        auto physTag = physBC.first;
        auto BCtype = physBC.second.first;
        auto BCvalue = physBC.second.second;

        gmsh::model::mesh::getNodesForPhysicalGroup(m_fDim, physTag, nodeTags, coord);
        if (BCtype == "Reflecting")
        {
            for (int f = 0; f < m_fNum; ++f)
            {
                if (m_fIsBoundary[f] && std::find(nodeTags.begin(), nodeTags.end(), fNodeTag(f)) != nodeTags.end())
                    m_fBC[f] = 1;
            }
        }
        else
        {
            for (int f = 0; f < m_fNum; ++f)
            {
                if (m_fIsBoundary[f] && std::find(nodeTags.begin(), nodeTags.end(), fNodeTag(f)) != nodeTags.end())
                    m_fBC[f] = 0;
            }
        }
    }
    profMark("15_boundary_conditions", pt);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    /**
     * Compute the R*K*R matrix product. This matrix product is related to the absorbing
     * boundary conditions in the specific context of acoustic waves. It is mainly used
     * to suppress the outgoing solution along characteristics lines.
     */

    screen_display::write_string("Compute the R*K*R matrix product", GREEN);
    start = std::chrono::system_clock::now();

    pt = prof_clk::now();
    RKR.resize(m_fNum * m_fNumIntPts);

    // #pragma omp parallel for
    for (int f = 0; f < m_fNum; ++f)
    {
        if (m_fBC[f] == 0)
        {
            for (int g = 0; g < m_fNumIntPts; ++g)
            {
                int i = f * m_fNumIntPts + g;
                double nx(fNormal(f, g, 0)), ny(fNormal(f, g, 1)), nz(fNormal(f, g, 2));
                double tx(fTangent(f, g, 0)), ty(fTangent(f, g, 1)), tz(fTangent(f, g, 2));
                double sx(fBiTangent(f, g, 0)), sy(fBiTangent(f, g, 1)), sz(fBiTangent(f, g, 2));
                double c0(config.c0), rho0(config.rho0);
                double vx0(config.v0[0]), vy0(config.v0[1]), vz0(config.v0[2]);
                double vn0 = vx0 * nx + vy0 * ny + vz0 * nz;
                double lambda = (vn0 < 0) ? 0 : -1; // FIXME: Warning: lambda=-1 and not 1, this work but there is probably a bug in the code
                                                    // (pre-processing)...

                // double L1(fabs(vn0-c0)), L2(fabs(vn0+c0)), L3(fabs(vn0-c0));

                // screen_display::write_value("lambda",lambda);
                // getchar();

                RKR[i].resize(16);

                RKR[i][0] = 0.25 * (c0 + vn0);
                RKR[i][1] = 0.25 * (c0 * rho0 * (c0 + vn0) * nx);
                RKR[i][2] = 0.25 * (c0 * rho0 * (c0 + vn0) * ny);
                RKR[i][3] = 0.25 * (c0 * rho0 * (c0 + vn0) * nz);

                RKR[i][4] = 0.25 * (nx * (c0 + vn0) / (rho0 * c0));
                RKR[i][5] = 0.25 * ((c0 + vn0) * nx * nx - vn0 * lambda * (tx * tx + sx * sx));
                RKR[i][6] = 0.25 * ((c0 + vn0) * nx * ny - vn0 * lambda * (ty * tx + sy * sx));
                RKR[i][7] = 0.25 * ((c0 + vn0) * nx * nz - vn0 * lambda * (tz * tx + sz * sx));

                RKR[i][8] = 0.25 * (ny * (c0 + vn0) / (rho0 * c0));
                RKR[i][9] = 0.25 * ((c0 + vn0) * ny * nx - vn0 * lambda * (tx * ty + sx * sy));
                RKR[i][10] = 0.25 * ((c0 + vn0) * ny * ny - vn0 * lambda * (ty * ty + sy * sy));
                RKR[i][11] = 0.25 * ((c0 + vn0) * ny * nz - vn0 * lambda * (tz * ty + sz * sy));

                RKR[i][12] = 0.25 * (nz * (c0 + vn0) / (rho0 * c0));
                RKR[i][13] = 0.25 * ((c0 + vn0) * nz * nx - vn0 * lambda * (tx * tz + sx * sz));
                RKR[i][14] = 0.25 * ((c0 + vn0) * nz * ny - vn0 * lambda * (ty * tz + sy * sz));
                RKR[i][15] = 0.25 * ((c0 + vn0) * nz * nz - vn0 * lambda * (tz * tz + sz * sz));
            }
        }
    }

    profMark("16_RKR_matrix", pt);

    assert(m_fIsBoundary.size() == m_fNum);

#ifdef DG_USE_MPI
    if (Parallel::size() > 1)
    {
        pt = prof_clk::now();
        classifyFaces();
        buildHalo();
        buildLocalFaceList();
        profMark("17_mpi_halo_setup", pt);
    }
#endif

    /**
     * Extra Memory allocation:
     * Instantiate Ghost Elements and numerical flux storage.
     */
    pt = prof_clk::now();
    m_fFlux.resize(m_fNum * m_fNumNodes);
    uGhost = std::vector<std::vector<double>>(4,
                                              std::vector<double>(m_fNum * m_fNumIntPts));
    FluxGhost = std::vector<std::vector<std::vector<double>>>(4,
                                                              std::vector<std::vector<double>>(m_fNum * m_fNumIntPts,
                                                                                               std::vector<double>(3)));
    profMark("18_ghost_alloc", pt);

    gmsh::logger::write("Boundary conditions successfuly loaded.");
    gmsh::logger::write("==================================================");
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time:", elapsed.count() * 1.0e-6, "s", BLUE);

    // ----- Phase 7 profiler output -----
    if (profileMesh)
    {
        long long totalUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                prof_clk::now() - profTotalStart).count();
        const int rank = Parallel::rank();
        // CSV header (rank 0 only) for easy ingestion
        if (rank == 0)
        {
            std::cerr << "PROF_CSV,rank,section,seconds,share_percent" << std::endl;
        }
        // Sum
        long long sumUs = 0;
        for (auto &kv : profUs) sumUs += kv.second;
        for (auto &kv : profUs)
        {
            const double s     = kv.second / 1e6;
            const double share = sumUs > 0 ? 100.0 * kv.second / sumUs : 0.0;
            std::cerr << "PROF_CSV," << rank << "," << kv.first
                      << "," << s << "," << share << std::endl;
        }
        std::cerr << "PROF_CSV," << rank << ",TOTAL," << totalUs / 1e6
                  << ",100" << std::endl;
        std::cerr << "PROF_CSV," << rank << ",num_elements,"
                  << m_elNum << ",-1" << std::endl;
        std::cerr << "PROF_CSV," << rank << ",num_faces,"
                  << m_fNum << ",-1" << std::endl;
    }
}

#ifdef DG_USE_MPI

/**
 * No-op stub: the new partition path sets m_localElStart/End and
 * m_elOwnerRank directly from the gmsh PartitionLayout (see the
 * MPI branch in Mesh::Mesh). This function used to do a balanced
 * range partition over the GLOBAL element array, which has no
 * meaning anymore now that each rank only loads its local + halo.
 *
 * Kept as a stub to avoid breaking the call site in Mesh::Mesh
 * (already migrated, but the symbol is still declared in Mesh.h).
 */
void Mesh::partitionMesh()
{
    // intentionally empty
}

/**
 * Classify each face as:
 *  - boundary  : physical boundary (m_fIsBoundary already set)
 *  - internal  : both neighbour elements belong to this rank
 *  - interface : neighbours belong to different ranks
 *
 * Must be called after getConnectivityFaceToElement() and m_fIsBoundary are set.
 */
void Mesh::classifyFaces()
{
    int myRank = Parallel::rank();
    m_fIsInterface.assign(m_fNum, false);
    m_fNbrRank.assign(m_fNum, -1);

    int nInterface = 0;
    for (int f = 0; f < m_fNum; ++f)
    {
        if (m_fIsBoundary[f] || m_fNbrElIds[f].size() < 2)
            continue;

        int r0 = m_elOwnerRank[m_fNbrElIds[f][0]];
        int r1 = m_elOwnerRank[m_fNbrElIds[f][1]];

        // Only count faces where THIS rank owns one side. Otherwise (face
        // between two remote ranks) it is none of our business — flagging it
        // here would add foreign elements to our halo lists and break
        // size symmetry with neighbours.
        if (r0 != r1 && (r0 == myRank || r1 == myRank))
        {
            m_fIsInterface[f] = true;
            m_fNbrRank[f]     = (r0 == myRank) ? r1 : r0;
            ++nInterface;
        }
    }

    std::cout << "[MPI rank " << myRank << "] "
              << nInterface << " interface faces detected.\n" << std::flush;
}

/**
 * Build halo send/receive element lists from the classified faces.
 * For each interface face:
 *   - the local element index goes into m_haloSendElIds[remoteRank]
 *   - the remote element index goes into m_haloRecvElIds[remoteRank]
 * Duplicates are avoided (an element adjacent to several interface faces
 * with the same remote rank is only listed once).
 */
void Mesh::buildHalo()
{
    int myRank = Parallel::rank();

    for (int f = 0; f < m_fNum; ++f)
    {
        if (!m_fIsInterface[f]) continue;

        m_haloFaces.push_back(f);

        size_t el0 = m_fNbrElIds[f][0];
        size_t el1 = m_fNbrElIds[f][1];
        int    r0  = m_elOwnerRank[el0];
        int    r1  = m_elOwnerRank[el1];

        size_t localEl  = (r0 == myRank) ? el0 : el1;
        size_t remoteEl = (r0 == myRank) ? el1 : el0;
        int    remRank  = m_fNbrRank[f];

        auto &sendList = m_haloSendElIds[remRank];
        if (std::find(sendList.begin(), sendList.end(), localEl) == sendList.end())
            sendList.push_back(localEl);

        auto &recvList = m_haloRecvElIds[remRank];
        if (std::find(recvList.begin(), recvList.end(), remoteEl) == recvList.end())
            recvList.push_back(remoteEl);
    }

    // Sort each halo list by GMSH GLOBAL TAG of the element. Both ranks of
    // a pair sort the same way, so A's sendList[B][i] and B's recvList[A][i]
    // refer to the SAME physical element. This makes haloExchange() a pure
    // value transfer, no IDs needed.
    auto sortByGmshTag = [&](std::vector<size_t> &lst) {
        std::sort(lst.begin(), lst.end(),
                  [&](size_t a, size_t b) { return m_elTags[a] < m_elTags[b]; });
    };
    for (auto &kv : m_haloSendElIds) sortByGmshTag(kv.second);
    for (auto &kv : m_haloRecvElIds) sortByGmshTag(kv.second);

    for (auto &[rank, elList] : m_haloSendElIds)
        std::cout << "[MPI rank " << myRank << "] "
                  << elList.size() << " elements to send to rank " << rank << "\n";
    for (auto &[rank, elList] : m_haloRecvElIds)
        std::cout << "[MPI rank " << myRank << "] "
                  << elList.size() << " elements to receive from rank " << rank << "\n";
    std::cout << std::flush;
}

/**
 * Populate m_localFaces with every face index that has at least one local
 * neighbour (an element in [0, m_localElEnd)). Faces with no local neighbour
 * — typically halo-only outer faces — contribute to no local element's RHS,
 * so precomputeFlux can skip them entirely.
 *
 * Must be called AFTER face connectivity (m_fNbrElIds) is set, which
 * happens at the end of getConnectivityFaceToElement.
 */
void Mesh::buildLocalFaceList()
{
    m_localFaces.clear();
    m_localFaces.reserve(m_fNum);
    const size_t localEnd = static_cast<size_t>(m_localElEnd);
    for (int f = 0; f < m_fNum; ++f)
    {
        for (size_t el : m_fNbrElIds[f])
        {
            if (el < localEnd) { m_localFaces.push_back(f); break; }
        }
    }
    std::cout << "[MPI rank " << Parallel::rank() << "] "
              << m_localFaces.size() << " local faces (of " << m_fNum << ")\n"
              << std::flush;
}

/**
 * Load elements (and their per-int-pt Jacobians/coordinates) for the
 * calling rank, given a PartitionLayout. Concatenates owned elements
 * first, then halo elements:
 *   indices [0, Nloc)            → owned by this rank
 *   indices [Nloc, Nloc + Nhalo) → halo (ghost) cells, owned by remote
 *
 * Outputs (passed by reference, cleared then filled):
 *   elTags           : gmsh global element tags
 *   elNodeTags       : (Nloc + Nhalo) * elNumNodes gmsh global node tags
 *   jacobians        : (Nloc + Nhalo) * intPts * 9 doubles
 *   jacobianDets     : (Nloc + Nhalo) * intPts doubles
 *   intPtCoords      : (Nloc + Nhalo) * intPts * 3 doubles
 *   haloOwnerRank    : size Nhalo, owner rank (0-based) of each halo element
 *   nLocal           : Nloc
 *   nHalo            : Nhalo
 */
static void loadElementsByPartition(const PartitionLayout &layout,
                                    int elementType,
                                    const std::vector<double> &integrationParamCoords,
                                    std::vector<std::size_t> &elTags,
                                    std::vector<std::size_t> &elNodeTags,
                                    std::vector<double> &jacobians,
                                    std::vector<double> &jacobianDets,
                                    std::vector<double> &intPtCoords,
                                    std::vector<int> &haloOwnerRank,
                                    int &nLocal,
                                    int &nHalo)
{
    elTags.clear();
    elNodeTags.clear();
    jacobians.clear();
    jacobianDets.clear();
    intPtCoords.clear();
    haloOwnerRank.clear();

    auto loadEntity = [&](int dim, int entTag) -> std::size_t
    {
        std::vector<std::size_t> et, ent;
        gmsh::model::mesh::getElementsByType(elementType, et, ent, entTag);
        if (et.empty()) return 0;

        std::vector<double> jac, det, coords;
        gmsh::model::mesh::getJacobians(elementType, integrationParamCoords,
                                        jac, det, coords, entTag);

        elTags.insert(elTags.end(), et.begin(), et.end());
        elNodeTags.insert(elNodeTags.end(), ent.begin(), ent.end());
        jacobians.insert(jacobians.end(), jac.begin(), jac.end());
        jacobianDets.insert(jacobianDets.end(), det.begin(), det.end());
        intPtCoords.insert(intPtCoords.end(), coords.begin(), coords.end());
        return et.size();
    };

    // 1) Owned elements
    nLocal = 0;
    for (int entTag : layout.ownedEntities3D)
        nLocal += static_cast<int>(loadEntity(3, entTag));

    // 2) Halo elements + owner rank lookup
    nHalo = 0;
    for (int entTag : layout.haloEntities3D)
    {
        // Get the ghost-tag → owner-partition map for THIS halo entity
        std::vector<std::size_t> ghostTags;
        std::vector<int>         ghostParts;
        gmsh::model::mesh::getGhostElements(3, entTag, ghostTags, ghostParts);
        std::unordered_map<std::size_t, int> tagToOwner;
        tagToOwner.reserve(ghostTags.size());
        for (std::size_t i = 0; i < ghostTags.size(); ++i)
            tagToOwner[ghostTags[i]] = ghostParts[i] - 1; // partitions are 1-based

        const std::size_t before = elTags.size();
        const std::size_t added  = loadEntity(3, entTag);
        nHalo += static_cast<int>(added);

        for (std::size_t i = 0; i < added; ++i)
        {
            auto it = tagToOwner.find(elTags[before + i]);
            haloOwnerRank.push_back(it != tagToOwner.end() ? it->second : -1);
        }
    }
}

/**
 * Concatenate barycenters owned-first then halo, in the same order as
 * elTags returned by loadElementsByPartition.
 */
static void loadBarycentersByPartition(const PartitionLayout &layout,
                                       int elementType,
                                       std::vector<double> &barycenters)
{
    barycenters.clear();
    auto add = [&](int dim, int entTag) {
        std::vector<double> b;
        gmsh::model::mesh::getBarycenters(elementType, entTag, false, true, b);
        barycenters.insert(barycenters.end(), b.begin(), b.end());
    };
    for (int t : layout.ownedEntities3D) add(3, t);
    for (int t : layout.haloEntities3D) add(3, t);
}

/**
 * Concatenate per-element face (or edge) node tags owned-first then halo,
 * in the same order as elTags. faceNumNodes = 4 (hex faces), 3 (tet faces),
 * or use the edge-node API when faceNumNodes <= 0.
 */
static void loadFaceNodesByPartition(const PartitionLayout &layout,
                                     int elementType,
                                     int faceNumNodes,
                                     std::vector<std::size_t> &fNodeTags)
{
    fNodeTags.clear();
    auto add = [&](int dim, int entTag) {
        std::vector<std::size_t> tmp;
        if (faceNumNodes > 0)
            gmsh::model::mesh::getElementFaceNodes(elementType, faceNumNodes, tmp, entTag);
        else
            gmsh::model::mesh::getElementEdgeNodes(elementType, tmp, entTag);
        fNodeTags.insert(fNodeTags.end(), tmp.begin(), tmp.end());
    };
    for (int t : layout.ownedEntities3D) add(3, t);
    for (int t : layout.haloEntities3D) add(3, t);
}

/**
 * Run gmsh partitioning and classify entities for the calling rank.
 * Idempotent: a second call on the same model does not partition again
 * — gmsh is left as-is and we just re-read the existing entity layout.
 *
 * The function intentionally does NOT touch any Mesh member: it only
 * inspects gmsh state and returns the layout struct. Plugging this
 * into Mesh::Mesh() is the next step (3b).
 */
PartitionLayout buildPartitionLayout()
{
    const int myPart = Parallel::rank() + 1; // gmsh partition tags are 1-based
    const int N      = Parallel::size();

    // 1) Partition the mesh (only if not already partitioned).
    //    Heuristic for "already partitioned": at least one 3D entity has
    //    a non-empty partition list.
    bool alreadyPartitioned = false;
    {
        std::vector<std::pair<int,int>> ents;
        gmsh::model::getEntities(ents, 3);
        for (auto &[d, t] : ents)
        {
            std::vector<int> ps;
            gmsh::model::getPartitions(d, t, ps);
            if (!ps.empty()) { alreadyPartitioned = true; break; }
        }
    }

    if (!alreadyPartitioned)
    {
        gmsh::option::setNumber("Mesh.PartitionCreateGhostCells", 1);
        gmsh::option::setNumber("Mesh.PartitionCreateTopology",   1);
        gmsh::model::mesh::partition(N);
    }

    PartitionLayout layout;

    // 2) Walk 3D entities. An entity belongs to this rank if its partition
    //    list contains myPart. We then split owned vs halo by the presence
    //    of ghost elements.
    {
        std::vector<std::pair<int,int>> ents;
        gmsh::model::getEntities(ents, 3);
        for (auto &[d, tag] : ents)
        {
            std::vector<int> partitions;
            gmsh::model::getPartitions(d, tag, partitions);
            if (std::find(partitions.begin(), partitions.end(), myPart) == partitions.end())
                continue;

            std::vector<std::size_t> ghostTags;
            std::vector<int>         ghostParts;
            gmsh::model::mesh::getGhostElements(d, tag, ghostTags, ghostParts);

            if (ghostTags.empty())
                layout.ownedEntities3D.push_back(tag);
            else
                layout.haloEntities3D.push_back(tag);
        }
    }

    // 3) Walk 2D entities. Three categories:
    //    - partitions = {myPart}        → physical boundary face of this rank
    //    - partitions = {myPart, k}     → interface with rank k-1
    //    - everything else              → not relevant for this rank
    {
        std::vector<std::pair<int,int>> ents;
        gmsh::model::getEntities(ents, 2);
        for (auto &[d, tag] : ents)
        {
            std::vector<int> partitions;
            gmsh::model::getPartitions(d, tag, partitions);
            if (partitions.empty()) continue;
            if (std::find(partitions.begin(), partitions.end(), myPart) == partitions.end())
                continue;

            if (partitions.size() == 1)
            {
                layout.boundaryEntities2D.push_back(tag);
            }
            else if (partitions.size() == 2)
            {
                int otherPart = (partitions[0] == myPart) ? partitions[1] : partitions[0];
                layout.interfaceEntitiesByRank[otherPart - 1].push_back(tag);
            }
            // size > 2 (triple junction etc.) is rare in practice; skip for now.
        }
    }

    // 4) Diagnostics
    {
        size_t nOwnedEls = 0, nHaloEls = 0, nBdryFaces = 0, nInterfaceFaces = 0;
        for (int tag : layout.ownedEntities3D)
        {
            std::vector<int> et;
            std::vector<std::vector<std::size_t>> elt, nt;
            gmsh::model::mesh::getElements(et, elt, nt, 3, tag);
            for (auto &v : elt) nOwnedEls += v.size();
        }
        for (int tag : layout.haloEntities3D)
        {
            std::vector<int> et;
            std::vector<std::vector<std::size_t>> elt, nt;
            gmsh::model::mesh::getElements(et, elt, nt, 3, tag);
            for (auto &v : elt) nHaloEls += v.size();
        }
        for (int tag : layout.boundaryEntities2D)
        {
            std::vector<int> et;
            std::vector<std::vector<std::size_t>> elt, nt;
            gmsh::model::mesh::getElements(et, elt, nt, 2, tag);
            for (auto &v : elt) nBdryFaces += v.size();
        }
        for (auto &kv : layout.interfaceEntitiesByRank)
            for (int tag : kv.second)
            {
                std::vector<int> et;
                std::vector<std::vector<std::size_t>> elt, nt;
                gmsh::model::mesh::getElements(et, elt, nt, 2, tag);
                for (auto &v : elt) nInterfaceFaces += v.size();
            }

        std::cout << "[MPI rank " << Parallel::rank() << "] partition layout: "
                  << nOwnedEls   << " owned 3D els, "
                  << nHaloEls    << " halo 3D els, "
                  << nBdryFaces  << " physical bdry faces, "
                  << nInterfaceFaces << " interface faces"
                  << " (" << layout.interfaceEntitiesByRank.size() << " neighbours)\n"
                  << std::flush;
    }

    return layout;
}

#endif // DG_USE_MPI

void Mesh::haloExchange(std::vector<std::vector<double>> &u)
{
#ifdef DG_USE_MPI
    if (Parallel::size() <= 1)
        return;

    std::set<int> neighbours;
    for (const auto &entry : m_haloSendElIds)
        neighbours.insert(entry.first);
    for (const auto &entry : m_haloRecvElIds)
        neighbours.insert(entry.first);

    const int fieldCount = static_cast<int>(u.size());
    const int dofsPerEl = m_elNumNodes;

    // Halo lists are sorted by gmsh element tag at construction time
    // (see buildHalo). So A's sendList[B][i] and B's recvList[A][i]
    // refer to the SAME physical element — no IDs need to be exchanged.
    for (int neighbourRank : neighbours)
    {
        const auto sendIt = m_haloSendElIds.find(neighbourRank);
        const auto recvIt = m_haloRecvElIds.find(neighbourRank);

        const std::vector<size_t> emptyEls;
        const std::vector<size_t> &sendEls = (sendIt != m_haloSendElIds.end()) ? sendIt->second : emptyEls;
        const std::vector<size_t> &recvEls = (recvIt != m_haloRecvElIds.end()) ? recvIt->second : emptyEls;

        std::vector<double> sendValues(sendEls.size() * fieldCount * dofsPerEl);
        std::vector<double> recvValues(recvEls.size() * fieldCount * dofsPerEl);

        size_t sendOffset = 0;
        for (size_t elIndex = 0; elIndex < sendEls.size(); ++elIndex)
        {
            const size_t el = sendEls[elIndex]; // local owned-region index
            for (int eq = 0; eq < fieldCount; ++eq)
            {
                std::copy_n(u[eq].begin() + el * dofsPerEl,
                            dofsPerEl,
                            sendValues.begin() + sendOffset);
                sendOffset += dofsPerEl;
            }
        }

        MPI_Sendrecv(sendValues.data(), static_cast<int>(sendValues.size()), MPI_DOUBLE, neighbourRank, 4101,
                     recvValues.data(), static_cast<int>(recvValues.size()), MPI_DOUBLE, neighbourRank, 4101,
                     Parallel::comm(), MPI_STATUS_IGNORE);

        size_t recvOffset = 0;
        for (size_t elIndex = 0; elIndex < recvEls.size(); ++elIndex)
        {
            const size_t el = recvEls[elIndex]; // local halo-region index
            for (int eq = 0; eq < fieldCount; ++eq)
            {
                std::copy_n(recvValues.begin() + recvOffset,
                            dofsPerEl,
                            u[eq].begin() + el * dofsPerEl);
                recvOffset += dofsPerEl;
            }
        }
    }
#else
    (void)u;
#endif
}

/**
 * Precompute and store the mass matris for all elements in m_elMassMatrix
 */
void Mesh::precomputeMassMatrix()
{
    m_elMassMatrices.resize(m_elNum * m_elNumNodes * m_elNumNodes);
    // Only local elements need an inverse mass matrix: numStep updates only
    // [m_localElStart, m_localElEnd). Halo cells are read-only (overwritten
    // by haloExchange).
#ifdef DG_USE_MPI
    if (Parallel::size() > 1)
    {
        for (int el = m_localElStart; el < m_localElEnd; ++el)
            getElMassMatrix(el, true, &elMassMatrix(el));
        return;
    }
#endif
    for (size_t el = 0; el < m_elNum; ++el)
        getElMassMatrix(el, true, &elMassMatrix(el));
}

/**
 * Compute the element mass matrix.
 *
 * @param el integer : element id (!= gmsh tag, it is the location in memory storage)
 * @param inverse boolean : Whether or not the mass matrix must be inverted before returned
 * @param elMassMatrix double array : Output storage of the element mass matrix
 */
void Mesh::getElMassMatrix(const size_t el, const bool inverse, double *elMassMatrix)
{
    for (int i = 0; i < m_elNumNodes; ++i)
    {
        for (int j = 0; j < m_elNumNodes; ++j)
        {
            elMassMatrix[i * m_elNumNodes + j] = 0.0;
            for (int g = 0; g < m_elNumIntPts; g++)
            {
                elMassMatrix[i * m_elNumNodes + j] += elBasisFct(g, i) * elBasisFct(g, j) *
                                                      m_elWeight[g] * elJacobianDet(el, g);
            }
        }
    }
    if (inverse)
        eigen::inverse(elMassMatrix, m_elNumNodes);
}

/**
 * Compute the element stiffness/convection matrix.
 *
 * @param el integer : element id
 * @param Flux double array : physical flux
 * @param u double array : solution at element node
 * @param elStiffVector double array : Output storage of the element stiffness vector
 */
void Mesh::getElStiffVector(const size_t el, std::vector<std::vector<double>> &Flux,
                            std::vector<double> &u, double *elStiffVector)
{
    int jId;
    for (int i = 0; i < m_elNumNodes; ++i)
    {
        elStiffVector[i] = 0.0;
        for (int j = 0; j < m_elNumNodes; ++j)
        {
            jId = el * m_elNumNodes + j;
            for (int g = 0; g < m_elNumIntPts; g++)
            {
                elStiffVector[i] += eigen::dot(Flux[jId].data(), &elGradBasisFct(el, g, i), m_Dim) *
                                    elBasisFct(g, j) * m_elWeight[g] * elJacobianDet(el, g);
            }
        }
    }
}

/**
 * Precompute the numerical flux through all the faces. The flux implemented is
 * the Rusanov Flux. Also note that the following code is paralelized using openMP.
 *
 * @param Flux double array : physical flux
 * @param u double array : solution at the node
 * @param eq : equation id (0 = pressure, 1 = velocity x, 2= vy, 3= vz)
 */
void Mesh::precomputeFlux(std::vector<double> &u, std::vector<std::vector<double>> &Flux, int eq)
{
    // In MPI mode, only iterate over faces touching at least one local
    // element. Faces with no local neighbour contribute to no local
    // element's getElFlux → wasted work that scales with the halo size.
#ifdef DG_USE_MPI
    const bool useLocalFaces = (Parallel::size() > 1);
    const int nFacesToProcess = useLocalFaces ? static_cast<int>(m_localFaces.size())
                                              : m_fNum;
#else
    const int nFacesToProcess = m_fNum;
#endif

#pragma omp parallel num_threads(config.numThreads)
    {
        // Memory allocation (Cross-plateform compatibility)
        size_t elUp, elDn;
        std::vector<double> FIntPts(m_fNumIntPts, 0);
        std::vector<double> Fnum(m_Dim, 0);

#pragma omp parallel for schedule(static)
        for (int kf = 0; kf < nFacesToProcess; ++kf)
        {
#ifdef DG_USE_MPI
            const int f = useLocalFaces ? static_cast<int>(m_localFaces[kf]) : kf;
#else
            const int f = kf;
#endif

            std::fill(FIntPts.begin(), FIntPts.end(), 0);

            // Numerical Flux at Integration points.
            // Interface faces (MPI) are NOT special-cased: after haloExchange,
            // u[halo] is valid and updateFlux fills Flux[halo] for the full element
            // range — so the standard interior Rusanov path applies.
            if (m_fIsBoundary[f])
            {
                for (int g = 0; g < m_fNumIntPts; ++g)
                    FIntPts[g] = FluxGhost[eq][f * m_fNumIntPts + g][0];
            }
            else
            {
                for (int i = 0; i < m_fNumNodes; ++i)
                {
                    elUp = fNbrElId(f, 0) * m_elNumNodes + fNToElNId(f, i, 0);
                    elDn = fNbrElId(f, 1) * m_elNumNodes + fNToElNId(f, i, 1);
                    for (int g = 0; g < m_fNumIntPts; ++g)
                    {
                        for (int x = 0; x < m_Dim; ++x)
                            Fnum[x] = 0.5 * ((Flux[elUp][x] + Flux[elDn][x]) + fc * config.c0 * fNormal(f, g, x) * (u[elUp] - u[elDn]));
/////////////////////////
#pragma omp atomic update
                        FIntPts[g] += eigen::dot(&fNormal(f, g), Fnum.data(), m_Dim) * fBasisFct(g, i);
                    }
                }
            }

            // Surface integral
            for (int n = 0; n < m_fNumNodes; ++n)
            {
                fFlux(f, n) = 0;
                for (int g = 0; g < m_fNumIntPts; ++g)
                {
////////////////////////
#pragma omp atomic update
                    fFlux(f, n) += m_fWeight[g] * fBasisFct(g, n) * FIntPts[g] * fJacobianDet(f, g);
                }
            }
        }
    }
}

/**
 * Compute flux through a given element from
 * the value of the flux at the face.
 *
 * @param el integer : element id
 * @param F double array : Output element flux
 */
void Mesh::getElFlux(const size_t el, double *F)
{
    int i;
    std::fill(F, F + m_elNumNodes, 0);
    // #pragma omp parallel num_threads(config.numThreads)
    {
        // #pragma omp parallel for schedule(static)
        for (int f = 0; f < m_fNumPerEl; ++f)
        {
            el == fNbrElId(elFId(el, f), 0) ? i = 0 : i = 1;
            for (int nf = 0; nf < m_fNumNodes; ++nf)
            {
                // #pragma omp atomic
                F[fNToElNId(elFId(el, f), nf, i)] += elFOrientation(el, f) * fFlux(elFId(el, f), nf);
            }
        }
    }
}

/**
 * Compute physical flux from the nodal solution. Also update
 * the ghost element and numerical flux.
 *
 * @param u : nodal solution vector
 * @param Flux : Physical flux
 * @param v0 : mean flow speed (v0x,v0y,v0z)
 * @param c0 : speed of sound
 * @param rho0: mean flow density
 */
void Mesh::updateFlux(std::vector<std::vector<double>> &u, std::vector<std::vector<std::vector<double>>> &Flux,
                      std::vector<double> &v0, double c0, double rho0)
{

// #pragma omp parallel for
#pragma omp parallel for schedule(static) num_threads(config.numThreads)
    for (size_t el = 0; el < m_elNum; ++el)
    {
        for (int n = 0; n < m_elNumNodes; ++n)
        {
            int i = el * m_elNumNodes + n;

            // Pressure flux
            Flux[0][i] = {v0[0] * u[0][i] + rho0 * c0 * c0 * u[1][i],
                          v0[1] * u[0][i] + rho0 * c0 * c0 * u[2][i],
                          v0[2] * u[0][i] + rho0 * c0 * c0 * u[3][i]};
            // Vx
            Flux[1][i] = {v0[0] * u[1][i] + u[0][i] / rho0,
                          v0[1] * u[1][i],
                          v0[2] * u[1][i]};
            // Vy
            Flux[2][i] = {v0[0] * u[2][i],
                          v0[1] * u[2][i] + u[0][i] / rho0,
                          v0[2] * u[2][i]};
            // Vz
            Flux[3][i] = {v0[0] * u[3][i],
                          v0[1] * u[3][i],
                          v0[2] * u[3][i] + u[0][i] / rho0};
        }

        // Ghost elements
        for (int f = 0; f < m_fNumPerEl; ++f)
        {
            int fId = elFId(el, f);
            if (m_fIsBoundary[fId])
            {

                for (int g = 0; g < m_fNumIntPts; ++g)
                {
                    int gId = fId * m_fNumIntPts + g;

                    // Interpolate solution at integration points
                    uGhost[0][gId] = 0;
                    uGhost[1][gId] = 0;
                    uGhost[2][gId] = 0;
                    uGhost[3][gId] = 0;
                    for (int n = 0; n < m_fNumNodes; ++n)
                    {
                        int nId = el * m_elNumNodes + fNToElNId(fId, n, 0);
////////////////////////
#pragma omp atomic
                        uGhost[0][gId] += u[0][nId] * fBasisFct(g, n);
#pragma omp atomic
                        uGhost[1][gId] += u[1][nId] * fBasisFct(g, n);
#pragma omp atomic
                        uGhost[2][gId] += u[2][nId] * fBasisFct(g, n);
#pragma omp atomic
                        uGhost[3][gId] += u[3][nId] * fBasisFct(g, n);
                    }

                    if (m_fBC[fId] == 1)
                    {
                        double nx(fNormal(fId, g, 0)), ny(fNormal(fId, g, 1)), nz(fNormal(fId, g, 2));
                        double dot = nx * uGhost[1][gId] +
                                     ny * uGhost[2][gId] +
                                     nz * uGhost[3][gId];
// #pragma omp critical
                        // std::cout << nx << " " << ny << " " << nz << std::endl;

// Remove normal component (Rigid Wall BC)
#pragma omp atomic
                        uGhost[1][gId] -= dot * nx;
#pragma omp atomic
                        uGhost[2][gId] -= dot * ny;
#pragma omp atomic
                        uGhost[3][gId] -= dot * nz;

                        // Flux at integration points
                        // 1) Pressure flux
                        FluxGhost[0][gId] = {v0[0] * uGhost[0][gId] + rho0 * c0 * c0 * uGhost[1][gId],
                                             v0[1] * uGhost[0][gId] + rho0 * c0 * c0 * uGhost[2][gId],
                                             v0[2] * uGhost[0][gId] + rho0 * c0 * c0 * uGhost[3][gId]};
                        // 2) Vx
                        FluxGhost[1][gId] = {v0[0] * uGhost[1][gId] + uGhost[0][gId] / rho0,
                                             v0[1] * uGhost[1][gId],
                                             v0[2] * uGhost[1][gId]};
                        // 3) Vy
                        FluxGhost[2][gId] = {v0[0] * uGhost[2][gId],
                                             v0[1] * uGhost[2][gId] + uGhost[0][gId] / rho0,
                                             v0[2] * uGhost[2][gId]};
                        // 4) Vz
                        FluxGhost[3][gId] = {v0[0] * uGhost[3][gId],
                                             v0[1] * uGhost[3][gId],
                                             v0[2] * uGhost[3][gId] + uGhost[0][gId] / rho0};

                        // Project Flux on the normal
                        for (int eq = 0; eq < 4; ++eq)
                            FluxGhost[eq][gId][0] = eigen::dot(&fNormal(fId, g), &FluxGhost[eq][gId][0], m_Dim);
                    }
                    else
                    {
                        // Absorbing boundary conditions
                        // /!\ Flux already projected on normal,
                        FluxGhost[0][gId][0] = RKR[gId][0] * uGhost[0][gId] +
                                               RKR[gId][1] * uGhost[1][gId] +
                                               RKR[gId][2] * uGhost[2][gId] +
                                               RKR[gId][3] * uGhost[3][gId];
                        FluxGhost[1][gId][0] = RKR[gId][4] * uGhost[0][gId] +
                                               RKR[gId][5] * uGhost[1][gId] +
                                               RKR[gId][6] * uGhost[2][gId] +
                                               RKR[gId][7] * uGhost[3][gId];
                        FluxGhost[2][gId][0] = RKR[gId][8] * uGhost[0][gId] +
                                               RKR[gId][9] * uGhost[1][gId] +
                                               RKR[gId][10] * uGhost[2][gId] +
                                               RKR[gId][11] * uGhost[3][gId];
                        FluxGhost[3][gId][0] = RKR[gId][12] * uGhost[0][gId] +
                                               RKR[gId][13] * uGhost[1][gId] +
                                               RKR[gId][14] * uGhost[2][gId] +
                                               RKR[gId][15] * uGhost[3][gId];
                    }
                }
            }
        }
    }

#ifdef DG_USE_MPI
    if (Parallel::size() > 1)
    {
#pragma omp parallel for schedule(static) num_threads(config.numThreads)
        for (int fId = 0; fId < m_fNum; ++fId)
        {
            if (!m_fIsInterface[fId])
                continue;

            const size_t el0 = fNbrElId(fId, 0);
            const size_t el1 = fNbrElId(fId, 1);

            for (int g = 0; g < m_fNumIntPts; ++g)
            {
                const int gId = fId * m_fNumIntPts + g;
                double u0[4] = {0.0, 0.0, 0.0, 0.0};
                double u1[4] = {0.0, 0.0, 0.0, 0.0};

                for (int n = 0; n < m_fNumNodes; ++n)
                {
                    const size_t n0 = el0 * m_elNumNodes + fNToElNId(fId, n, 0);
                    const size_t n1 = el1 * m_elNumNodes + fNToElNId(fId, n, 1);
                    const double basis = fBasisFct(g, n);

                    for (int eq = 0; eq < 4; ++eq)
                    {
                        u0[eq] += u[eq][n0] * basis;
                        u1[eq] += u[eq][n1] * basis;
                    }
                }

                const double nx = fNormal(fId, g, 0);
                const double ny = fNormal(fId, g, 1);
                const double nz = fNormal(fId, g, 2);

                const double flux0[4][3] = {
                    {v0[0] * u0[0] + rho0 * c0 * c0 * u0[1], v0[1] * u0[0] + rho0 * c0 * c0 * u0[2], v0[2] * u0[0] + rho0 * c0 * c0 * u0[3]},
                    {v0[0] * u0[1] + u0[0] / rho0, v0[1] * u0[1], v0[2] * u0[1]},
                    {v0[0] * u0[2], v0[1] * u0[2] + u0[0] / rho0, v0[2] * u0[2]},
                    {v0[0] * u0[3], v0[1] * u0[3], v0[2] * u0[3] + u0[0] / rho0}};

                const double flux1[4][3] = {
                    {v0[0] * u1[0] + rho0 * c0 * c0 * u1[1], v0[1] * u1[0] + rho0 * c0 * c0 * u1[2], v0[2] * u1[0] + rho0 * c0 * c0 * u1[3]},
                    {v0[0] * u1[1] + u1[0] / rho0, v0[1] * u1[1], v0[2] * u1[1]},
                    {v0[0] * u1[2], v0[1] * u1[2] + u1[0] / rho0, v0[2] * u1[2]},
                    {v0[0] * u1[3], v0[1] * u1[3], v0[2] * u1[3] + u1[0] / rho0}};

                for (int eq = 0; eq < 4; ++eq)
                {
                    const double proj0 = nx * flux0[eq][0] + ny * flux0[eq][1] + nz * flux0[eq][2];
                    const double proj1 = nx * flux1[eq][0] + ny * flux1[eq][1] + nz * flux1[eq][2];
                    FluxGhost[eq][gId][0] = 0.5 * ((proj0 + proj1) + fc * config.c0 * (u0[eq] - u1[eq]));
                }
            }
        }
    }
#endif
}

/**
 * List of nodes for each unique face given a list of node per face and per elements
 */

void Mesh::getUniqueFaceNodeTags()
{
    // Phase 9.2 refactor: hashmap-based deduplication of face node tags.
    //
    // Old algorithm: for each face in m_elFNodeTags, scan a "global face list"
    // (from gmsh::createFaces / getAllFaces) by linear search → O(Nf²)
    // dominated everything (94% of preprocessing on 16k tets).
    //
    // New algorithm: build a sorted "native-node" key per face and insert
    // into an unordered_map. First-occurrence wins; subsequent occurrences
    // are duplicates and dropped. → O(Nf) average.

    const bool hasQuadrilateralFaces = (m_fDim == 2 && m_fName == "quadrangle");
    const size_t fNumNativeNodes = (m_fDim < 2) ? 2 : (hasQuadrilateralFaces ? 4 : 3);
    const size_t totalFaces = m_elFNodeTags.size() / m_fNumNodes;

    auto start = std::chrono::system_clock::now();

    // 1) Pre-sort each face's full node list (kept for downstream consumers).
    m_elFNodeTagsOrdered = m_elFNodeTags;
    for (size_t i = 0; i + m_fNumNodes <= m_elFNodeTagsOrdered.size(); i += m_fNumNodes)
        std::sort(m_elFNodeTagsOrdered.begin() + i,
                  m_elFNodeTagsOrdered.begin() + (i + m_fNumNodes));

    // 2) Build canonical key per face from its sorted native-node tuple.
    //    For typical sizes (2/3/4 native nodes), keep keys in a small fixed
    //    array and use a custom hash combiner — much faster than std::string.
    auto canonicalKey = [&](size_t faceIdx, std::array<size_t, 4> &out) {
        for (size_t k = 0; k < fNumNativeNodes; ++k)
            out[k] = m_elFNodeTagsOrdered[faceIdx * m_fNumNodes + k];
        for (size_t k = fNumNativeNodes; k < 4; ++k)
            out[k] = 0;
    };

    struct KeyHash {
        size_t operator()(const std::array<size_t, 4> &k) const noexcept {
            // splitmix-style mix; cheap and well-distributed for moderate Nf
            size_t h = 1469598103934665603ull;
            for (auto v : k) {
                h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    std::unordered_map<std::array<size_t, 4>, size_t, KeyHash> firstOccurrence;
    firstOccurrence.reserve(totalFaces);
    std::vector<size_t> uniqueFaceIndices;
    uniqueFaceIndices.reserve(totalFaces);

    std::array<size_t, 4> key{0, 0, 0, 0};
    for (size_t i = 0; i < totalFaces; ++i)
    {
        canonicalKey(i, key);
        auto [it, inserted] = firstOccurrence.try_emplace(key, i);
        if (inserted) uniqueFaceIndices.push_back(i);
    }

    // 3) Materialize m_fNodeTags by copying the first-occurrence face's full
    //    node list — preserves the original node ordering used downstream
    //    (orientation, Jacobians, addElementsByType).
    m_fNodeTags.clear();
    m_fNodeTags.reserve(uniqueFaceIndices.size() * m_fNumNodes);
    for (size_t idx : uniqueFaceIndices)
    {
        m_fNodeTags.insert(m_fNodeTags.end(),
                           m_elFNodeTags.begin() + idx * m_fNumNodes,
                           m_elFNodeTags.begin() + (idx + 1) * m_fNumNodes);
    }

    // 4) Sorted version, used by the connectivity step.
    m_fNodeTagsOrdered = m_fNodeTags;
    for (size_t i = 0; i + m_fNumNodes <= m_fNodeTagsOrdered.size(); i += m_fNumNodes)
        std::sort(m_fNodeTagsOrdered.begin() + i,
                  m_fNodeTagsOrdered.begin() + (i + m_fNumNodes));

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time (getUniqueFaceNodeTags):", elapsed.count() * 1.0e-6, "s", BLUE);
}

std::vector<size_t> vector_of_tags(size_t vec_size, size_t offset)
{
    std::vector<size_t> value;
    // std::cout<<offset<<std::endl;
    for (size_t i = 0, id = 0; i < vec_size; i++)
    {
        if (i % offset == 0)
            id++;
        value.push_back(id - 1);
        // value.push_back((i % offset == 0)?(id++ - 1):id);
    }
    return value;
}
void Mesh::getConnectivityFaceToElement()
{
    //! TIMER start ////////////////////////////////
    auto start = std::chrono::system_clock::now();
    //! ////////////////////////////////////////////
    m_fNbrElIds.resize(m_fNum);

    std::vector<std::vector<size_t>> m_elFNodeTagsOrdered_tab = vector_to_matrix(m_elFNodeTagsOrdered, m_fNumNodes);
    std::vector<std::vector<size_t>> m_fNodeTagsOrdered_tab = vector_to_matrix(m_fNodeTagsOrdered, m_fNumNodes);

    std::vector<size_t> elFtags = vector_of_tags(m_elFNodeTagsOrdered_tab.size(), m_fNumPerEl); //! vector of elements face tags
    std::vector<size_t> ftags = vector_of_tags(m_fNodeTagsOrdered_tab.size(), 1);               //! vector of face tags

    for (size_t i = 0; i < m_elFNodeTagsOrdered_tab.size(); i++)
    {
        for (size_t j = 0; j < m_fNodeTagsOrdered_tab.size(); j++)
        {
            if (isNCoincidentValues(m_elFNodeTagsOrdered_tab[i], m_fNodeTagsOrdered_tab[j], m_fNumNodes))
            {
                m_elFIds.push_back(ftags[j]);
                m_fNbrElIds[ftags[j]].push_back(elFtags[i]);
                if (m_fNbrElIds[ftags[j]].size() == 2)
                {
                    erase_row_from_matrix(m_fNodeTagsOrdered_tab, j);
                    erase_row_from_vector(ftags, j);
                    break;
                }
                // if(flag) break;
            }
        }
    }

    //! TIMER END ///////////////////////////////////////////////////////////////////
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    screen_display::write_value("Elapsed time (getConnectivityFaceToElement):", elapsed.count() * 1.0e-6, "s", BLUE);
    //! //////////////////////////////////////////////////////////////////////////////
}

/**
 * @brief Write VTK & PVD
 */

void Mesh::writeVTUb(std::string filename, std::vector<std::vector<double>> &u)
{
    // std::string filename = filename;
    screen_display::write_string("Write VTU: " + filename, BOLDRED);

    int vtkCellType = VTK_EMPTY_CELL;
    size_t vtkCellNumNodes = 0;

    if (m_elName.find("Quadrilateral") != std::string::npos)
    {
        vtkCellType = VTK_QUAD;
        vtkCellNumNodes = 4;
    }
    else if (m_elName.find("Hexahedron") != std::string::npos)
    {
        vtkCellType = VTK_HEXAHEDRON;
        vtkCellNumNodes = 8;
    }
    else if (m_elName.find("Triangle") != std::string::npos)
    {
        vtkCellType = VTK_TRIANGLE;
        vtkCellNumNodes = 3;
    }
    else if (m_elName.find("Tetrahedron") != std::string::npos)
    {
        vtkCellType = VTK_TETRA;
        vtkCellNumNodes = 4;
    }
    else
    {
        Fatal_Error("Unsupported VTU cell type");
    }

    vtkNew<vtkPoints> points;
    points->SetDataTypeToDouble();

    vtkNew<vtkCellArray> cellArray;
    vtkNew<vtkDoubleArray> pressure, density, velocity;
    vtkNew<vtkUnstructuredGrid> unstructuredGrid;
    vtkNew<vtkXMLUnstructuredGridWriter> writer;

        const size_t elBegin =
    #ifdef DG_USE_MPI
        (Parallel::size() > 1) ? static_cast<size_t>(m_localElStart) : 0;
    #else
        0;
    #endif
        const size_t elEnd =
    #ifdef DG_USE_MPI
        (Parallel::size() > 1) ? static_cast<size_t>(m_localElEnd) : static_cast<size_t>(getElNum());
    #else
        static_cast<size_t>(getElNum());
    #endif

    std::vector<size_t> usedNodeTags;
        usedNodeTags.reserve((elEnd - elBegin) * vtkCellNumNodes);
    std::unordered_map<size_t, vtkIdType> vtkPointIdByNodeTag;

        for (size_t el = elBegin; el < elEnd; ++el)
    {
        for (size_t n = 0; n < vtkCellNumNodes; ++n)
        {
            const size_t nodeTag = elNodeTag(el, n);
            if (vtkPointIdByNodeTag.find(nodeTag) == vtkPointIdByNodeTag.end())
            {
                vtkPointIdByNodeTag[nodeTag] = static_cast<vtkIdType>(usedNodeTags.size());
                usedNodeTags.push_back(nodeTag);
            }
        }
    }

    for (size_t nodeTag : usedNodeTags)
    {
        std::vector<double> coord, paramCoord;
        int _dim, _tag;
        gmsh::model::mesh::getNode(nodeTag, coord, paramCoord, _dim, _tag);
        points->InsertNextPoint(coord[0], coord[1], coord[2]);
    }

    for (size_t i = elBegin; i < elEnd; i++)
    {
        vtkNew<vtkHexahedron> hexa;
        vtkNew<vtkQuad> quad;
        vtkNew<vtkTetra> tetra;
        vtkNew<vtkTriangle> tri;
        for (size_t j = 0; j < vtkCellNumNodes; j++)
        {
            const auto pointIt = vtkPointIdByNodeTag.find(elNodeTag(i, j));
            if (pointIt == vtkPointIdByNodeTag.end())
                Fatal_Error("VTU writer node mapping error");

            if (vtkCellType == VTK_HEXAHEDRON)
                hexa->GetPointIds()->SetId(j, pointIt->second);
            else if (vtkCellType == VTK_TETRA)
                tetra->GetPointIds()->SetId(j, pointIt->second);
            else if (vtkCellType == VTK_QUAD)
                quad->GetPointIds()->SetId(j, pointIt->second);
            else
                tri->GetPointIds()->SetId(j, pointIt->second);
        }

        if (vtkCellType == VTK_HEXAHEDRON)
            cellArray->InsertNextCell(hexa);
        else if (vtkCellType == VTK_TETRA)
            cellArray->InsertNextCell(tetra);
        else if (vtkCellType == VTK_QUAD)
            cellArray->InsertNextCell(quad);
        else
            cellArray->InsertNextCell(tri);
    }

    pressure->SetName("Pressure [Pa]");
    density->SetName("Density [kg/m³]");
    velocity->SetName("Velocity [m/s]");
    velocity->SetNumberOfComponents(3);

    for (size_t el = elBegin; el < elEnd; ++el)
    {
        double p(0.0), rho(0.0), vx(0.0), vy(0.0), vz(0.0);
        for (size_t n = 0; n < getElNumNodes(); ++n)
        {
            size_t elN = el * getElNumNodes() + n;
            p += u[0][elN];
            rho += u[0][elN] / (config.c0 * config.c0);
            vx += u[1][elN];
            vy += u[2][elN];
            vz += u[3][elN];
        }
        double V[] = {vx / getElNumNodes(), vy / getElNumNodes(), vz / getElNumNodes()};
        pressure->InsertNextValue(p / getElNumNodes());
        density->InsertNextValue(rho / getElNumNodes());
        velocity->InsertNextTuple(V);
    }

    unstructuredGrid->SetPoints(points);

    unstructuredGrid->SetCells(vtkCellType, cellArray);

    unstructuredGrid->GetCellData()->AddArray(pressure);
    unstructuredGrid->GetCellData()->AddArray(density);
    unstructuredGrid->GetCellData()->AddArray(velocity);

    // Write file

    writer->SetFileName(filename.c_str());
    writer->SetInputData(unstructuredGrid);
    writer->Write();
}

void Mesh::writePVTUb(std::string filename)
{
    screen_display::write_string("Write PVTU: " + filename, BOLDRED);

    std::ofstream file(filename.c_str(), std::ios::trunc);
    const size_t slashPos = filename.find_last_of("/\\");
    const std::string baseName = (slashPos == std::string::npos) ? filename : filename.substr(slashPos + 1);
    const size_t dotPos = baseName.find_last_of('.');
    const std::string stem = (dotPos == std::string::npos) ? baseName : baseName.substr(0, dotPos);

    file << "<?xml version=\"1.0\"?>" << std::endl;
    file << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">" << std::endl;
    file << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl;
    file << "    <PCellData>" << std::endl;
    file << "      <PDataArray type=\"Float64\" Name=\"Pressure [Pa]\"/>" << std::endl;
    file << "      <PDataArray type=\"Float64\" Name=\"Density [kg/m³]\"/>" << std::endl;
    file << "      <PDataArray type=\"Float64\" Name=\"Velocity [m/s]\" NumberOfComponents=\"3\"/>" << std::endl;
    file << "    </PCellData>" << std::endl;
    file << "    <PPoints>" << std::endl;
    file << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>" << std::endl;
    file << "    </PPoints>" << std::endl;

    for (int rank = 0; rank < Parallel::size(); ++rank)
    {
        file << "    <Piece Source=\"" << stem << "_rank" << rank << ".vtu\"/>" << std::endl;
    }

    file << "  </PUnstructuredGrid>" << std::endl;
    file << "</VTKFile>" << std::endl;
    file.close();
}

void Mesh::writePVD(std::string filename)
{
    screen_display::write_string("Write PVD at " + filename, BOLDRED);
    std::ofstream file(filename.c_str(), std::ios::trunc);

    file << "<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << std::endl;
    file << "  <Collection>" << std::endl;
    for (double t = config.timeStart, step = 0, tDisplay = 0; t <= config.timeEnd;
         t += config.timeStep, tDisplay += config.timeStep, ++step)
    {
        if (tDisplay >= config.timeRate || step == 0)
        {
            tDisplay = 0;
            std::string vtu_filename = "results/result" + std::to_string((int)step);
            if (Parallel::size() > 1)
                file << "    <DataSet timestep=\"" << t << "\" part=\"0\" file=\"" << vtu_filename << ".pvtu\"/>" << std::endl;
            else
                file << "    <DataSet timestep=\"" << t << "\" part=\"0\" file=\"" << vtu_filename << ".vtu\"/>" << std::endl;
        }
    }
    file << "  </Collection>" << std::endl;
    file << "</VTKFile>" << std::endl;

    file.close();
}
