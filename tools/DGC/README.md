[python-img]: https://img.shields.io/github/languages/top/yueliu1999/Awesome-Deep-Graph-Clustering?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/yueliu1999/Awesome-Deep-Graph-Clustering?color=yellow
[stars-url]: https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/stargazers
[fork-img]: https://img.shields.io/github/forks/yueliu1999/Awesome-Deep-Graph-Clustering?color=lightblue&label=fork
[fork-url]: https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=yueliu1999.Awesome-Deep-Graph-Clustering
[adgc-url]: https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering

# ADGC: Awesome Deep Graph Clustering

ADGC is a collection of state-of-the-art (SOTA), novel deep graph clustering methods (papers, codes and datasets). Any other interesting papers and codes are welcome. Any problems, please contact yueliu19990731@163.com. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles: If you use our code or the processed datasets in this repository for your research, please cite 1-2 papers in the citation part. :heart:

[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

--------------

## What's Deep Graph Clustering?

Deep graph clustering, which aims to reveal the underlying graph structure and divide the nodes into different groups, has attracted intensive attention in recent years.

<div  align="center">    
    <img src="./assets/logo.png" width=60% />
</div>


## Important Survey Papers

| Year | Title                                                        |    Venue    |                            Paper                             | Code |
| ---- | ------------------------------------------------------------ | :---------: | :----------------------------------------------------------: | :--: |
| 2022 | **A Comprehensive Survey on Community Detectionwith Deep Learning** |    TNNLS    | [Link](https://arxiv.org/pdf/2105.12584.pdf?ref=https://githubhelp.com) |  -   |
| 2020 | **A Comprehensive Survey on Graph Neural Networks**          |    TNNLS    | [Link](https://ieeexplore.ieee.org/abstract/document/9046288) |  -   |
| 2020 | **Deep Learning for Community Detection: Progress, Challenges and Opportunities** |    IJCAI    |           [Link](https://arxiv.org/pdf/2005.08225)           |  -   |
| 2018 | **A survey of clustering with deep learning: From the perspective of network architecture** | IEEE Access | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8412085) |  -   |





## Papers

### Generative Deep Graph Clustering

| Year | Title                                                        |        Venue         |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2022 | **Graph embedding clustering: Graph attention auto-encoder with cluster-specificity distribution (GEC-CSD)** |          NN          | [Link](https://www.sciencedirect.com/science/article/pii/S0893608021002008) |          [Link](https://github.com/xdweixia/SGCMC)           |
| 2022 | **Exploring temporal community structure via network embedding (VGRGMM)** |         TCYB         | [Link](https://ieeexplore.ieee.org/abstract/document/9768181) |                              -                               |
| 2022 | **Hierarchical attention network for attributed community detection of joint representation (HiAN)** | Neural Comput. Appl. |      [Link](https://doi.org/10.1007/s00521-021-06723-y)      |                              -                               |
| 2022 | **Cluster-Aware Heterogeneous Information Network Embedding (VaCA-HINE)** |         WWW          |  [Link](https://dl.acm.org/doi/abs/10.1145/3488560.3498385)  |                              -                               |
| 2022 | **Self-Supervised Graph Convolutional Network for Multi-View Clustering (SGCMC)** |         TMM          | [Link](https://ieeexplore.ieee.org/abstract/document/9472979/) |          [Link](https://github.com/xdweixia/SGCMC)           |
| 2022 | **Efficient Graph Convolution for Joint Node Representation Learning and Clustering (GCC)** |         WSDM         |  [Link](https://dl.acm.org/doi/pdf/10.1145/3488560.3498533)  | [Link](https://github.com/chakib401/graph_convolutional_clustering) |
| 2022 | **ZINB-based Graph Embedding Autoencoder for Single-cell RNA-seq Interpretations (scTAG)** |         AAAI         | [Link](https://www.aaai.org/AAAI22Papers/AAAI-5060.YuZ.pdf)  |          [Link](https://github.com/Philyzh8/scTAG)           |
| 2022 | **Graph community infomax(GCI)**                             |         TKDD         |        [Link](https://dl.acm.org/doi/10.1145/3480244)        |                              -                               |
| 2022 | **Graph Clustering via Variational Graph Embedding (GC-VAE)** |          PR          | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005148) |                              -                               |
| 2022 | **Deep neighbor-aware embedding for node clustering in attributed graphs (DNENC)** |          PR          | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0031320321004118) |                              -                               |
| 2022 | **Collaborative Decision-Reinforced Self-Supervision for Attributed Graph Clustering (CDRS)** |        TNNLS         | [Link](https://ieeexplore.ieee.org/abstract/document/9777842) |       [Link](https://github.com/Jillian555/TNNLS_CDRS)       |
| 2022 | **Embedding Graph Auto-Encoder for Graph Clustering (EGAE)** |        TNNLS         |     [Link](https://ieeexplore.ieee.org/document/9741755)     |          [Link](https://github.com/hyzhang98/EGAE)           |
| 2021 | **Adaptive Hypergraph Auto-Encoder for Relational Data Clustering (AHGAE)** |         TKDE         | [Link](https://ieeexplore.ieee.org/iel7/69/4358933/09525190.pdf%3Fcasa_token%3DmbL8SLkmu8AAAAAA:mNPoE2n3BwaMZsYdRotHwa8Qs3uyzY53ZPVd0ixXutwqovM4vA7OSmsYWN3qXOAGW3CgH-LugHo&hl=en&sa=T&oi=ucasa&ct=ucasa&ei=_dvpYcTXCcCVy9YPgta4-AM&scisig=AAGBfm2V50SkaPV0K8x2F_mYsC15x028wA) |                              -                               |
| 2021 | **Deep Attention-guided Graph Clustering with Dual Self-supervision (DAGC)** |        arXiv         |           [Link](https://arxiv.org/abs/2111.05548)           |       [Link](https://github.com/ZhihaoPENG-CityU/DAGC)       |
| 2021 | **Attention-driven Graph Clustering Network (AGCN)**         |        ACM MM        |                           [Link]()                           |                           [Link]()                           |
| 2021 | **Rethinking Graph Auto-Encoder Models for Attributed Graph Clustering (R-GAE)** |        arXiv         |           [Link](https://arxiv.org/pdf/2107.08562)           |           [Link](https://github.com/nairouz/R-GAE)           |
| 2021 | **Deep Fusion Clustering Network (DFCN)**                    |         AAAI         | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/17198/17005) |             [Link](https://github.com/WxTu/DFCN)             |
| 2020 | **Collaborative Graph Convolutional Networks: Unsupervised Learning Meets Semi-Supervised Learning** |         AAAI         | [Link](https://ojs.aaai.org/index.php/AAAI/article/download/5843/5699) | [Link](https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE) |
| 2020 | **Deep multi-graph clustering via attentive cross-graph association (DMGC)** |         WSDM         |  [Link](https://dl.acm.org/doi/abs/10.1145/3336191.3371806)  |          [Link](https://github.com/flyingdoog/DMGC)          |
| 2020 | **Going Deep: Graph Convolutional Ladder-Shape Networks (GCLN)** |         AAAI         | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/5673/5529) |                              -                               |
| 2020 | **Multi-view attribute graph convolution networks for clustering (MAGCN)** |        IJCAI         |   [Link](https://www.ijcai.org/Proceedings/2020/0411.pdf)    |                              -                               |
| 2020 | **One2Multi Graph Autoencoder for Multi-view Graph Clustering (O2MAC)** |         WWW          |            [Link](http://shichuan.org/doc/83.pdf)            |     [Link](https://github.com/googlebaba/WWW2020-O2MAC)      |
| 2020 | **Structural Deep Clustering Network (SDCN/SDCN_Q)**         |         WWW          |           [Link](https://arxiv.org/pdf/2002.01633)           |           [Link](https://github.com/bdy9527/SDCN)            |
| 2020 | **Dirichlet Graph Variational Autoencoder (DGVAE)**          |       NeurIPS        | [Link](https://proceedings.neurips.cc/paper/2020/file/38a77aa456fc813af07bb428f2363c8d-Paper.pdf) |          [Link](https://github.com/xiyou3368/DGVAE)          |
| 2019 | **RWR-GAE: Random Walk Regularization for Graph Auto Encoders (RWR-GAE)** |        arXiv         |           [Link](https://arxiv.org/pdf/1908.04003)           |      [Link](https://github.com/MysteryVaibhav/RWR-GAE)       |
| 2019 | **Symmetric Graph Convolutional Autoencoder for Unsupervised Graph Representation Learning (GALA)** |         ICCV         | [Link](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Symmetric_Graph_Convolutional_Autoencoder_for_Unsupervised_Graph_Representation_Learning_ICCV_2019_paper.pdf) |       [Link](https://github.com/sseung0703/GALA_TF2.0)       |
| 2019 | **Attributed Graph Clustering: A Deep Attentional Embedding Approach (DAEGC)** |        IJCAI         |   [Link](https://www.ijcai.org/proceedings/2019/0509.pdf)    |         [Link](https://github.com/Tiger101010/DAEGC)         |
| 2019 | **Network-Specific Variational Auto-Encoder for Embedding in Attribute Networks (NetVAE)** |        IJCAI         |      [Link](https://www.ijcai.org/proceedings/2019/370)      |                              -                               |
| 2017 | **MGAE: Marginalized Graph Autoencoder for Graph Clustering (MGAE)** |         CIKM         | [Link](https://www.researchgate.net/profile/Shirui-Pan-3/publication/320882195_MGAE_Marginalized_Graph_Autoencoder_for_Graph_Clustering/links/5b76157b45851546c90a3d74/MGAE-Marginalized-Graph-Autoencoder-for-Graph-Clustering.pdf) |          [Link](https://github.com/GRAND-Lab/MGAE)           |
| 2016 | **Variational Graph Auto-Encoders (GAE)**                    |       NeurIPS        | [Link](https://ieeexplore.ieee.org/abstract/document/9046288) |             [Link](https://github.com/tkipf/gae)             |
| 2014 | **Learning Deep Representations for Graph Clustering (SAE)** |         AAAI         | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/8916/8775) | [Link](https://github.com/quinngroup/deep-representations-clustering) |







### Adversarial Deep Graph Clustering

| Year | Title                                                        | Venue |                           Paper                            |                      Code                      |
| ---- | ------------------------------------------------------------ | :---: | :--------------------------------------------------------: | :--------------------------------------------: |
| 2021 | **Wasserstein Adversarially Regularized Graph Autoencoder**  | arXiv |          [Link](https://arxiv.org/pdf/2111.04981)          | [Link](https://github.com/LeonResearch/WARGA)  |
| 2020 | **JANE: Jointly adversarial network embedding**              | IJCAI |  [Link](https://www.ijcai.org/Proceedings/2020/0192.pdf)   |                       -                        |
| 2020 | **Multi-class imbalanced graph convolutional network learning** | IJCAI |  [Link](https://www.ijcai.org/proceedings/2020/0398.pdf)   |                       -                        |
| 2019 | **Adversarial Graph Embedding for Ensemble Clustering (AGAE)** | IJCAI |     [Link](https://par.nsf.gov/servlets/purl/10113653)     |                       -                        |
| 2019 | **CommunityGAN: Community detection with generative adversarial nets** |  WWW  | [Link](https://dl.acm.org/doi/abs/10.1145/3308558.3313564) | [Link](https://github.com/SamJia/CommunityGAN) |
| 2019 | **ProGAN: Network embedding via proximity generative adversarial network** |  KDD  | [Link](https://dl.acm.org/doi/pdf/10.1145/3292500.3330866) |                       -                        |
| 2019 | **Learning Graph Embedding with Adversarial Training Methods (ARGA/ARVGA)** | TCYB  |          [Link](https://arxiv.org/pdf/1901.01250)          |   [Link](https://github.com/GRAND-Lab/ARGA)    |
| 2018 | **Adversarially Regularized Graph Autoencoder for Graph Embedding (ARGA/ARVGA)** | IJCAI |          [Link](https://arxiv.org/pdf/1802.04407)          |   [Link](https://github.com/GRAND-Lab/ARGA)    |



### Contrastive Deep Graph Clustering

| Year | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2022 | **NCAGC: A Neighborhood Contrast Framework for Attributed Graph Clustering (NCAGC)** |  arXiv  |           [Link](https://arxiv.org/abs/2206.07897)           | [Link](https://github.com/wangtong627/Dual-Contrastive-Attributed-Graph-Clustering-Network) |
| 2022 | **Simple Contrastive Graph Clustering (SCGC)**               |  arXiv  |           [Link](https://arxiv.org/abs/2205.07865)           |                              -                               |
| 2022 | **SCGC : Self-Supervised Contrastive Graph Clustering (SCGC)** |  arXiv  |           [Link](https://arxiv.org/pdf/2204.12656)           |           [Link](https://github.com/gayanku/SCGC)            |
| 2022 | **Improved Dual Correlation Reduction Network (IDCRN)**      |  arXiv  |           [Link](https://arxiv.org/abs/2202.12533)           |                              -                               |
| 2022 | **CGC: Contrastive Graph Clustering for Community Detection and Tracking (CGC)** |   WWW   |           [Link](https://arxiv.org/abs/2204.08504)           |                              -                               |
| 2022 | **Towards Unsupervised Deep Graph Structure Learning (SUBLIME)** |   WWW   |         [Link](https://arxiv.org/pdf/2201.06367.pdf)         |         [Link](https://github.com/GRAND-Lab/SUBLIME)         |
| 2022 | **Attributed Graph Clustering with Dual Redundancy Reduction (AGC-DRR)** |  IJCAI  |   [Link](https://www.ijcai.org/proceedings/2022/0418.pdf)    |                              -                               |
| 2022 | **Deep Graph Clustering via Dual Correlation Reduction (DCRN)** |  AAAI   | [Link](https://www.aaai.org/AAAI22Papers/AAAI-5928.LiuY.pdf) |          [Link](https://github.com/yueliu1999/DCRN)          |
| 2022 | **RepBin: Constraint-Based Graph Representation Learning for Metagenomic Binning (RepBin)** |  AAAI   | [Link](https://www.aaai.org/AAAI22Papers/AAAI-4979.XueH.pdf) |        [Link](https://github.com/xuehansheng/RepBin)         |
| 2022 | **Augmentation-Free Self-Supervised Learning on Graphs (AFGRL)** |  AAAI   |           [Link](https://arxiv.org/pdf/2112.02472)           |          [Link](https://github.com/Namkyeong/AFGRL)          |
| 2022 | **SAIL: Self-Augmented Graph Contrastive Learning (SAIL)**   |  AAAI   |           [Link](https://arxiv.org/abs/2009.00934)           |                              -                               |
| 2021 | **Graph Debiased Contrastive Learning with Joint Representation Clustering (GDCL)** |  IJCAI  |   [Link](https://www.ijcai.org/proceedings/2021/0473.pdf)    |           [Link](https://github.com/hzhao98/GDCL)            |
| 2021 | **Multilayer Graph Contrastive Clustering Network (MGCCN)**  |  arXiv  |                           [Link]()                           |                              -                               |
| 2021 | **Self-supervised Contrastive Attributed Graph Clustering (SCAGC)** |  arXiv  |           [Link](https://arxiv.org/abs/2110.08264)           |                              -                               |
| 2021 | **Multi-view Contrastive Graph Clustering (MCGC)**           | NeurIPS | [Link](https://papers.nips.cc/paper/2021/file/10c66082c124f8afe3df4886f5e516e0-Paper.pdf) |            [Link](https://github.com/panern/mcgc)            |
| 2021 | **Self-supervised heterogeneous graph neural network with co-contrastive learning** |   KDD   |    [Link](https://dl.acm.org/doi/10.1145/3447548.3467415)    |         [Link](https://github.com/liun-online/HeCo)          |
| 2020 | **Adaptive Graph Encoder for Attributed Graph Embedding (AGE)** | SIGKDD  |           [Link](https://arxiv.org/pdf/2007.01594)           |            [Link](https://github.com/thunlp/AGE)             |
| 2020 | **CommDGI: Community Detection Oriented Deep Graph Infomax (CommDGI)** |  CIKM   |  [Link](https://dl.acm.org/doi/abs/10.1145/3340531.3412042)  |          [Link](https://github.com/FDUDSDE/CommDGI)          |
| 2020 | **Contrastive Multi-View Representation Learning on Graphs (MVGRL)** |  ICML   | [Link](http://proceedings.mlr.press/v119/hassani20a/hassani20a.pdf) |        [Link](https://github.com/kavehhassani/mvgrl)         |





## Other Related Papers

### Deep Clustering

| Year | Title                                                        | **Venue** |                            Paper                             |                             Code                             |
| :--: | :----------------------------------------------------------- | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2022 | **Twin Contrastive Learning for Online Clustering**          |   IJCV    | [Link](http://pengxi.me/wp-content/uploads/2022/07/Twin-Contrastive-Learning-for-Online-Clustering.pdf) | [Link](https://github.com/Yunfan-Li/Twin-Contrastive-Learning) |
| 2022 | **Ada-nets: Face clustering via adaptive neighbor discovery in the structure space** |   ICLR    |           [Link](https://arxiv.org/pdf/2202.03800)           |         [Link](https://github.com/damo-cv/Ada-NETS)          |
| 2021 | **Adaptive Graph Auto-Encoder for General Data Clustering**  |   TPAMI   | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9606581) |         [Link](https://github.com/hyzhang98/AdaGAE)          |
| 2021 | **Contrastive Clustering**                                   |   AAAI    |         [Link](https://arxiv.org/pdf/2009.09687.pdf)         | [Link](https://github.com/Yunfan-Li/Contrastive-Clustering)  |
| 2017 | **Towards k-means-friendly spaces: Simultaneous deep learning and clustering** |   ICML    | [Link](http://proceedings.mlr.press/v70/yang17b/yang17b.pdf) |           [Link](https://github.com/boyangumn/DCN)           |
| 2017 | **Improved Deep Embedded Clustering with Local Structure Preservation** |   IJCAI   | [Link](https://www.researchgate.net/profile/Xifeng-Guo/publication/317095655_Improved_Deep_Embedded_Clustering_with_Local_Structure_Preservation/links/59263224458515e3d4537edc/Improved-Deep-Embedded-Clustering-with-Local-Structure-Preservation.pdf) |          [Link](https://github.com/XifengGuo/IDEC)           |
| 2016 | **Unsupervised Deep Embedding for Clustering Analysis**      |   ICML    |     [Link](http://proceedings.mlr.press/v48/xieb16.pdf)      |           [Link](https://github.com/piiswrong/dec)           |

### Other Related Methods

| Year | Title                                                        | **Venue** |                            Paper                             |                             Code                             |
| :--: | :----------------------------------------------------------- | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2022 | **GRACE: A General Graph Convolution Framework for Attributed Graph Clustering** |   TKDD    |      [Link](https://dl.acm.org/doi/pdf/10.1145/3544977)      |                              -                               |
| 2022 | **Fine-grained Attributed Graph Clustering**                 |    SDM    | [Link](https://epubs.siam.org/doi/epdf/10.1137/1.9781611977172.42) |            [Link](https://github.com/sckangz/FGC)            |
| 2022 | **Multi-view graph embedding clustering network: Joint self-supervision and block diagonal representation** |    NN     | [Link](https://www.sciencedirect.com/science/article/pii/S089360802100397X?via%3Dihub) |       [Link](https://github.com/xdweixia/NN-2022-MVGC)       |
| 2022 | **SAGES: Scalable Attributed Graph Embedding with Sampling for Unsupervised Learning** |   TKDE    | [Link](https://ieeexplore.ieee.org/abstract/document/9705119) |                              -                               |
| 2022 | **Automated Self-Supervised Learning For Graphs**            |   ICLR    |     [Link](https://openreview.net/forum?id=rFbR4Fv-D6-)      |       [Link](https://github.com/ChandlerBang/AutoSSL)        |
| 2022 | **Stationary diffusion state neural estimation for multi-view clustering** |   AAAI    |           [Link](https://arxiv.org/abs/2112.01334)           |           [Link](https://github.com/kunzhan/SDSNE)           |
| 2021 | **Simple Spectral Graph Convolution**                        |   ICLR    |      [Link](https://openreview.net/pdf?id=CYO5T-YjWZV)       |         [Link](https://github.com/allenhaozhu/SSGC)          |
| 2021 | **Spectral embedding network for attributed graph clustering (SENet)** |    NN     | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0893608021002227) |                              -                               |
| 2021 | **Smoothness Sensor: Adaptive Smoothness Transition Graph Convolutions for Attributed Graph Clustering** |   TCYB    | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9514513) |           [Link](https://github.com/aI-area/NASGC)           |
| 2021 | **Multi-view Attributed Graph Clustering**                   |   TKDE    | [Link](https://www.researchgate.net/profile/Zhao-Kang-6/publication/353747180_Multi-view_Attributed_Graph_Clustering/links/612059cd0c2bfa282a5cd55e/Multi-view-Attributed-Graph-Clustering.pdf) |           [Link](https://github.com/sckangz/MAGC)            |
| 2021 | **High-order Deep Multiplex Infomax**                        |    WWW    |           [Link](https://arxiv.org/abs/2102.07810)           |          [Link](https://github.com/baoyujing/HDMI)           |
| 2021 | **Graph InfoClust: Maximizing Coarse-Grain Mutual Information in Graphs** |   PAKDD   | [Link](https://link.springer.com/chapter/10.1007%2F978-3-030-75762-5_43) |    [Link](https://github.com/cmavro/Graph-InfoClust-GIC)     |
| 2021 | **Graph Filter-based Multi-view Attributed Graph Clustering** |   IJCAI   |   [Link](https://www.ijcai.org/proceedings/2021/0375.pdf)    |           [Link](https://github.com/sckangz/MvAGC)           |
| 2021 | **Graph-MVP: Multi-View Prototypical Contrastive Learning for Multiplex Graphs** |   arXiv   |           [Link](https://arxiv.org/abs/2109.03560)           |         [Link](https://github.com/chao1224/GraphMVP)         |
| 2021 | **Contrastive Laplacian Eigenmaps**                          |  NeurIPS  | [Link](https://proceedings.neurips.cc/paper/2021/file/2d1b2a5ff364606ff041650887723470-Paper.pdf) |         [Link](https://github.com/allenhaozhu/COLES)         |
| 2020 | **Cluster-Aware Graph Neural Networks for Unsupervised Graph Representation Learning** |   arXiv   |           [Link](https://arxiv.org/abs/2009.01674)           |                                                              |
| 2020 | **Distribution-induced Bidirectional GAN for Graph Representation Learning** |   CVPR    |           [Link](https://arxiv.org/pdf/1912.01899)           |           [Link](https://github.com/SsGood/DBGAN)            |
| 2020 | **Adaptive Graph Converlutional Network with Attention Graph Clustering for Co saliency Detection** |   CVPR    | [Link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Adaptive_Graph_Convolutional_Network_With_Attention_Graph_Clustering_for_Co-Saliency_CVPR_2020_paper.pdf) |      [Link](https://github.com/ltp1995/GCAGC-CVPR2020)       |
| 2020 | **Spectral Clustering with Graph Neural Networks for Graph Pooling** |   ICML    | [Link](http://proceedings.mlr.press/v119/bianchi20a/bianchi20a.pdf) | [Link](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling) |
| 2020 | **MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding** |    WWW    |           [Link](https://arxiv.org/abs/2002.01680)           |          [Link](https://github.com/cynricfu/MAGNN)           |
| 2020 | **Unsupervised Attributed Multiplex Network Embedding**      |   AAAI    |           [Link](https://arxiv.org/abs/1911.06750)           |           [Link](https://github.com/pcy1302/DMGI)            |
| 2020 | **Cross-Graph: Robust and Unsupervised Embedding for Attributed Graphs with Corrupted Structure** |   ICDM    |     [Link](https://ieeexplore.ieee.org/document/9338269)     |      [Link](https://github.com/FakeTibbers/Cross-Graph)      |
| 2020 | **Multi-View Attribute Graph Convolution Networks for Clustering** |   IJCAI   |   [Link](https://www.ijcai.org/proceedings/2020/0411.pdf)    |           [Link](https://github.com/IMKBLE/MAGCN)            |
| 2020 | **Attributed Graph Clustering via Deep Adaptive Graph Maximization** |   ICCKE   | [Link](https://ieeexplore-ieee-org-s.nudtproxy.yitlink.com/stamp/stamp.jsp?tp=&arnumber=9303694) |                              -                               |
| 2019 | **Heterogeneous Graph Attention Network**                    |    WWW    |         [Link](https://arxiv.org/pdf/1903.07293.pdf)         |            [Link](https://github.com/Jhy1993/HAN)            |
| 2019 | **Multi-view Consensus Graph Clustering**                    |    TIP    | [Link](https://ieeexplore.ieee.org/abstract/document/8501973) |           [Link](https://github.com/kunzhan/MCGC)            |
| 2019 | **Attributed Graph Clustering via Adaptive Graph Convolution (AGC)** |   IJCAI   |   [Link](https://www.ijcai.org/Proceedings/2019/0601.pdf)    |      [Link](https://github.com/karenlatong/AGC-master)       |



## Benchmark Datasets

We divide the datasets into two categories, i.e. graph datasets and non-graph datasets. Graph datasets are some graphs in real-world, such as citation networks, social networks and so on. Non-graph datasets are NOT graph type. However, if necessary, we could construct "adjacency matrices"  by K-Nearest Neighbors (KNN) algorithm.



#### Quick Start

- Step1: Download all datasets from \[[Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK?usp=sharing) | [Nutstore](https://www.jianguoyun.com/p/DfzK1pwQwdaSChjI2aME)]. Optionally, download some of them from URLs in the tables (Google Drive)
- Step2: Unzip them to **./dataset/**
- Step3: Change the type and the name of the dataset in **main.py**
- Step4: Run the **main.py**



#### Code

- **utils.py**
  1. **load_graph_data**: load graph datasets 
  2. **load_data**: load non-graph datasets
  3. **normalize_adj**: normalize the adjacency matrix
  4. **diffusion_adj**: calculate the graph diffusion
  5. **construct_graph**: construct the knn graph for non-graph datasets
  6. **numpy_to_torch**: convert numpy to torch
  7. **torch_to_numpy**: convert torch to numpy
- **clustering.py**
  1. **setup_seed**:  fix the random seed
  2. **evaluation**: evaluate the performance of clustering
  3. **k_means**: K-means algorithm
- **visualization.py**
  1. **t_sne**: t-SNE algorithm
  2. **similarity_plot**: visualize cosine similarity matrix of the embedding or feature



#### Datasets Details

About the introduction of each dataset, please check [here](./dataset/README.md)

1. Graph Datasets

   | Dataset  | # Samples | # Dimension | # Edges | # Classes |                             URL                              |
   | :------: | :-------: | :---------: | :-----: | :-------: | :----------------------------------------------------------: |
   |   CORA   |   2708    |    1433     |  5278   |     7     | [cora.zip](https://drive.google.com/file/d/1_LesghFTQ02vKOBUfDP8fmDF1JP3MPrJ/view?usp=sharing) |
   | CITESEER |   3327    |    3703     |  4552   |     6     | [citeseer.zip](https://drive.google.com/file/d/1dEsxq5z5dc35tS3E46pg6pc2LUMlF6jF/view?usp=sharing) |
   |   CITE   |   3327    |    3703     |  4552   |     6     | [cite.zip](https://drive.google.com/file/d/1dEsxq5z5dc35tS3E46pg6pc2LUMlF6jF/view?usp=sharing) |
   |  PUBMED  |   19717   |     500     |  44324  |     3     | [pubmed.zip](https://drive.google.com/file/d/1tdr20dvvjZ9tBHXj8xl6wjO9mQzD0rzA/view?usp=sharing) |
   |   DBLP   |   4057    |     334     |  3528   |     4     | [dblp.zip](https://drive.google.com/file/d/1XWWMIDyvCQ4VJFnAmXS848ksN9MFm5ys/view?usp=sharing) |
   |   ACM    |   3025    |    1870     |  13128  |     3     | [acm.zip](https://drive.google.com/file/d/19j7zmQ-AMgzTX7yZoKzUK5wVxQwO5alx/view?usp=sharing) |
   |   AMAP   |   7650    |     745     | 119081  |     8     | [amap.zip](https://drive.google.com/file/d/1qqLWPnBOPkFktHfGMrY9nu8hioyVZV31/view?usp=sharing) |
   |   AMAC   |   13752   |     767     | 245861  |    10     | [amac.zip](https://drive.google.com/file/d/1DJhSOYWXzlRDSTvaC27bSmacTbGq6Ink/view?usp=sharing) |
   | CORAFULL |   19793   |    8710     |  63421  |    70     | [corafull.zip](https://drive.google.com/file/d/1XLqs084J3xgWW9jtbBXJOmmY84goT1CE/view?usp=sharing) |
   |   WIKI   |   2405    |    4973     |  8261   |    17     | [wiki.zip](https://drive.google.com/file/d/1vxupFQaEvw933yUuWzzgQXxIMQ_46dva/view?usp=sharing) |
   |   COCS   |   18333   |    6805     |  81894  |    15     | [cocs.zip](https://drive.google.com/file/d/186twSfkDNmqh9L618iCeWq4DA7Lnpte0/view?usp=sharing) |
   | CORNELL  |    183    |    1703     |   149   |     5     | [cornell.zip](https://drive.google.com/file/d/1EjpHP26Oh0_qHl13vOfEzc4ZyzkGrR-M/view?usp=sharing) |
   |  TEXAS   |    183    |    1703     |   162   |     5     | [texas.zip](https://drive.google.com/file/d/1kpz6b9-OsEU1RsAyxWWeUgzhdd3-koI2/view?usp=sharing) |
   |   WISC   |    251    |    1703     |   257   |     5     | [wisc.zip](https://drive.google.com/file/d/1I8v1H1IthEiWd4IoV-wXNF6g1Wtg_sVC/view?usp=sharing) |
   |   FILM   |   7600    |     932     |  15009  |     5     | [film.zip](https://drive.google.com/file/d/1s5K9Gb235-gO-IwevJLKAts7jExnnmrC/view?usp=sharing) |
   |   BAT    |    131    |     81      |  1038   |     4     | [bat.zip](https://drive.google.com/file/d/1hRPtdFo9CzcxlFb84NWXg-HmViZnqshu/view?usp=sharing) |
   |   EAT    |    399    |     203     |  5994   |     4     | [eat.zip](https://drive.google.com/file/d/1iE0AFKs1V5-nMk2XhV-TnfmPhvh0L9uo/view?usp=sharing) |
   |   UAT    |   1190    |     239     |  13599  |     4     | [uat.zip](https://drive.google.com/file/d/1RUTHp54dVPB-VGPsEk8tV32DsSU0l-n_/view?usp=sharing) |
   

**Edges**: Here, we just count the number of undirected edges.

2. Non-graph Datasets

   | Dataset | Samples | Dimension |  Type  | Classes |                             URL                              |
   | :-----: | :-----: | :-------: | :----: | :-----: | :----------------------------------------------------------: |
   |  USPS   |  9298   |    256    | Image  |   10    | [usps.zip](https://drive.google.com/file/d/19oBkSeIluW3A5kcV7W0UM1Bt6V9Q62e-/view?usp=sharing) |
   |  HHAR   |  10299  |    561    | Record |    6    | [hhar.zip](https://drive.google.com/file/d/126OFuNhf2u-g9Tr0wukk0T8uM1cuPzy2/view?usp=sharing) |
   |  REUT   |  10000  |   2000    |  Text  |    4    | [reut.zip](https://drive.google.com/file/d/12MpPWyN87bu-AQYTyjdEcofy1mgjgzi9/view?usp=sharing) |



## Citation

```
@inproceedings{DCRN,
  title={Deep Graph Clustering via Dual Correlation Reduction},
  author={Liu, Yue and Tu, Wenxuan and Zhou, Sihang and Liu, Xinwang and Song, Linxuan and Yang, Xihong and Zhu, En},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={7},
  pages={7603-7611},
  year={2022}
}
@article{mrabah2021rethinking,
  title={Rethinking Graph Auto-Encoder Models for Attributed Graph Clustering},
  author={Mrabah, Nairouz and Bouguessa, Mohamed and Touati, Mohamed Fawzi and Ksantini, Riadh},
  journal={arXiv preprint arXiv:2107.08562},
  year={2021}
}
```



## Other Related Awesome Repository

[awesome-awesome-machine-learning](https://github.com/ZhiningLiu1998/awesome-awesome-machine-learning)

[Awesome Incomplete Graph Representation Learning](https://github.com/WxTu/Awesome-Incomplete-Graph-Representation-Learning)

[awesome-self-supervised-gnn](https://github.com/ChandlerBang/awesome-self-supervised-gnn)

