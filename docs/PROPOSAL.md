**A HYBRID LEARNING FRAMEWORK FOR**   
**AUTOMATED SAT SOLVER SELECTION**

By

**Zishaan Ahmed**

**A PROPOSAL**

SUBMITTED TO

**Universiti Tunku Abdul Rahman**

in partial fulfillment of the requirements

for the degree of

**BACHELOR OF COMPUTER SCIENCE (HONOURS)**

**Faculty of Information and Communication Technology**

**(Kampar Campus)**

**JUNE 2026**  
**ABSTRACT**

Boolean Satisfiability (SAT) solving underlies a wide range of practical applications, including hardware verification, software testing, automated planning, and combinatorial design. Decades of engineering effort have produced highly optimised Conflict-Driven Clause Learning (CDCL) solvers, yet empirical studies consistently show that no single solver dominates across all problem instances; different solvers exhibit strongly complementary performance depending on the structural properties of the formula being solved. This observation motivates per-instance algorithm selection rather than reliance on a single fixed solver. Existing portfolio-based selectors, most notably SATzilla, rely on manually engineered structural and probing features that are informative but costly to compute and may not capture higher-order relational structure present in a formula's clause-variable graph. More recent graph neural network (GNN) approaches learn structural embeddings directly from the instance graph, but typically discard the substantial domain expertise encoded in handcrafted feature sets and can require considerable training data to generalise reliably to unseen instance families. This project proposes a hybrid learning framework that fuses handcrafted, SATzilla-style instance features with GNN-derived structural embeddings within a unified meta-classifier, combining the sample efficiency and interpretability of feature-based selection with the representational flexibility of graph learning. Each SAT instance will be represented as a literal-clause graph from which both a handcrafted feature vector and a learned graph embedding are extracted; at least two fusion strategies for combining the two representations will be implemented and compared. The framework will be trained and evaluated on benchmark instances drawn from recent SAT Competition benchmark suites, using a portfolio of established open-source CDCL solvers, and assessed against the single-best-solver baseline and the virtual-best-solver oracle using standard algorithm-selection metrics such as PAR10. It is anticipated that the hybrid approach will achieve higher selection accuracy and lower penalised average runtime than either a purely feature-based or a purely graph-based selector evaluated in isolation.

**Area of Study:** Machine Learning for Systems, Combinatorial Optimisation, Theoretical Computer Science

**Keywords:** SAT Solving, Algorithm Selection, Graph Neural Networks

**TABLE OF CONTENTS**

[**SECTION I	1**](#heading=)

[1.1 Introduction	1](#heading=)

[1.2 Problem Statement	3](#heading=)

[1.3 Motivation	5](#heading=)

[**SECTION II	6**](#heading=)

[2.1 The Algorithm Selection Problem and Empirical Hardness Models	6](#heading=)

[2.1.1 SATzilla and Feature-Based Portfolio Selection for SAT	7](#heading=)

[2.1.2 Algorithm Selection Benchmarking Infrastructure: ASlib and AutoFolio	8](#heading=)

[2.2 Early Machine Learning and Reinforcement Learning Approaches	10](#heading=)

[2.3 Deep Learning and Graph Neural Networks for SAT	12](#heading=)

[2.3.1 End-to-End Neural SAT Solvers	12](#heading=)

[2.3.2 Neural-Guided CDCL Solvers	12](#heading=)

[2.3.3 GNN-Based Algorithm Selection	13](#heading=)

[2.4 Complementary Feature-Free Approaches	17](#heading=)

[2.5 Limitations of Existing Studies	18](#heading=)

[2.6 Proposed Solution	21](#heading=)

[**SECTION III	23**](#heading=)

[3.1 Project Scope	23](#heading=)

[3.2 Project Objectives	24](#heading=)

[3.3 Contributions	25](#heading=)

[**SECTION IV	26**](#heading=)

[4.1 Overall System Architecture	26](#heading=)

[4.2 SAT Instance Representation and Handcrafted Feature Extraction	28](#heading=)

[4.3 Graph Neural Network Embedding Module	30](#heading=)

[4.4 Hybrid Fusion and Meta-Classifier	32](#4.4-hybrid-fusion-and-meta-classifier)

[4.5 Solver Portfolio and Benchmark Datasets	34](#heading=)

[4.6 Comparative Performance of Candidate SAT Solvers	35](#heading=)

[4.7 Experimental Design and Evaluation Metrics	38](#heading=)

[4.8 Development Tools and Environment	40](#heading=)

[4.9 Project Schedule	42](#heading=)

[**REFERENCES	43**](#heading=)  
**LIST OF FIGURES**

**Figure 2.1 \-** Taxonomy of per-instance algorithm-selection approaches for SAT, organised by instance representation. Bracketed numbers refer to the reference list.	**7**

**Figure 2.2 \-** General pipeline of feature-based per-instance algorithm selection, following the architecture popularised by SATzilla \[5\]	**8**

**Figure 2.3 \-** Comparison of instance-representation families across the criteria that matter for per-instance solver selection, synthesised from the studies reviewed in Section II.	18

**Figure 2.4 \-** Development of learning-based algorithm selection for SAT, from Rice’s formalisation of the problem to the hybrid approaches of 2024, showing the gap addressed by this project.	19

**Figure 2.5 \-** Positioning of the proposed framework relative to existing approaches, along the two axes that distinguish them: reliance on handcrafted domain knowledge and reliance on learned structural representation.	20

**Figure 4.1 \-** Proposed hybrid architecture combining handcrafted features and a graph neural network embedding for SAT solver selection.	25

**Figure 4.2 \-** Main-track podiums of SAT Competition 2024 for satisfiable, unsatisfiable and all instances \[22\]. Solvers change position between categories, demonstrating per-instance complementarity within a single controlled evaluation.	35

**Figure 4.3 \-** Experimental pipeline, from benchmark instances and solver portfolio through runtime-data collection and label generation to selector training and evaluation.	37

**Figure 4.4 \-** Schematic of the evaluation. (a) Solved-instance profile against runtime cutoff; (b) the SBS–VBS interval and the portion of it closed by each selector. Axes carry no measured data; the figure defines the quantities to be reported, not anticipated results.	38

**Figure 4.5 \-** Tentative project schedule, showing the sub-objectives of Section 3.2 against a fourteen-week trimester. Milestones: M1 (week 6\) runtime matrix complete; M2 (week 11\) all selectors trained; M3 (week 14\) submission.	41  
**LIST OF TABLES**

**Table 4.1: Candidate solver portfolio for the proposed hybrid selection framework	33**

**Table 4.2: Reference competition results for the candidate solver portfolio	34**  
**LIST OF ABBREVIATIONS**

**ASlib**	Algorithm Selection Library

**CDCL**	Conflict-Driven Clause Learning

**CNF**	Conjunctive Normal Form

**DPLL**	Davis–Putnam–Logemann–Loveland (procedure)

**DIMACS**	Standard input format for CNF instances

**GNN**	Graph Neural Network

**MLP**	Multilayer Perceptron

**PAR2**	Penalised Average Runtime (penalty factor 2\)

**ML4Sys**	Machine Learning for Systems

**PAR10**	Penalised Average Runtime (penalty factor 10\)

**QBF**	Quantified Boolean Formula

**SAT**	Boolean Satisfiability (Problem)

**SBS**	Single Best Solver

**VBS**	Virtual Best Solver

**VSIDS**	Variable State Independent Decaying Sum

# **SECTION I**

**PROJECT BACKGROUND**

## **1.1 Introduction**

The Boolean Satisfiability problem, commonly abbreviated as SAT, asks whether there exists an assignment of truth values to the variables of a propositional formula, usually expressed in Conjunctive Normal Form (CNF), such that the formula evaluates to true. SAT was the first problem proven to be NP-complete, and it remains a cornerstone of theoretical computer science \[1\]. Despite this worst-case intractability, modern SAT solvers built around the Conflict-Driven Clause Learning (CDCL) paradigm routinely solve industrial instances containing millions of variables and clauses \[2\]. This apparent contradiction between theoretical hardness and practical tractability has made SAT solving a critical enabling technology for hardware and software verification, automated planning, cryptanalysis, combinatorial design, and a growing number of applications in which real-world problems are encoded as satisfiability queries.

Decades of engineering effort have produced a diverse ecosystem of CDCL solvers, including MiniSat, Glucose, CaDiCaL, CryptoMiniSat, and Kissat, each incorporating different heuristics for variable branching, clause deletion, restart policies, and preprocessing \[2\]. A well-documented empirical observation in the SAT community is that no single solver is uniformly superior: solvers exhibit strongly complementary performance, with one solver dramatically outperforming another on a given family of instances while being outperformed in turn on a different family. This complementarity has motivated the study of the algorithm selection problem, formally introduced by Rice, which asks how to construct a mapping from problem instances to the algorithm expected to perform best on that instance \[3\].

This project situates itself at the intersection of machine learning and systems software, an area often referred to as ML4Sys, in which learned models are used to improve the operation of traditional systems components rather than to replace them outright. Specifically, this project proposes a hybrid learning framework for automated SAT solver selection that combines two complementary families of instance representation: handcrafted structural and probing features in the tradition of SATzilla, and learned structural embeddings produced by a graph neural network (GNN) operating on the literal-clause graph of a SAT instance. The remainder of this proposal reviews the relevant literature on algorithm selection and learning-based approaches to SAT solving, defines the scope and objectives of the proposed project, and describes the methods and technologies that will be used to design, implement, and evaluate the proposed framework.

## 

## **1.2 Problem Statement**

Although individual CDCL solvers have been engineered to a high degree of sophistication, their performance remains highly dependent on the characteristics of the SAT instance being solved. Different solvers can perform substantially differently on the same instance because they employ different branching heuristics, clause-learning strategies, restart policies, preprocessing techniques, and other search mechanisms. Consequently, relying on a single fixed solver for all instances may fail to exploit the complementarity that exists within a solver portfolio. The potential benefit of solver selection can be observed through the Virtual Best Solver (VBS), an oracle that selects the fastest solver for each individual instance and therefore provides an upper-bound reference for portfolio performance.

Automated algorithm-selection approaches have been developed to address this problem, with SATzilla representing one of the most established feature-based approaches \[5\]. SATzilla-style methods use handcrafted structural and probing features to characterise each SAT instance before predicting which solver is likely to perform best. These features provide useful and interpretable information, but some feature families, particularly probing-based features that require short search procedures, introduce additional computational overhead. Furthermore, a fixed feature vocabulary may not capture every structural relationship within a SAT instance, especially when interactions between variables and clauses are difficult to express through manually designed statistics. In this project, this limitation is addressed by using a computationally practical subset of established handcrafted features rather than attempting to reproduce the complete feature-extraction pipeline.

Graph neural network approaches provide an alternative because SAT instances have an inherent relational structure that can be represented as a literal-clause or variable-clause graph. A GNN can learn an instance representation directly from this graph and capture structural relationships through message passing, without requiring every potentially useful structural property to be specified manually. Existing studies have demonstrated that graph-based representations can provide useful information for SAT solver selection, while also indicating that their effectiveness depends on the available training data and benchmark distribution. Related hybrid approaches, such as GraSS \[15\], further suggest that combining graph-based learning with domain knowledge can be beneficial. However, the existing literature provides limited evidence on the relative contribution of handcrafted and graph-based representations when they are evaluated as explicitly separate components under the same experimental conditions and on openly available benchmark data.

This project therefore addresses the following problem: **given a portfolio of complete SAT solvers and a collection of SAT instances, how can handcrafted instance features and GNN-derived structural embeddings be combined within a unified learning framework to improve per-instance solver selection, and what additional predictive value does each representation provide when evaluated individually and in combination?**

## 

## **1.3 Motivation**

The motivation for this project is threefold. From a practical standpoint, SAT solving is embedded in verification and testing pipelines where wasted compute time directly translates into slower development cycles, higher energy consumption, and higher infrastructure cost; even modest improvements in solver selection accuracy can produce meaningful reductions in aggregate runtime when applied across the thousands of SAT queries generated in a typical verification workflow. From a research standpoint, the specific combination of handcrafted algorithm-selection features with graph-based deep learning for SAT solver selection is a young and still-developing research direction, with the most closely related published work having appeared only within the past two to three years; there remains room to systematically study fusion strategies and to quantify the marginal contribution of each representation, rather than treating the hybrid design as a single fixed architectural choice. From an academic standpoint, the project offers an opportunity to engage directly with core themes of the ML4Sys area, applying supervised and graph-based learning techniques to a well-defined, measurable systems problem with an established evaluation methodology and publicly available benchmark infrastructure.

# **SECTION II**

**LITERATURE REVIEW**

## **2.1 The Algorithm Selection Problem and Empirical Hardness Models**

The theoretical foundation for automated solver selection was laid by Rice, who formalised the algorithm selection problem in terms of a problem space P, an algorithm space A, and a performance measure that maps each problem-algorithm pair to an observed performance value \[3\]. Given this formulation, the task is to construct a selection mapping from problem instances to algorithms that maximises overall performance across the problem space, rather than committing to a single algorithm that performs well only on average. *Kerschke et al.* provide a comprehensive survey of the substantial body of work that has grown out of Rice's original formulation, distinguishing per-instance selection, in which a different algorithm may be chosen for every individual problem instance, from per-class selection, in which a single algorithm is chosen for an entire category of instances \[4\]. Per-instance selection is the more general and, in practice, the more powerful setting, and it is the setting adopted in this project.

A common realisation of per-instance algorithm selection is the empirical hardness model, in which cheaply computable features of a problem instance are used to predict the runtime, or some other performance measure, of each candidate algorithm on that instance; the algorithm with the best predicted performance is then selected. This regression-based approach, together with classification-based variants that directly predict the best-performing algorithm without an intermediate runtime estimate, forms the basis of most practical algorithm selection systems, including those developed specifically for SAT.

Figure 2.1 summarises how the approaches reviewed in this section relate to one another. The literature divides into four families that differ chiefly in how a SAT instance is represented before a selection decision is made, and it is the position of the present project within this structure, rather than any single technique it uses, that defines its contribution.

![][image1]

**Figure 2.1: Taxonomy of per-instance algorithm-selection approaches for SAT, organised by instance representation. Bracketed numbers refer to the reference list.**

### **2.1.1 SATzilla and Feature-Based Portfolio Selection for SAT**

The best-known instantiation of feature-based algorithm selection for SAT is SATzilla, developed by Xu, Hutter, Hoos, and Leyton-Brown \[5\]. SATzilla constructs a per-instance portfolio selector by extracting a broad set of structural, syntactic, and probing features from each CNF instance, including problem-size statistics, variable-clause graph statistics, measures of proximity to a Horn formula, features derived from a linear programming relaxation of the instance, and features obtained by running short, bounded probes of DPLL search and stochastic local search. These features build on the earlier work of Nudelman, Leyton-Brown, Hoos, Devkar, and Shoham, who demonstrated that features going beyond the simple clause-to-variable ratio are informative predictors of instance hardness for random SAT \[6\]. Using these features, SATzilla trains empirical hardness models, originally regression models and later classification and hierarchical mixture-of-experts models, to predict which solver in its portfolio is expected to perform best on a given instance. SATzilla achieved substantial success in the SAT Competition, winning multiple medals across several editions of the competition, and its general design, extract features offline, train a selection model, and select a solver online, has become the reference architecture for feature-based algorithm selection in SAT and beyond. Figure 2.2 illustrates this general pipeline.

![][image2]

**Figure 2.2: General pipeline of feature-based per-instance algorithm selection, following the architecture popularised by SATzilla \[5\].**

Although highly effective, the SATzilla feature set is not without cost. Several of the probing features require running a solver or a local search procedure for a bounded amount of time before the main selection decision can be made, adding non-trivial computational overhead relative to the eventual solving time on easy instances. In addition, because the feature set was designed and validated primarily on the instance distributions available at the time of its development, its continued relevance to more recent, more diverse competition benchmarks is an empirical question rather than a foregone conclusion; this question is addressed directly by Shavit and Hoos, discussed in Section 2.3.3 below.

### **2.1.2 Algorithm Selection Benchmarking Infrastructure: ASlib and AutoFolio**

As algorithm selection research matured beyond SAT into other combinatorial domains, the community recognised the need for a standardised way to represent and compare algorithm selection scenarios. *Bischl et al.* introduced the Algorithm Selection Library (ASlib), a standard scenario format together with a growing repository of algorithm selection datasets drawn from SAT, constraint satisfaction, answer set programming, and other domains, along with baseline results from straightforward selection approaches \[7\]. ASlib has since become a common evaluation substrate for new algorithm selection methods, allowing fair comparison across studies without each study having to recollect runtime data from scratch.

Building on this infrastructure, Lindauer, Hoos, Hutter, and Schaub proposed AutoFolio, which observed that the algorithm selection pipeline itself, that is, the choice of machine learning model, its hyperparameters, and auxiliary components such as pre-solving schedules, is itself a design space that can be automatically configured rather than fixed by hand \[8\]. Using automated algorithm configuration techniques, AutoFolio was shown to match or exceed the performance of several hand-tuned, state-of-the-art selectors across a broad range of ASlib scenarios. This line of work underscores an important methodological point that informs the present project: the choice of downstream selection or fusion model is itself a design decision that benefits from systematic comparison rather than a single default choice, which motivates the explicit comparison of multiple fusion strategies described in Section IV.

## **2.2 Early Machine Learning and Reinforcement Learning Approaches**

Learning-based decision-making in SAT solving predates the modern deep learning era. Lagoudakis and Littman proposed an early reinforcement learning approach that operates at a finer granularity than portfolio-level solver selection: rather than choosing a solver once for an entire instance, their method learns a value function that selects among several candidate branching rules at each decision point within a DPLL search procedure \[9\]. This work demonstrated that machine learning could be used to adapt the search strategy to the characteristics of a problem instance, establishing an important precedent for applying learning to SAT solving.

However, this approach differs fundamentally from the objective of the present project. Reinforcement learning at the branching-rule level requires the learning model to make repeated decisions throughout the execution of the SAT solver. As a result, the learning component becomes part of the solver's search loop and introduces computational overhead during solving, rather than making a single selection before search begins. The learned policy must therefore be sufficiently efficient to operate repeatedly without negating the performance gained from improved search decisions.

The present project instead focuses on offline, per-instance algorithm selection, where the aim is to determine which complete SAT solver should be executed before solving begins. This formulation is motivated by the observed complementarity of modern CDCL solvers: different solvers may perform very differently on the same instance because their underlying heuristics and search strategies are designed differently. Rather than modifying these mature solvers or introducing additional decision-making inside their search procedures, the proposed framework uses machine learning to analyse the structural characteristics of an instance once and select the solver expected to perform best.

This approach has several advantages for the scope of the project. First, it keeps the candidate CDCL solvers unchanged, allowing their established implementations and optimised search procedures to be used directly. Second, the computational cost of the learning model is concentrated in the pre-solving selection stage rather than being incurred repeatedly during search. Third, it allows the proposed GNN-based representation to capture structural information from the complete SAT instance before solver execution and combine this information with handcrafted features through the proposed hybrid framework. Finally, the offline formulation provides a clearer experimental setting for measuring the contribution of different instance representations by comparing feature-only, graph-only, and hybrid selectors under the same solver portfolio and benchmark conditions.

Therefore, although reinforcement learning provides an important foundation for learning-based SAT search, the present project does not adopt this approach because its objective is not to learn *how a solver should search*. Instead, the objective is to learn *which solver should be used* for a given instance. This distinction places the project within the per-instance algorithm-selection paradigm established by SATzilla and subsequent portfolio-based approaches, while allowing modern GNNs to be used as a source of learned structural information.

## **2.3 Deep Learning and Graph Neural Networks for SAT**

### **2.3.1 End-to-End Neural SAT Solvers**

The application of graph neural networks to SAT was popularised by Selsam, Lamm, Bünz, Liang, de Moura, and Dill, who introduced NeuroSAT, a message-passing neural network trained purely as a classifier to predict the satisfiability of a formula from a single bit of supervision \[10\]. NeuroSAT represents a CNF instance as a bipartite literal-clause graph and performs several rounds of message passing between literal and clause nodes before producing a satisfiability prediction; the authors further showed that, after training only on random SAT instances, the resulting literal embeddings could be decoded into a satisfying assignment more often than chance, and that the trained network generalised to instances of several other NP-complete problems encoded as SAT. Cameron, Chen, Hartford, and Leyton-Brown later compared permutation-invariant architectures with message-passing networks of the NeuroSAT family for the related task of directly predicting propositional satisfiability \[12\], confirming that end-to-end neural approaches can achieve non-trivial predictive accuracy but remain well short of the reliability guarantees and raw performance offered by engineered CDCL solvers. These end-to-end approaches are valuable for demonstrating that useful structural information can be learned purely from the graph representation of a formula, but their role is best understood as complementary to, rather than a replacement for, complete search-based solvers, which motivates their use in this project as a source of learned structural representations rather than as a stand-alone solving method.

### **2.3.2 Neural-Guided CDCL Solvers**

A second line of work embeds neural predictions inside an otherwise conventional CDCL solver. Selsam and Bjørner proposed NeuroCore, in which a simplified NeuroSAT-style graph neural network is trained to predict the likelihood that each variable participates in an unsatisfiable core, and the resulting predictions are used to periodically overwrite the variable activity scores that drive branching decisions in the solver \[11\]. Their experiments showed that solvers modified in this way, including MiniSat and Glucose, solved noticeably more instances than their unmodified counterparts within a fixed timeout on standard competition benchmarks. This approach demonstrates that learned structural signals can meaningfully improve solver behaviour even when the solver's core search procedure is left otherwise unchanged. However, because it requires the solver to repeatedly query the neural network during search, it introduces recurring computational overhead throughout solving, which is a materially different cost profile from the one-off, pre-search feature extraction used by portfolio selectors such as SATzilla and by the framework proposed in this project.

### **2.3.3 GNN-Based Algorithm Selection**

Graph neural networks (GNNs) are particularly relevant to SAT solver selection because SAT instances possess an inherent relational structure that is difficult to represent fully using a fixed set of scalar features. A CNF formula can naturally be represented as a graph in which literals or variables are connected to the clauses in which they occur. This representation preserves relationships between variables and clauses, rather than reducing the instance to aggregate statistics such as the number of variables, number of clauses, or clause-to-variable ratio. Since the performance of different SAT solvers can depend on structural properties and interactions within the formula, a representation that explicitly retains these relationships may provide useful information for predicting which solver is likely to perform best.

The main advantage of a GNN is its ability to learn representations directly from this graph structure. Through message passing, each node can iteratively aggregate information from its neighbouring nodes, allowing the model to capture increasingly broader structural patterns over multiple layers. For a SAT instance, this means that the representation of a literal or clause can incorporate information not only about its immediate connections, but also about other clauses and variables connected through the graph. Such multi-hop dependencies can represent structural characteristics of an instance that may be difficult to describe using manually designed features alone. In addition, graph-based representations are less dependent on the arbitrary ordering of variables and clauses in the input formula, allowing the model to focus more directly on structural relationships.

This capability is particularly valuable for algorithm selection. Different CDCL solvers employ different branching heuristics, restart policies, clause-learning strategies, preprocessing techniques, and other search mechanisms, meaning that they may respond differently to the same structural characteristics of a SAT instance. A GNN can therefore be used to learn a compact embedding of the instance that captures structural patterns associated with solver performance. Rather than replacing the underlying SAT solvers, the GNN acts as a representation-learning component that provides information to a higher-level selection model. The resulting embedding can then be used either independently to form a graph-based selector or combined with conventional handcrafted features in the proposed hybrid framework.

Closer to the focus of this project is a small but growing body of work that applies graph neural networks directly to the algorithm selection task, rather than to end-to-end solving or in-solver guidance. Shavit's master's thesis investigated feature-free algorithm selection for SAT using graph neural networks operating on literal-clause and variable-clause graph representations of CNF instances, training and evaluating the approach on a dataset of roughly three thousand synthetically generated instances \[13\]. The study found that a purely graph-based selector could be competitive with feature-based baselines in some settings, while also highlighting open questions around data efficiency and generalisation to larger and more heterogeneous instance sets. These findings suggest that graph representations can capture useful information for solver selection, while also indicating that relying exclusively on learned representations may not be sufficient when training data are limited.

This limitation is particularly important in the context of the present project because handcrafted SAT features continue to provide strong predictive information. Shavit and Hoos revisited the original SATzilla feature set nearly two decades after its introduction, re-implementing and evaluating the features against contemporary SAT Competition benchmarks \[14\]. Their re-implementation extracts features from a broader range of instances than the original tool and, on benchmarks from recent SAT Competitions, achieves up to 26% lower RMSE for running time prediction and up to fifteen times higher closed gap for algorithm selection, confirming that the handcrafted feature set continues to carry strong predictive signal despite the emergence of deep learning methods. This provides an important motivation for retaining handcrafted features in the proposed framework: learned graph representations should not be assumed to make established domain knowledge obsolete. Instead, the two representation types can provide complementary information, with handcrafted features capturing explicitly defined properties of the instance and GNN embeddings learning relational patterns that may not be captured by those features.

The work most directly related to the present proposal is GraSS, introduced by Zhang, Chételat, Cotnareanu, Ghose, Xiao, Zhen, Zhang, Hao, Coates, and Yuan \[15\]. GraSS represents each SAT instance as a tripartite graph over literals, clauses, and an additional node type, and applies a heterogeneous graph neural network enriched with domain-specific design choices, including hand-designed node features, positional encodings that capture the order in which clauses appear in the instance, and a runtime-sensitive loss function tailored to the algorithm selection objective. The authors report improvements over both purely feature-based selectors and prior graph-based selectors on an industrial circuit-design benchmark and on data derived from the SAT Competition. GraSS therefore provides strong evidence that graph-based representation learning can benefit from domain knowledge and that hybrid approaches are a promising direction for SAT solver selection.

However, the proposed framework differs from GraSS in its architectural design and experimental objective. Rather than incorporating handcrafted information directly into a single heterogeneous graph model, this project maintains two separate representation branches: a SATzilla-style handcrafted feature extractor and a GNN that learns an embedding from the literal-clause graph. The two representations are subsequently combined through an explicit fusion layer. This design makes it possible to evaluate the contribution of each representation independently and to compare different fusion strategies, such as concatenation, learned gating, or ensemble-based combination. Consequently, the project does not simply investigate whether a GNN can perform solver selection; it investigates whether the structural information learned by a GNN provides additional predictive value when combined with established handcrafted SAT features.

The rationale for using a GNN in this project can therefore be summarised as a complementarity between representation types. Handcrafted features provide interpretable and domain-informed measurements that have already demonstrated predictive value, while the GNN provides a learned representation of the relational structure of the SAT instance. Combining these representations may allow the selector to exploit both explicit domain knowledge and higher-order structural patterns, potentially producing a more robust solver-selection model than either representation alone. This motivates the hybrid architecture proposed in this project and the subsequent experimental comparison between handcrafted-only, graph-only, and hybrid selectors.

## 

## **2.4 Complementary Feature-Free Approaches**

A further strand of work has explored feature-free algorithm selection using representations other than graphs. Loreggia, Malitsky, Samulowitz, and Saraswat proposed converting the textual DIMACS encoding of a SAT instance into a fixed-size grayscale image, by mapping each character to a pixel intensity and rescaling the result to a standard resolution, and then applying a convolutional neural network trained to predict the fastest solver in a portfolio directly from this image \[16\]. This approach removes the need for hand-designed numeric features entirely, but the fixed-size rescaling step is lossy, discarding information for instances whose textual encoding is substantially larger or smaller than the target image resolution. It is included in this review because it illustrates an alternative, non-graph route to feature-free selection and reinforces the broader observation, consistent across this body of work, that there is no single representation that is uniformly best; different representations trade off information preservation, computational cost, and ease of training in different ways, which is precisely the trade-off that a hybrid approach seeks to navigate more effectively than any single representation alone.

This trade-off is not unique to SAT solving. Bengio, Lodi, and Prouvost, surveying the broader use of machine learning for combinatorial optimisation, similarly observe that learned models are most effective when combined with, rather than substituted for, the domain knowledge already encoded in mature combinatorial algorithms, and they identify algorithm selection as one of the clearest and most mature examples of this combination \[17\]. This broader methodological observation supports the design philosophy adopted in this project, namely that handcrafted domain knowledge and learned graph representations should be treated as complementary sources of information to be fused, rather than as competing approaches from which only one must be chosen.

## 

## **2.5 Limitations of Existing Studies**

Taken together, the literature reviewed above reveals a consistent pattern of complementary strengths and weaknesses across the two dominant representation families used for SAT solver selection. Handcrafted-feature approaches in the SATzilla tradition are mature, interpretable, and sample-efficient, performing well even when trained on relatively small datasets, but they depend on a fixed, manually engineered feature vocabulary that is costly to compute in full and may not capture higher-order relational structure that is not explicitly encoded by any individual feature. Graph-based, feature-free approaches such as NeuroSAT-derived selectors and Shavit's feature-free selector can, in principle, learn structural regularities that were not anticipated by the designers of any fixed feature set, but the studies reviewed above indicate that they typically require larger training sets, are less interpretable, and have so far been evaluated primarily on comparatively small or synthetic benchmarks, leaving open questions about generalisation to the full diversity of industrial and crafted instances found in recent SAT Competitions.

![][image3]

**Figure 2.3: Comparison of instance-representation families across the criteria that matter for per-instance solver selection, synthesised from the studies reviewed in Section II.**

Figure 2.3 sets out this trade-off explicitly. No column dominates the others: the handcrafted representation leads on interpretability and sample efficiency but is expensive to extract and structurally shallow, while the graph representation inverts almost every one of those judgments. It is precisely this pattern of mutually offsetting strengths that motivates combining the two rather than choosing between them.

Hybrid approaches such as GraSS provide encouraging evidence that combining the two representation families outperforms either alone, but the existing hybrid design embeds domain knowledge directly inside a single heterogeneous graph architecture and loss function, evaluated in part on proprietary industrial data, which makes it difficult to isolate how much of the reported improvement is attributable to the graph representation itself, to the injected domain knowledge, or to their specific combination. A systematic, openly reproducible study that keeps handcrafted and graph-based representations as separately measurable components, compares multiple explicit fusion strategies, and evaluates the resulting selectors on open SAT Competition benchmark data has not yet been established in the literature. This is the gap that the present project addresses.

![][image4]

**Figure 2.4: Development of learning-based algorithm selection for SAT, from Rice’s formalisation of the problem to the hybrid approaches of 2024, showing the gap addressed by this project.**

Figure 2.4 places this gap in historical context. Feature-based selection matured over roughly a decade from Rice’s formalisation to SATzilla and its supporting infrastructure; graph-based learning entered the field only in 2019; and the first explicitly hybrid selector appeared in 2024\. The convergence of the two lines is therefore recent enough that the relative contribution of each representation has not yet been systematically measured.

## 

## **2.6 Proposed Solution**

Building on the strengths and limitations identified in the literature review, this project proposes a hybrid learning framework that combines two complementary representations of a SAT instance: a SATzilla-style handcrafted feature vector and a learned graph embedding produced by a message-passing graph neural network operating on the instance's literal-clause graph. The handcrafted branch will capture established, interpretable structural characteristics of the instance, while the GNN branch will learn relational patterns directly from the graph representation.

![][image5]

**Figure 2.5: Positioning of the proposed framework relative to existing approaches, along the two axes that distinguish them: reliance on handcrafted domain knowledge and reliance on learned structural representation.**

Figure 2.5 shows where this places the project relative to existing work. The proposal occupies the same upper-right quadrant as GraSS, drawing on both domain knowledge and learned structure, but reaches it by a different route: GraSS fuses the two inside a single heterogeneous graph model, whereas this project keeps them as separately measurable branches joined by an explicit fusion layer.

The two representations will be maintained as separate branches so that their individual contributions can be evaluated before they are combined. At least two fusion strategies will be implemented and compared, including early fusion through feature concatenation and a second strategy such as learned gating or stacking. A meta-classifier will then use the resulting representation to predict which solver from the selected portfolio is expected to perform best for the given instance.

The proposed hybrid selector will be trained and evaluated on publicly available benchmark instances drawn from recent SAT Competition suites using a fixed portfolio of established open-source CDCL solvers. Its performance will be compared against a Single Best Solver (SBS), a handcrafted feature-only selector, a graph-only selector, and the Virtual Best Solver (VBS) oracle. This design allows the study to determine not only whether the hybrid representation improves solver selection, but also whether the GNN provides predictive information beyond the handcrafted feature representation and whether combining the two representations produces a measurable benefit.

The detailed graph representation, feature extraction procedure, GNN architecture, fusion strategies, solver portfolio, benchmark data, and evaluation methodology are described in Section IV.

# **SECTION III**

**PROJECT SCOPE AND OBJECTIVES**

## **3.1 Project Scope**

The deliverable of this project is a software framework and accompanying experimental study for hybrid SAT solver selection. The framework will extract a computationally practical set of handcrafted features from each SAT instance, construct a literal-clause graph representation, generate a fixed-size structural embedding using a message-passing graph neural network, combine the handcrafted and graph representations using at least two fusion strategies, and train a meta-classifier to predict the solver expected to perform best on a previously unseen instance.

The scope of the project is deliberately bounded to allow a meaningful experimental study within a single trimester. The project focuses on complete, decision-form SAT solving; related problems such as Maximum Satisfiability (MaxSAT) and Quantified Boolean Formulas (QBF) are outside the current scope. The project adopts **offline, per-instance algorithm selection**, in which a solver is selected once before search begins. Online in-solver learning and guidance, such as neural intervention within the CDCL search process, are not considered. Similarly, dynamic parallel scheduling of multiple solvers is outside the scope.

The handcrafted representation will use a practical subset of SATzilla-style and revisited SATzilla feature families rather than the full feature set, particularly where complete feature extraction would introduce unnecessary computational overhead. The GNN branch will operate on the literal-clause graph of each instance and will produce an instance-level embedding for solver selection. The candidate solver portfolio will consist of a small number of established open-source CDCL solvers, while benchmark instances will be obtained from publicly available SAT Competition suites. No proprietary industrial datasets are required, ensuring that the experimental setup remains reproducible.

## **3.2 Project Objectives**

The main objective of this project is to design, implement, and empirically evaluate a hybrid learning framework that combines handcrafted instance features and graph neural network embeddings for automated, per-instance SAT solver selection. The study will determine whether the hybrid representation provides an advantage over using either representation independently and will quantify the contribution of each representation under the same experimental conditions.

This main objective is divided into the following sub-objectives:

* **O1:** Implement a computationally practical handcrafted feature-extraction pipeline for CNF instances based on SATzilla and the revisited SATzilla feature families.  
* **O2:** Implement a literal-clause graph construction and message-passing graph neural network module that converts each CNF instance into a fixed-size structural embedding.  
* **O3:** Design and implement at least two fusion strategies for combining the handcrafted feature vector and GNN embedding, together with a meta-classifier for predicting the best solver in the portfolio.  
* **O4:** Evaluate the feature-only, graph-only, and hybrid selectors against the Single Best Solver baseline and the Virtual Best Solver oracle using standard algorithm-selection metrics, and analyse the resulting performance through ablation and comparative studies in order to determine the marginal contribution of the GNN representation and the relative effectiveness of the fusion strategies implemented under O3.

## 

## **3.3 Contributions**

This project is expected to make several contributions to the study of learning-based SAT solver selection. First, it will produce a reproducible implementation of a hybrid feature-and-graph solver-selection framework using open-source SAT solvers and publicly available benchmark instances. This provides a concrete experimental framework for studying the combination of handcrafted and learned representations.

Second, the project will provide an empirical comparison of three representation settings: handcrafted features alone, GNN-derived graph embeddings alone, and the combination of both. By keeping the solver portfolio, benchmark instances, and evaluation procedure consistent across these settings, the study will quantify whether the GNN captures predictive information that is complementary to established handcrafted SAT features.

Third, the project will compare multiple fusion strategies for combining the two representation types. This will provide evidence on whether a simple combination of the representations is sufficient or whether a learned mechanism for weighting their contributions provides additional benefit.

Finally, the project will contribute a systematic evaluation of the proposed hybrid approach against established algorithm-selection baselines, including the Single Best Solver and Virtual Best Solver. The results may provide practical insight into the design of learning-based solver selectors and the potential role of graph-based representations in SAT solving and related ML4Sys applications.

# **SECTION IV**

**METHODS/TECHNOLOGIES INVOLVED**

## **4.1 Overall System Architecture**

The proposed hybrid selector adopts a two-branch architecture followed by a feature fusion layer and a meta-classification stage, as illustrated in Figure 4.1. The architecture is designed to capture complementary information from both the statistical characteristics and the structural properties of a given SAT instance. Each input instance is provided in the standard DIMACS CNF format and is processed simultaneously by two independent branches: a handcrafted feature extraction branch and a graph-based learning branch.

![][image6]

**Figure 4.1: Proposed hybrid architecture combining handcrafted features and a graph neural network embedding for SAT solver selection.**

In the **handcrafted feature extraction branch**, a set of manually designed numerical features is extracted from the SAT instance. These features describe various characteristics of the problem, such as the number of variables and clauses, clause-to-variable ratios, literal distributions, and other statistical properties that can provide useful information about the underlying instance. The resulting features are organised into a fixed-length numerical feature vector, providing a compact representation of the instance based on domain-specific knowledge.

In parallel, the **graph-based branch** transforms the SAT instance into a graph representation that captures relationships between its constituent variables and clauses. This graph is subsequently processed by a graph neural network (GNN), which learns latent representations by aggregating information from neighbouring nodes. The node-level representations generated by the GNN are then aggregated through a pooling operation to obtain a fixed-length graph embedding vector. Unlike the handcrafted branch, this representation is learned directly from the structural characteristics of the SAT instance.

The outputs from both branches are subsequently passed to a **fusion layer**, where the handcrafted feature vector and learned graph embedding are combined into a unified representation. The fusion mechanism is an important component of the proposed system and constitutes one of the experimental variables investigated in this study. Different fusion strategies can therefore be evaluated to determine how effectively the complementary information from the two branches can be integrated.

Finally, the fused representation is provided to a **meta-classifier**, which learns to associate the combined characteristics of an SAT instance with the performance of the candidate solver portfolio. The meta-classifier produces a predicted solver that is expected to perform best on the given instance. Overall, this architecture enables the proposed selector to jointly exploit interpretable, domain-informed features and automatically learned structural representations, with the objective of improving solver-selection accuracy and generalisation across diverse SAT instances.

## **4.2 SAT Instance Representation and Handcrafted Feature Extraction**

Each SAT instance is first parsed from its standard **DIMACS CNF (Conjunctive Normal Form)** representation, in which the Boolean formula is expressed as a conjunction of clauses, with each clause consisting of one or more literals. The DIMACS representation provides a consistent input format for both feature extraction and subsequent graph construction, ensuring that the same underlying SAT instance can be processed by the two branches of the proposed hybrid architecture.

The handcrafted feature extraction module will implement a practical subset of feature families commonly used in **SATzilla** \[5\] and subsequently revisited by Shavit and Hoos \[14\]. These features are selected to capture complementary characteristics of SAT instances that may provide useful indications of solver performance. The extracted features will include **problem-size features**, such as the number of variables, number of clauses, and clause-to-variable ratio, which provide a basic description of the scale and density of the instance.

In addition, **variable-clause graph features** will be used to describe the connectivity and degree characteristics of variables and clauses. These features can provide information about the structural complexity of the underlying formula. **Balance features** will also be extracted to measure the distribution of positive and negative literals, allowing the representation to capture potential asymmetries within the Boolean formula. Furthermore, **proximity-to-Horn-formula features** will be included to indicate how closely an instance resembles a Horn formula, a structural property that can be relevant to the behaviour of particular SAT-solving techniques.

A limited set of **bounded probing features** will additionally be considered. These features are obtained through short, controlled runs of local-search procedures and are intended to capture behavioural information that cannot be inferred solely from static structural statistics. Since probing introduces additional computational overhead, each probing operation will be performed under a strict time budget. This constraint ensures that feature extraction remains computationally inexpensive and represents only a small fraction of the time normally required to solve the SAT instance.

Where practical, an existing **open-source SATzilla-style feature extractor** will be reused or adapted rather than reimplemented entirely from first principles. This approach reduces unnecessary implementation effort while maintaining consistency with established SAT feature definitions. Any modifications will be restricted primarily to compatibility with the DIMACS instance formats, feature-processing pipeline, and experimental requirements of the proposed system. The resulting features will be normalised where appropriate and assembled into a fixed-length numerical vector, which will subsequently be supplied to the handcrafted branch of the hybrid selector.

## 

## **4.3 Graph Neural Network Embedding Module**

To complement the handcrafted statistical representation, each CNF instance will additionally be transformed into a **literal-clause bipartite graph**, following the representation adopted in NeuroSAT \[10\] and subsequently used in Shavit's algorithm-selection study \[13\]. This representation allows the proposed system to capture structural relationships within the SAT formula that may not be fully described by manually engineered numerical features.

In the constructed graph, each **literal**, representing a Boolean variable together with its polarity, and each **clause** is represented as a separate node. An edge is established between a literal node and a clause node whenever the corresponding literal occurs in that clause. Consequently, the resulting bipartite graph contains two distinct types of nodes and captures the incidence relationships between literals and clauses. This representation preserves important structural information about the original CNF formula, including variable participation, clause composition, and connectivity patterns.

A **message-passing graph neural network (GNN)** will then be applied to the constructed graph. Initially, each literal and clause node is assigned a learnable or feature-based embedding. During each message-passing round, nodes receive information from their neighbouring nodes and update their representations based on the aggregated neighbourhood information. Separate update functions may be employed for literal and clause nodes to account for their different semantic roles within the graph. These update functions will be implemented using standard neural network layers, allowing the model to progressively learn higher-level structural representations of the SAT instance.

Multiple rounds of message passing will be performed so that information can propagate across increasingly larger portions of the graph. As a result, the final representation of a literal node can incorporate information not only about the clauses in which it occurs, but also about other literals and clauses connected through the surrounding graph structure. This enables the GNN to learn structural patterns that may be associated with the relative performance of different SAT solvers.

Following the final message-passing round, the resulting node-level embeddings will be aggregated to produce a **fixed-length instance-level graph embedding**. Pooling methods such as mean pooling or attention-weighted pooling over the literal node embeddings will be investigated. Mean pooling provides a simple permutation-invariant representation, whereas attention-based pooling can allow the model to assign greater importance to structurally informative nodes. The resulting graph embedding will serve as the learned representation of the SAT instance and will subsequently be combined with the handcrafted feature vector in the fusion stage described in Section 4.4.

The GNN module will be implemented in **Python using PyTorch**, together with a graph-learning framework such as **PyTorch Geometric (PyG)**. PyG provides established implementations of message-passing and graph-processing operations, enabling the model to be developed efficiently while maintaining compatibility with standard PyTorch training procedures. The use of an established graph-learning framework also reduces the need to implement low-level graph operations manually and facilitates experimentation with different GNN architectures and pooling strategies.

## 

## **4.4 Hybrid Fusion and Meta-Classifier**

The handcrafted feature representation and the learned graph embedding provide complementary information about each SAT instance. To effectively integrate these two representations, the proposed system will implement and compare at least **two fusion strategies: early fusion and late/gated fusion**. The purpose of evaluating multiple fusion mechanisms is to determine whether directly combining the representations or allowing the model to learn their relative contributions provides better solver-selection performance.

The first approach, **early fusion**, will concatenate the fixed-length handcrafted feature vector with the pooled graph embedding to form a single unified feature vector. This combined representation will then be supplied to a downstream meta-classifier. Candidate classifiers will include a **gradient-boosted tree ensemble** and a **multilayer perceptron (MLP)**. The tree-based approach can effectively model nonlinear relationships within heterogeneous numerical features, while the MLP provides a flexible neural architecture capable of learning interactions between the handcrafted and graph-derived representations.

The second approach, **late or gated fusion**, will process the two representations through separate predictive branches before combining their outputs. In the gated formulation, a learned gating mechanism will determine the relative contribution of the handcrafted and graph-based predictions for each SAT instance. This allows the model to adaptively favour one representation when it is more informative for a particular type of instance. For example, instances whose solver performance is strongly associated with structural graph characteristics may benefit more from the GNN representation, whereas instances characterised by readily measurable statistical properties may rely more heavily on handcrafted features. As an alternative late-fusion implementation, a **stacked ensemble** may be used, in which the predictions generated by the two individual predictors are provided to a higher-level classifier that learns how to combine them.

The primary meta-classification task will be formulated as a **multi-class classification problem**. Each candidate solver in the portfolio will correspond to a class, and the trained model will predict the solver expected to achieve the best performance for a previously unseen SAT instance. The predicted class therefore determines which solver is selected for execution. This formulation directly addresses the central objective of the proposed system: selecting the most suitable solver based on the characteristics of the input instance.

In addition to the classification-based formulation, a **regression-based approach** will be implemented as a secondary point of comparison. In this setting, the model will predict the expected runtime of each candidate solver rather than directly predicting a solver label. The solver associated with the lowest predicted runtime will then be selected. This formulation follows the general principle of the original SATzilla methodology \[5\], where solver selection can be treated as a performance-prediction problem. Comparing the classification and regression approaches will provide further insight into whether directly predicting the best solver or indirectly selecting it through runtime estimation is more effective for the proposed hybrid architecture.

Overall, the fusion and meta-classification stage will allow the study to evaluate not only the effectiveness of combining handcrafted and learned representations, but also the impact of different decision-making strategies on automated SAT solver selection.

## 

## **4.5 Solver Portfolio and Benchmark Datasets**

Table 4.1 lists the candidate solver portfolio for this project. The portfolio comprises MiniSat \[18\], Glucose \[19\], CaDiCaL \[20\], CryptoMiniSat \[21\] and Kissat \[20\]. It is deliberately kept to a small number of well-established, actively maintained, open-source CDCL solvers that represent different heuristic design choices, so that meaningful performance complementarity is likely to be observed while keeping the runtime-data collection process manageable within the available time budget.

**Table 4.1: Candidate solver portfolio for the proposed hybrid selection framework**

| Solver | Core Technique | Rationale for Inclusion |
| :---- | :---- | :---- |
| MiniSat | Baseline CDCL with VSIDS branching | Widely used reference solver; common in prior algorithm-selection studies |
| Glucose | CDCL with Literal Block Distance clause deletion | Strong performance on industrial-style instances |
| CaDiCaL | Modern CDCL with efficient inprocessing | Competitive recent SAT Competition solver |
| CryptoMiniSat | CDCL with Gaussian elimination extensions | Strong on cryptanalysis-derived and XOR-heavy instances |
| Kissat | Highly optimised single-threaded CDCL | Recent SAT Competition top performer |

Benchmark instances will be drawn from recent SAT Competition benchmark suites, which provide a well-established mix of industrial, crafted, and randomly generated instances and are the standard evaluation substrate used throughout the literature reviewed in Section II. Runtime data for each solver on each selected instance will be collected through controlled, sequential batch execution under a fixed CPU and wall-clock timeout, with unsolved instances treated as censored observations following the standard PAR10 convention described below. The resulting instance-runtime dataset will be partitioned into training, validation, and test splits; where practical, splits will additionally be stratified by instance family so that the framework's ability to generalise across, rather than merely within, instance families can be assessed.

## **4.6 Comparative Performance of Candidate SAT Solvers**

The five candidate solvers span roughly two decades of CDCL development, and each has held a leading position in a competitive evaluation of its own era. Table 4.2 records one such reference result per solver, together with the competition it was obtained from. Because these results come from different competitions, with different benchmark sets, hardware and timeouts, the figures in the table are not comparable with one another; they are given as evidence that each solver is an established and competitive implementation, not as a controlled head-to-head measurement. 

**Table 4.2: Reference competition results for the candidate solver portfolio**

| Solver (version) | Reference competition | Reported result | Principal architectural characteristic |
| :---- | :---- | :---- | :---- |
| Kissat (sc2024) | SAT Competition 2024, main track; 400 instances, 5,000 s timeout \[22\] | First place overall; 306 instances solved (153 SAT / 153 UNSAT); PAR-2 2,788.13 | Highly optimised single-threaded CDCL with congruence closure and clausal equivalence sweeping \[20\] |
| CaDiCaL (sc2025) | SAT Competition 2025, main sequential track; 400 instances, 5,000 s timeout \[23\] | First place on unsatisfiable instances; 161 solved; PAR-2 2,327.00 | Modern CDCL with aggressive inprocessing, bounded variable addition and vivification \[20\] |
| CryptoMiniSat (5.11) | SAT Race 2010 \[24\] | Gold medal | Gauss–Jordan elimination over XOR constraints recovered from the CNF encoding \[21\] |
| Glucose (3.0 / 4.2.1) | SAT Competition 2011 \[24\] | Gold medal | Literal Block Distance clause scoring with aggressive learnt-clause deletion \[19\] |
| MiniSat (2.2.0) | SAT Race 2006 and 2008 \[24\] | Gold medal | Reference CDCL implementation with VSIDS branching; the codebase from which Glucose and CryptoMiniSat descend \[18\] |

The table also reveals an important characteristic of SAT solving: there is no single architectural strategy that is uniformly optimal across all instance types. CryptoMiniSat's specialised XOR reasoning, for example, targets a substantially different structural property from CaDiCaL's general-purpose inprocessing, while Kissat focuses heavily on overall search and implementation efficiency. Consequently, aggregate competition scores alone do not explain *which* solver should be selected for an individual SAT instance.

This is not merely an inference across competition years; it is visible within a single competition. Figure 4.2 shows the three main-track podiums of SAT Competition 2024, which were computed over the same 400 instances under identical conditions. Only the overall winner holds its position in all three categories: the solver placing second on satisfiable instances does not appear on the unsatisfiable podium, and the solver placing second on unsatisfiable instances does not appear on the satisfiable one. Complementarity of exactly the kind this project seeks to exploit is therefore already measurable among near-identical, state-of-the-art solvers.

![][image7]

**Figure 4.2: Main-track podiums of SAT Competition 2024 for satisfiable, unsatisfiable and all instances \[22\]. Solvers change position between categories, demonstrating per-instance complementarity within a single controlled evaluation.**

This observation provides the primary motivation for the proposed learning-based approach. Instead of attempting to develop a universally superior SAT solver, the objective is to learn a mapping between SAT-instance characteristics and solver performance. Structural and graph-based features extracted from an input formula can be used to estimate which of the candidate architectures is most likely to perform efficiently. The five solvers consequently provide a deliberately heterogeneous candidate pool spanning classical CDCL, LBD-based learning, XOR-aware reasoning, aggressive inprocessing and modern low-level CDCL optimisation.

It should be noted that the reported solved-instance counts and PAR-2 values originate from different solver generations, configurations and competition contexts and should therefore be interpreted primarily as evidence of architectural evolution rather than as a controlled head-to-head experiment. For the experimental evaluation of the proposed model, all candidate solvers should subsequently be executed under identical hardware, timeout, benchmark and configuration conditions. This will produce an internally consistent performance matrix from which solver-selection labels and quantitative performance targets can be generated.

## 

## **4.7 Experimental Design and Evaluation Metrics**

The hybrid selector will be compared against four baselines, all evaluated on the same train/test partition: the Single Best Solver (SBS), the fixed solver from the portfolio with the best aggregate performance on the training set; the Virtual Best Solver (VBS), an oracle that always selects the fastest solver for each test instance and represents the theoretical upper bound on selector performance; a reproduced feature-only selector, following the general SATzilla methodology described in Section 2.1.1; and a reproduced graph-only selector, following the general methodology described in Section 2.3.3.

![][image8]

**Figure 4.3: Experimental pipeline, from benchmark instances and solver portfolio through runtime-data collection and label generation to selector training and evaluation.**

The primary evaluation metric will be the Penalised Average Runtime with a penalty factor of ten (PAR10), the standard metric used throughout the algorithm-selection literature reviewed in Section II, in which the runtime of any instance not solved within the timeout is replaced by ten times the timeout before averaging. Selection accuracy, defined as the proportion of test instances for which the selector's chosen solver matches the oracle's choice, and the percentage of the gap between the single-best-solver and virtual-best-solver performance recovered by the selector, will be reported as secondary metrics. An ablation study will additionally compare the hybrid selector against variants using only the handcrafted feature branch and only the graph embedding branch, holding the meta-classifier and training procedure fixed, in order to isolate the marginal contribution of each representation; where the training set size permits, learning curves at varying dataset sizes will also be examined to assess the sample efficiency of each representation. Results will be reported as averages over multiple cross-validation folds, together with an indication of variability across folds.

Figure 4.4 illustrates the two quantities that these metrics express. Panel (a) shows the solved-instance profile that each selector and baseline will produce, and panel (b) shows the interval between the single-best-solver and virtual-best-solver results within which every selector must fall. The proportion of that interval closed by a given selector is the figure of merit by which the hybrid, feature-only and graph-only variants will be compared.

![][image9]

**Figure 4.4: Schematic of the evaluation. (a) Solved-instance profile against runtime cutoff; (b) the SBS–VBS interval and the portion of it closed by each selector. Axes carry no measured data; the figure defines the quantities to be reported, not anticipated results.**

## 

## **4.8 Development Tools and Environment**

**Hardware \-** Development and experimentation will be carried out on a faculty laboratory workstation. The intended configuration is a modern multi-core x86-64 processor with at least eight physical cores, 16 GB of system memory, a mid-range CUDA-capable GPU with at least 8 GB of video memory for graph neural network training, and approximately 512 GB of local SSD storage for benchmark instances, extracted features and cached graph representations. Two distinct workloads must be supported, and they place different demands on this machine. Runtime-data collection requires the candidate solvers to be timed under controlled conditions, so those runs will be executed sequentially, one solver job per physical core with the remaining cores left idle, in order to avoid the memory-bandwidth contention that would otherwise distort the measured runtimes on which every subsequent label depends. Model training, by contrast, is throughput-bound rather than timing-sensitive and will use the GPU and all available cores.

The GPU accelerates training but is not a precondition for it. Should GPU access prove unavailable, the project remains feasible on CPU alone by reducing the number of message-passing rounds, the embedding dimension and the size of the instance subset; this would lengthen training and narrow the scope of the hyperparameter search, but would not invalidate the comparison between feature-only, graph-only and hybrid selectors, since all three would be affected equally. The timed solver runs, which are the more schedule-critical of the two workloads, require no GPU at all.

**Software \-** The framework will be implemented in Python 3.11 on Ubuntu Linux. PyTorch will provide the neural network components and PyTorch Geometric the graph neural network layers and batching utilities for variable-sized graphs. scikit-learn and XGBoost will supply the classical machine learning baselines and the gradient-boosted meta-classifier described in Section 4.4, with NumPy, pandas and SciPy used for data handling and statistical analysis, and Matplotlib for result visualisation. The five candidate solvers listed in Table 4.1 will be compiled from their public source repositories with a C/C++ toolchain, and each will be pinned to a specific release tag so that the runtime matrix can be regenerated exactly. Solver runs will be executed under the runsolver utility, or an equivalent wrapper, to enforce CPU-time and memory limits consistently and to record resource usage per run.

Handcrafted feature extraction will reuse the publicly released SATzilla feature extractor, in the revised form provided by Shavit and Hoos \[14\], rather than reimplementing the feature definitions. Benchmark instances will be obtained from the public SAT Competition benchmark archives. Version control will be maintained using Git and hosted on GitHub, with the Python environment captured in a lock file and experiment configurations stored alongside the code, so that the reported results can be reproduced from the repository alone.

## 

## **4.9 Project Schedule**

Figure 4.5 presents a tentative week-by-week schedule for the project, aligned with the sub-objectives defined in Section 3.2 and the submission deadline for the preliminary proposal.

![][image10]

**Figure 4.5: Tentative project schedule, showing the sub-objectives of Section 3.2 against a fourteen-week trimester. Milestones: M1 (week 6\) runtime matrix complete; M2 (week 11\) all selectors trained; M3 (week 14\) submission.**

# **REFERENCES**

\[1\] A. Biere, M. Heule, H. van Maaren, and T. Walsh, Eds., Handbook of Satisfiability, 2nd ed. Amsterdam, The Netherlands: IOS Press, 2021\.

\[2\] J. Marques-Silva, I. Lynce, and S. Malik, "Conflict-driven clause learning SAT solvers," in Handbook of Satisfiability, 2nd ed., A. Biere, M. Heule, H. van Maaren, and T. Walsh, Eds. Amsterdam, The Netherlands: IOS Press, 2021, pp. 133–182.

\[3\] J. R. Rice, "The algorithm selection problem," Advances in Computers, vol. 15, pp. 65–118, 1976\.

\[4\] P. Kerschke, H. H. Hoos, F. Neumann, and H. Trautmann, "Automated algorithm selection: Survey and perspectives," Evolutionary Computation, vol. 27, no. 1, pp. 3–45, 2019\.

\[5\] L. Xu, F. Hutter, H. H. Hoos, and K. Leyton-Brown, "SATzilla: Portfolio-based algorithm selection for SAT," Journal of Artificial Intelligence Research, vol. 32, pp. 565–606, 2008\.

\[6\] E. Nudelman, K. Leyton-Brown, H. H. Hoos, A. Devkar, and Y. Shoham, "Understanding random SAT: Beyond the clauses-to-variables ratio," in Proc. 10th Int. Conf. Principles and Practice of Constraint Programming (CP), Toronto, Canada, 2004, pp. 438–452.

\[7\] B. Bischl, P. Kerschke, L. Kotthoff, M. Lindauer, Y. Malitsky, A. Fréchette, H. Hoos, F. Hutter, K. Leyton-Brown, K. Tierney, and J. Vanschoren, "ASlib: A benchmark library for algorithm selection," Artificial Intelligence, vol. 237, pp. 41–58, 2016\.

\[8\] M. Lindauer, H. H. Hoos, F. Hutter, and T. Schaub, "AutoFolio: An automatically configured algorithm selector," Journal of Artificial Intelligence Research, vol. 53, pp. 745–778, 2015\.

\[9\] M. G. Lagoudakis and M. L. Littman, "Learning to select branching rules in the DPLL procedure for satisfiability," Electronic Notes in Discrete Mathematics, vol. 9, pp. 344–359, 2001\.

\[10\] D. Selsam, M. Lamm, B. Bünz, P. Liang, L. de Moura, and D. L. Dill, "Learning a SAT solver from single-bit supervision," in Proc. 7th Int. Conf. Learning Representations (ICLR), New Orleans, LA, USA, 2019\.

\[11\] D. Selsam and N. Bjørner, "Guiding high-performance SAT solvers with unsat-core predictions," in Proc. 22nd Int. Conf. Theory and Applications of Satisfiability Testing (SAT), Lisbon, Portugal, 2019, pp. 336–353.

\[12\] C. Cameron, R. Chen, J. Hartford, and K. Leyton-Brown, "Predicting propositional satisfiability via end-to-end learning," in Proc. AAAI Conf. Artificial Intelligence, vol. 34, no. 4, 2020, pp. 3324–3331.

\[13\] H. Shavit, "Algorithm selection for SAT using graph neural networks," M.S. thesis, Leiden Institute of Advanced Computer Science, Leiden Univ., Leiden, The Netherlands, 2023\.

\[14\] H. Shavit and H. H. Hoos, "Revisiting SATzilla features in 2024," in Proc. 27th Int. Conf. Theory and Applications of Satisfiability Testing (SAT), Pune, India, 2024, pp. 27:1–27:26.

\[15\] Z. Zhang, D. Chételat, J. Cotnareanu, A. Ghose, W. Xiao, H.-L. Zhen, Y. Zhang, J. Hao, M. Coates, and M. Yuan, "GraSS: Combining graph neural networks with expert knowledge for SAT solver selection," in Proc. 30th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD), Barcelona, Spain, 2024, pp. 6301–6311.

\[16\] A. Loreggia, Y. Malitsky, H. Samulowitz, and V. Saraswat, "Deep learning for algorithm portfolios," in Proc. 30th AAAI Conf. Artificial Intelligence, Phoenix, AZ, USA, 2016, pp. 1280–1286.

\[17\] Y. Bengio, A. Lodi, and A. Prouvost, "Machine learning for combinatorial optimization: A methodological tour d'horizon," European Journal of Operational Research, vol. 290, no. 2, pp. 405–421, 2021\.

\[18\] N. Eén and N. Sörensson, "An extensible SAT-solver," in Proc. 6th Int. Conf. Theory and Applications of Satisfiability Testing (SAT), Santa Margherita Ligure, Italy, 2003, pp. 502–518.

\[19\] G. Audemard and L. Simon, "Predicting learnt clauses quality in modern SAT solvers," in Proc. 21st Int. Joint Conf. Artificial Intelligence (IJCAI), Pasadena, CA, USA, 2009, pp. 399–404.

\[20\] A. Biere, T. Faller, K. Fazekas, M. Fleury, N. Froleyks, and F. Pollitt, "CaDiCaL, Gimsatul, IsaSAT and Kissat entering the SAT Competition 2024," in Proc. SAT Competition 2024: Solver, Benchmark and Proof Checker Descriptions, M. Heule, M. Iser, M. Järvisalo, and M. Suda, Eds., Dept. Comput. Sci. Rep. Ser. B, vol. B-2024-1. Helsinki, Finland: Univ. Helsinki, 2024, pp. 8–10.

\[21\] M. Soos, K. Nohl, and C. Castelluccia, "Extending SAT solvers to cryptographic problems," in Proc. 12th Int. Conf. Theory and Applications of Satisfiability Testing (SAT), Swansea, U.K., 2009, pp. 244–257.

\[22\] M. Heule, M. Iser, M. Järvisalo, and M. Suda, "The results of SAT Competition 2024," presented at the 27th Int. Conf. Theory and Applications of Satisfiability Testing (SAT), Pune, India, Aug. 2024\.

\[23\] C. Codel, K. Fazekas, M. Heule, and M. Iser, "The results of SAT Competition 2025," presented at the 28th Int. Conf. Theory and Applications of Satisfiability Testing (SAT), Glasgow, U.K., Aug. 2025\.

\[24\] A. Biere, M. Fleury, N. Froleyks, and M. Heule, "The SAT Museum," in Proc. 14th Int. Workshop on Pragmatics of SAT (POS), Alghero, Italy, 2023, pp. 72–86.

[image1]: figures/image1.png

[image2]: figures/image2.png

[image3]: figures/image3.png

[image4]: figures/image4.png

[image5]: figures/image5.png

[image6]: figures/image6.png

[image7]: figures/image7.png

[image8]: figures/image8.png

[image9]: figures/image9.png

[image10]: figures/image10.png
