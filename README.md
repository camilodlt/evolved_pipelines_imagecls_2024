# Evolved and Transparent Pipelines for Biomedical Image Classification

This article presents an interpretable approach to binary image classification using 
Genetic Programming, applied to the PCAM dataset, which contains small tissue biopsy patches labeled as 
malignant or benign. While DNNs achieve high performance in image classification, their opaque decision-making processes, 
prone to overfitting behavior and dependency on large amounts of annotated data limit their utility in critical fields like 
digital pathology, where interpretability is essential. To address this, we employ GP, specifically using the MAGE framework, 
to evolve end-to-end image classification pipelines. 

We trained MAGE a hundred times with the best optimized key hyperparameters for this task. Among all MAGE models trained, 
the best one achieved 78\% accuracy on the validation set and 76\% accuracy on the test set. Among CNNs, our baseline, 
the best model obtained 84.5\% accuracy on the validation set and 77.1\% accuracy on the test set. 
Unlike CNNs, our GP approach enables program-level transparency, facilitating interpretability through example-based reasoning. 
By analyzing evolved programs with medical experts, we highlight the transparency of decision-making in MAGE pipelines, 
offering an interpretable alternative for medical image classification tasks where model interpretability is paramount.

## Data

For this paper we used the PCAM dataset available from [this link](https://github.com/basveeling/pcam)
following the same data splits. 

## MAGE 

[Paper presenting the MAGE framework (and extension of CGP)](https://link.springer.com/chapter/10.1007/978-3-031-70055-2_19)
```
De La Torre, C., Lavinas, Y., Cortacero, K., Luga, H., Wilson, D. G., & Cussat-Blanc, S. (2024, September). Multimodal Adaptive Graph Evolution for Program Synthesis. In International Conference on Parallel Problem Solving from Nature (pp. 306-321). Cham: Springer Nature Switzerland.
```

For this paper we used MAGE at this [commit](https://github.com/camilodlt/MAGE.jl/tree/evostar). For extracting the HED channels,
we used a python environment with the skimage library.

# Supplementary materials

[SUPPLEMENTARY MATERIALS PDF](https://github.com/camilodlt/evolved_pipelines_imagecls_2024/blob/main/sup_mat.pdf)

# Citation 

