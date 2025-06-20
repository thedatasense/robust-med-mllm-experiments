## Visual perturbations 

1. **Natural Noise and Artifacts** - Regular CXR vs Porrable ER noises, grid lines from digitizers
2. **Adversarial Noises** : **FGSM**  -a single-step attack that calculates the gradient of the model's loss with respect to the input image and then adds a small perturbation in the direction of the sign of that gradient to maximize the loss, often causing misclassification & PGD - An iterative extension of FGSM, PGD takes multiple smaller steps in the direction of the gradient of the loss, and after each step, it projects the perturbed image back into a small, allowed neighborhood around the original image to ensure the changes are imperceptible.
3. **Adversarial Patches**: Instead of distributed noise, insert a localized patch
4. **Domain Shifts** :  different hospitals, populations, demographics
5. Data Quality Issues
6. Composite Attacks


## Current Defense Techniques 

1. **Adversarial Training**: The model is explicitly trained on adversarially perturbed images so that it learns to be resilient. computationally expensive vs accuracy compromises
2. **Data Augmentation:**   augment the training data with various perturbations: adding noise, random rotations, scaling, simulating lower quality, Standard augmentations (flips, rotations).
3. **Defensive Preprocessing**:  inference time,  apply filters to the input to remove potential adversarial signals.
4. **Certified and Provable Defenses** - like the randomized smoothing
5. **Ensemble and Redundancy**


## VLM Specific Issues

### Architecture 
1. Vision Encoder (ViT, SigLIP)) - encoder might output a set of image feature tokens or a fixed-length embedding vector.MedGemma’s encoder normalizes a cxr  to 896×896 and encodes it into 256 latent tokens
2. Fusion Layer  - map image features into the LLM’s token spac,  cross-attention where the LLaVA  learned projection that turns the visual encoder’s output into pseudo-word embeddings Vicuna attend to . During training the model learns to align visual features with textual descriptions (feature vector for an enlarged cardiac silhouette aligns with the token “cardiomegaly”)
3. LLM - model generates an output sequence based conditioned on the fused representation. the cxr has [MASK]

### Vulnerability Areas 

Vision Encoder: a high-dimensional pattern recognizer where adversarial pixel perturbations directly target this stage by exploiting the encoder’s gradients. domain shifts - just with intensity distribution 
Fusion Layer : If the image features are out-of-distribution  the projection layer might produce embedding tokens that are not meaningful to the LLM,   or  accidentally   correspond   to   unrelated   tokens   a steganographic attack where hidden text is present in the image pixels
LLM : all robustness issues like hallucination

## Possible Robustness techniques 

###  Architecture Level

#### Frequency Aware Vision Encoders - 

- **Integrate FM-ViT modules** after each SigLIP self-attention block, Fourier-decomposing attention outputs into low/high frequencies and re-weighting them with learnable coefficients.
    
- **Adjust the input embedding** to accept single-channel CXRs so the FM layers preserve subtle lung textures and soft-tissue details.
    
- **Add a sample-adaptive branch** using DWT-derived statistics (mean/variance of sub-bands) to dynamically tune low/high frequency weights per chest X-ray.
    
- **Fine-tune in three stages**: (1) contrastive tuning on CXR–report pairs, (2) cross-modal projector training to align image and report embeddings, (3) instruction tuning with radiology Q&A/report-generation prompts.
    
- **Benchmark and hyperparameter-tune** on CXR classification and VQA datasets, optimizing frequency-band thresholds for radiographic feature robustness.

#### MoE 

Multiple vision encoders - trained on norma, noisy, and overlays then weght their votes

###  Training Level


** Fine-Tuning of LLM with Noisy Visual Inputs:**  ,when fine-tuning a multimodal model like LLaVA-Med, we use  clean image-text pairs. we can use fine-tuning the language model part  by  feeding it  perturbed images with the correct explanation. 

*Contrastive loss** - train the vision encoder with a contrastive loss where an image and a perturbed version of the same image must produce similar embeddings (invariant representation).


###  Test time or inference time  Level

**Preprocessing filters**  - Filter or compression

**Test time augmentation** - use the multiple version -  ask the model the same
question on the original and some augmented images and then use a  algorithm to see if the answers agree. If they differ significantly, attack or uncertainty and the system could respond with caution

**Uncertainty aware outputs**



Key questions 

1. What evaluation methods (VSF-Med, CARES, RexVQA) best capture robustness in medical AI?
2. How do multimodal prompt attacks (image + text) interact, and how to defend them?
3. Which robustness techniques transfer across datasets and pathologies?
4. What is the trade-off between model complexity and robustness in this domain?
5. Can incorporating medical domain knowledge or even  more modalities enhance robustness
