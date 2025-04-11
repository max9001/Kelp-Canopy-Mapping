**Phase 1: Solidify Foundation & Rigorous Evaluation (Imperative for Paper)**

1.  **Establish and Document the Pretrained ResNet18 Baseline (350x350):**
    *   **Action:** Re-run/confirm the training for the ResNet18 model on the *original, unbalanced 350x350 data*. Ensure the code is clean and the hyperparameters (LR, epochs, optimizer, loss=BCEWithLogits) are documented precisely.
    *   **Why:** This is your current best performing model on the *intended task format*. It serves as the essential baseline against which all other improvements will be measured. You *need* a reproducible baseline for the paper.

2.  **Implement Comprehensive & Consistent Evaluation:**
    *   **Action:** Create a standard evaluation script/process. For *every* model you train (baseline, tiled, augmented, etc.), evaluate it on the *same held-out, original format (350x350), unbalanced test set*.
    *   **Action:** Calculate and report *multiple* metrics:
        *   **IoU (Jaccard Index):** Already doing this - good!
        *   **Pixel Accuracy:** (Be cautious, high due to imbalance, but report it).
        *   **Precision (for Kelp class):** How many pixels predicted as kelp actually are kelp?
        *   **Recall (for Kelp class):** How many actual kelp pixels did the model find?
        *   **F1-Score (for Kelp class):** Harmonic mean of Precision and Recall.
    *   **Why:** You need consistent evaluation to compare methods fairly. Accuracy alone is misleading. Precision/Recall/F1 give insight into performance on the minority (kelp) class. Testing the tiled model on unbalanced data was comparing apples and oranges; this fixes that.

3.  **Re-evaluate the Tiled Model Strategy:**
    *   **Action:** Decide on a *consistent* evaluation for the tiled model:
        *   **Option A (Preferred):** Create a *tiled version* of your original 350x350 *test set*. Train the tiled model on tiled training data, and test it on the *tiled test data*. Report metrics based on the tiles.
        *   **Option B (Harder):** Develop a method to "stitch" the 25x25 predictions back together into a full 350x350 prediction. Evaluate *this reconstructed mask* against the original 350x350 ground truth using the standard metrics.
    *   **Action:** Re-train the tiled model if necessary and evaluate using the chosen consistent method (A or B). Compare its performance (IoU, F1, etc.) directly against the 350x350 ResNet18 baseline evaluated on the 350x350 test set.
    *   **Why:** The ~0.49 IOU is promising, but the previous evaluation wasn't sound. You need to know if tiling *truly* outperforms the baseline under fair conditions.

**Phase 2: Systematic Improvement & Data Handling (High Priority for Better Results & Paper Content)**

4.  **Systematically Apply Imbalance Techniques to ResNet18 (350x350):**
    *   **Action:** Take your working ResNet18 baseline (from step 1) and *systematically* add/test the techniques you tried on the U-Net, *one at a time* initially:
        *   **Loss Functions:** Replace BCEWithLogitsLoss with Dice Loss, Focal Loss, or a combined Dice+BCE loss. Train and evaluate rigorously (step 2).
        *   **Data Augmentation:** Implement standard augmentations (flips, rotations, maybe mild color jitter) for the 350x350 data. Train the ResNet18 baseline *with* augmentation and evaluate.
        *   **Weighted BCE:** Calculate class weights for the 350x350 dataset and use `pos_weight` or `weight` in `BCEWithLogitsLoss`. Train and evaluate.
    *   **Why:** You need to see if these standard techniques, which failed on the simple U-Net, actually help the more powerful pretrained model. This provides crucial experimental results for your paper.

5.  **Data Cleaning & Preprocessing:**
    *   **Action:** **Handle Corrupted Data:** Identify and remove/exclude the image(s) with "weird stripes" (VC433864) from *all* datasets (train/val/test). Document this exclusion.
    *   **Action:** **Address Cloud Cover:** Analyze the impact of heavily clouded images.
        *   *Option 1 (Simple):* Keep them in, assuming the cloud mask (band 5) helps the model ignore these areas.
        *   *Option 2 (Explore):* Train a model variant *excluding* images with > X% cloud cover (using band 5) and see if performance on *less cloudy test images* improves. This could be an interesting analysis point.
    *   **Action:** **Implement Dataset-Wide Normalization:** Calculate the mean and standard deviation for each of the 7 channels across your *entire training set* (350x350 or tiled, whichever you are focusing on). Apply this consistent normalization to all splits (train/val/test) instead of basic `/ 255.0`.
    *   **Why:** Clean data is essential. Normalization improves training stability. Handling clouds explicitly might improve robustness.

6.  **Implement Elevation Mask Post-processing:**
    *   **Action:** After generating a prediction mask (e.g., from your best ResNet18 model), create a post-processing step. Load the corresponding Digital Elevation Model (DEM) band. Set any predicted kelp pixels where the DEM is > 0 (or some small threshold like 0.5 meters, accounting for tides/inaccuracy) back to 0 (no kelp).
    *   **Action:** Evaluate the model *with* and *without* this post-processing step. Report the difference in metrics.
    *   **Why:** This is physically motivated feature engineering. It should directly remove impossible predictions and likely improve precision and IoU. It's a strong point for the paper.

**Phase 3: Refinement & Exploration (Medium Priority for Stronger Paper & Pushing Boundaries)**

7.  **Combine Successful Techniques:**
    *   **Action:** Based on the results from step 4, combine the techniques that showed individual improvement on the ResNet18 baseline (e.g., best loss function + augmentation + dataset normalization). Train this "combined best" model and evaluate.
    *   **Why:** Often, the synergy between techniques yields the best results.

8.  **Hyperparameter Tuning:**
    *   **Action:** For your best performing model configuration (e.g., ResNet18 + Dice Loss + Augmentation), tune key hyperparameters:
        *   Learning Rate (use LR finder if not already done)
        *   Optimizer (try AdamW)
        *   Learning Rate Scheduler (try Cosine Annealing, ReduceLROnPlateau)
        *   Batch Size (as large as fits in memory)
    *   **Why:** Optimizing hyperparameters can provide a significant performance boost.

9.  **Ablation Study:**
    *   **Action:** Take your "combined best" model (from step 7). Train variants where you *remove* one key component at a time (e.g., train without augmentation, train with BCE instead of Dice, train without dataset normalization). Evaluate each variant.
    *   **Why:** This scientifically demonstrates the contribution of each component of your final method. It makes the paper much stronger.

10. **Qualitative Analysis & Visualization:**
    *   **Action:** Generate visual examples of predictions from your baseline and best models on select test images. Include examples where the model does well, where it fails, and where improvements (like the DEM mask) make a difference.
    *   **Why:** Visual results make the paper much more compelling and help illustrate the model's behavior and limitations.

**Phase 4: Advanced Exploration (Lower Priority / Future Work Section)**

11. **Explore More Advanced Augmentations:** Try Elastic Deformations, Grid Distortion, Cutout, Mixup/CutMix specifically for the kelp data.
12. **Explore Different Architectures:** Try a deeper ResNet (ResNet34/50) as the encoder, or a different segmentation architecture like DeepLabV3+.
13. **Post-processing (Morphological Operations):** Apply morphological opening or closing to the predicted masks to remove small noise or fill small holes.
14. **Investigate Tiling Further:** If tiling showed promise, explore different tile sizes or overlap strategies.
15. **Ensembling:** Train multiple models (e.g., with different random seeds or slightly different hyperparameters) and average their predictions.

**Paper Writing Integration:**

*   **Document As You Go:** Keep detailed notes (or use experiment tracking tools like Weights & Biases/MLflow) for *every* experiment: hyperparameters, code version, dataset used, evaluation metrics.
*   **Structure:** Use the standard paper structure (Intro, Related Work (briefly), Data, Methods, Experiments, Results, Discussion, Conclusion).
*   **Focus:** Clearly describe the baseline, the challenges (imbalance), the techniques you applied *systematically*, the evaluation protocol, and the results (quantitative and qualitative). The ablation study and DEM masking will be strong discussion points.