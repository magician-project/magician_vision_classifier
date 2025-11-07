import cv2
import time
import numpy as np
import pytorch_lightning as pl
import torch

from liveClassifierTorch import ClassifierPnm, runSingle, readPolarPNMToRGBALive, tile_and_cast_data_torch, classify_tiles, generate_heatmap


def parallel_classify_tiles(classifiers, rgba_image, tile_size, step, majorityVote=False, max_workers=None):
    """
    Runs classify_tiles() for each model in parallel.
    Returns a list of predictions (one tensor per model).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [None] * len(classifiers)
    with ThreadPoolExecutor(max_workers=max_workers or len(classifiers)) as executor:
        futures = {
            executor.submit(
                classify_tiles, clf.model, rgba_image,
                tile_size=tile_size, step=step,
                majorityVote=majorityVote
            ): i
            for i, clf in enumerate(classifiers)
        }

        for future in as_completed(futures):
            i = futures[future]
            preds = future.result()
            results[i] = torch.tensor(preds).to(classifiers[i].device)

    return results


class EnsembleClassifierPnm:
 def __init__(self, initial_model_cfg, model_cfg_list, tile_size=48, step=16):
        """
        initial_model_cfg: (model_path, cfg_path) for the first classifier
        model_cfg_list:    list of (model_path, cfg_path) for ensemble models
        """
        assert len(model_cfg_list) > 0, "You must provide at least one ensemble (model_path, cfg_path) pair"
 
        self.maxProbabilityThreshold = 0.0
        self.tile_size = tile_size
        self.step = step
        self.hz = 0.0

        # --- Load the first classifier (pre-filter) ---
        init_model_path, init_cfg_path = initial_model_cfg
        self.first_clf = ClassifierPnm(model_path=init_model_path, 
                                       cfg_path=init_cfg_path,
                                       tile_size=tile_size, 
                                       step=step)

        # --- Load the ensemble classifiers ---
        self.classifiers = []
        for model_path, cfg_path in model_cfg_list:
            clf = ClassifierPnm(model_path=model_path, cfg_path=cfg_path,
                                tile_size=tile_size, step=step)
            self.classifiers.append(clf)

        # --- Common class/color definitions from the first ensemble model ---
        self.classes = self.classifiers[0].classes
        self.class_colors = self.classifiers[0].class_colors
        self.device = self.classifiers[0].device


        # --- Find first model clean class ID ---
        self.firstCleanClassID = None
        for i, c in enumerate(self.first_clf.classes):
            if c in ("class_clean", "Clean"):
                self.firstCleanClassID = i
                break
        if self.firstCleanClassID is None:
            raise ValueError("No 'class_clean' or 'Clean' class found in first model")

        # --- Find shared clean class ID ---
        self.cleanClassID = None
        for i, c in enumerate(self.classes):
            if c in ("class_clean", "Clean"):
                self.cleanClassID = i
                break
        if self.cleanClassID is None:
            raise ValueError("No 'class_clean' or 'Clean' class found in ensemble model classes")

        print(f"Initialized EnsembleClassifierPnm with 1 initial model + {len(self.classifiers)} ensemble models")
        print("Clean class ID:", self.cleanClassID)

 @torch.no_grad()
 def forward(self, image, majorityVote=False, legend=True, strict=True, parallel=False):
    """
    Step 1: Run the initial classifier on the image.
    Step 2: Identify non-clean tiles from initial classifier.
    Step 3: Run ensemble classifiers only on those tiles.
    Step 4: Perform majority vote among ensemble models for non-clean tiles.
            If strict=True, majority 'clean' votes override to clean.
    """
    start = time.time()

    # --- Step 1: Run first classifier (pre-filter) ---
    init_clf = self.first_clf
    heatmap, occupancy, responses = runSingle(
        image, init_clf.model, init_clf.device, init_clf.classes, init_clf.class_colors,
        init_clf.tile_size, init_clf.step, majorityVote=majorityVote
    )

    # Convert image to torch tensor
    rgba_image = readPolarPNMToRGBALive(image)
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
    rgba_image = (rgba_image.astype('float32') / 255.0)
    rgba_image = torch.tensor(rgba_image).float().to(self.device, dtype=torch.float32)

    # Tile image for reuse
    npTiles = tile_and_cast_data_torch(rgba_image, tile_size=self.tile_size,
                                       step=self.step).to(torch.float32).permute(0, 3, 1, 2).contiguous().to(self.device)

    # --- Step 2: Get base predictions from first model ---
    base_preds = classify_tiles(init_clf.model, rgba_image,
                                tile_size=self.tile_size, step=self.step,
                                majorityVote=majorityVote,
                                thresholdMaxProbability=self.maxProbabilityThreshold,
                                forceLowMaxProbToThisClass=self.firstCleanClassID)
    base_preds = torch.tensor(base_preds).to(self.device)

    # Find indices of tiles that are not 'clean'
    non_clean_indices = (base_preds != self.cleanClassID).nonzero(as_tuple=True)[0]

    if len(non_clean_indices) == 0:
        print("All tiles classified as clean — skipping ensemble voting.")
        final_predictions = base_preds
    else:

        # --- Step 3: Run ensemble classifiers on all tiles ---
        all_predictions = []
        if (parallel):
          print(f"Parallel running ensemble voting for {len(non_clean_indices)} non-clean tiles")
          all_predictions = parallel_classify_tiles(
                                                     self.classifiers, rgba_image,
                                                     tile_size=self.tile_size, step=self.step,
                                                     majorityVote=majorityVote )
        else:
          print(f"Running ensemble voting for {len(non_clean_indices)} non-clean tiles")
          for clf in self.classifiers:
            preds = classify_tiles(clf.model, rgba_image,
                                   tile_size=self.tile_size, step=self.step,
                                   majorityVote=majorityVote,
                                   thresholdMaxProbability=self.maxProbabilityThreshold,
                                   forceLowMaxProbToThisClass=self.cleanClassID)
            preds = torch.tensor(preds).to(self.device)
            all_predictions.append(preds)

        # Stack predictions: shape = (num_models, num_tiles)
        all_predictions = torch.stack(all_predictions)

        # --- Step 4: Perform voting only on non-clean tiles ---
        final_predictions = base_preds.clone()
        num_models = all_predictions.shape[0]

        for idx in non_clean_indices:
            votes = all_predictions[:, idx].tolist()
            clean_votes = votes.count(self.cleanClassID)

            if strict:
                # In strict mode: if majority of models say "clean", mark as clean
                if clean_votes > num_models / 2:
                    voted_class = self.cleanClassID
                else:
                    # Otherwise use normal majority voting among all predictions
                    voted_class = max(set(votes), key=votes.count)
            else:
                # Normal mode: plain majority voting
                voted_class = max(set(votes), key=votes.count)

            final_predictions[idx] = voted_class

    # Convert predictions to numpy for visualization
    final_predictions = final_predictions.cpu().numpy()

    # --- Step 5: Recompute heatmap using ensemble output ---
    heatmap, occupancy, responses = generate_heatmap(
        final_predictions, self.classes, self.class_colors, rgba_image * 255.0,
        tile_size=self.tile_size, step=self.step
    )

    if legend:
        heatmap = self.classifiers[0].add_legend(heatmap)

    seconds = time.time() - start
    self.hz = 1 / (seconds + 1e-4)

    return heatmap, occupancy, responses


