import cv2
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
from liveClassifierTorch import (
    ClassifierPnm,
    readPolarPNMToRGBALive,
    tile_and_cast_data_torch,
    classify_tiles,
    generate_heatmap,
    runSingle,
    log_performance
)
# -------------------------------------------------------------------------------------
def dump_predictions_to_file(preds, filename, header=None):
    """
    Dumps predictions (torch tensor, numpy array, or list)
    to a readable text file for debugging.

    Args:
        preds (list/ndarray/tensor): Predicted class IDs
        filename (str): Output file path
        header (str): Optional header string to include at top
    """

    # Convert to numpy for safety
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    elif isinstance(preds, list):
        preds = np.array(preds)

    # Ensure 1D
    preds = preds.reshape(-1)

    with open(filename, "w") as f:
        if header is not None:
            f.write(f"# {header}\n")
            f.write(f"# Length: {len(preds)}\n\n")

        # Write predictions line-by-line
        for i, p in enumerate(preds):
            f.write(f"{i}: {int(p)}\n")

    print(f"[dump_predictions_to_file] Wrote {len(preds)} entries to {filename}")
# -------------------------------------------------------------------------------------
@torch.no_grad()
def tile_and_cast_selected_tiles_torch(image, selected_indices, tile_size=24, step=2):
    """
    Works like tile_and_cast_data_torch(), but returns ONLY the tiles whose
    flat indices are listed in selected_indices.

    selected_indices : 1D tensor or list of indices into the flattened tile list.
    """

    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # image is H, W, C → convert to C, H, W
    image = image.permute(2, 0, 1)

    C, H, W = image.shape

    # unfold along height and width
    tiles = image.unfold(1, tile_size, step).unfold(2, tile_size, step)
    # tiles shape: (C, nH, nW, tile_size, tile_size)

    nH, nW = tiles.shape[1], tiles.shape[2]

    # rearrange to (nH*nW, tile_size, tile_size, C)
    tiles = tiles.permute(1, 2, 3, 4, 0).contiguous()
    tiles = tiles.view(-1, tile_size, tile_size, C)

    # Cast to float32 (same as old function)
    tiles = tiles.to(torch.float32)

    # Return only the selected tiles
    return tiles[selected_indices]
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
@torch.no_grad()
def classify_selected_tiles(name,
                            model,
                            rgba_image,
                            npTiles,
                            tile_size=64,
                            step=0,
                            chunks=0,
                            thresholdMaxProbability=0.50,
                            forceLowMaxProbToThisClass=None):
    """
    Classify only the tiles in `selected_indices`.

    Works like classify_tiles(), but uses tile_and_cast_selected_tiles_torch()
    and does NOT apply majorityVote because tiles are not in a full grid.
    """

    start = time.time() 

    channels = 4
    if npTiles.shape[1:] != (channels, tile_size, tile_size):
        raise ValueError(f"Expected {channels}x{tile_size}x{tile_size}, got {npTiles.shape[1:]}")

    softmax = torch.nn.Softmax(dim=1)
    low_activations = 0

    # --- Inference ---
    if chunks == 0:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
            preds = model(npTiles)

        probs = softmax(preds)
        max_probs, predictions = torch.max(probs, dim=1)

        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            mask = max_probs < thresholdMaxProbability
            low_activations += mask.sum().item()
            predictions[mask] = forceLowMaxProbToThisClass

    else:
        preds_list = []
        for chunk in npTiles.chunk(chunks):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                preds_list.append(model(chunk))

        preds = torch.cat(preds_list)
        probs = softmax(preds)
        max_probs, predictions = torch.max(probs, dim=1)

        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            mask = max_probs < thresholdMaxProbability
            low_activations += mask.sum().item()
            predictions[mask] = forceLowMaxProbToThisClass

    predictions = predictions.cpu().numpy()
    confidences = max_probs.cpu().numpy()

    print(f"Low-confidence tiles reassigned: {low_activations}")
    print(f"classify_selected_tiles ({name}) done in {time.time() - start:.2f}s, on {len(predictions)} selected tiles")

    return predictions, confidences
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
@torch.no_grad()
def majority_vote_final(predictions, confidences, tilesW, tilesH, window_size=3):
    """
    Apply 2D majority voting (spatial smoothing) to FINAL tile predictions,
    and propagate confidences for the winning class in each window.

    Parameters
    ----------
    predictions : 1D numpy or torch array of class IDs
        Should have length == tilesW * tilesH
    confidences : 1D numpy or torch array of float confidences (same length)
    tilesW : int
        Number of tiles horizontally
    tilesH : int
        Number of tiles vertically
    window_size : int
        Must be odd (3, 5, 7...). Typical = 3

    Returns
    -------
    smoothed_predictions : 1D numpy array of smoothed predictions
    smoothed_confidences : 1D numpy array of confidences aligned to smoothed predictions
    """

    # Convert to tensors
    preds = torch.tensor(predictions, dtype=torch.long)    if not isinstance(predictions, torch.Tensor) else predictions.long()
    confs = torch.tensor(confidences, dtype=torch.float32) if not isinstance(confidences, torch.Tensor) else confidences.float()

    expectedDimensionality = tilesW * tilesH 
    assert preds.numel() == expectedDimensionality, str("predictions length mismatch, expected %u, encountered %u " % (expectedDimensionality,preds.numel()) )
    assert confs.numel() == expectedDimensionality, str("confidences length mismatch, expected %u, encountered %u " % (expectedDimensionality,confs.numel()))

    # Reshape into 2D grids
    grid_preds = preds.view(tilesH, tilesW)
    grid_confs = confs.view(tilesH, tilesW)

    pad = window_size // 2
    padded_preds = torch.nn.functional.pad(grid_preds, (pad, pad, pad, pad), mode='constant', value=-1)
    padded_confs = torch.nn.functional.pad(grid_confs, (pad, pad, pad, pad), mode='constant', value=0.0)

    out_preds = grid_preds.clone()
    out_confs = grid_confs.clone()

    for y in range(tilesH):
        for x in range(tilesW):
            window_preds = padded_preds[y:y+window_size, x:x+window_size].flatten()
            window_confs = padded_confs[y:y+window_size, x:x+window_size].flatten()

            # Majority vote
            vals, counts = torch.unique(window_preds, return_counts=True)
            majority_class = vals[counts.argmax()]

            out_preds[y, x] = majority_class

            # Confidence: mean of confidences of tiles that match the majority class
            mask = window_preds == majority_class
            if mask.any():
                out_confs[y, x] = window_confs[mask].mean()
            else:
                out_confs[y, x] = 0.0

    return out_preds.flatten().numpy(), out_confs.flatten().numpy()

# -------------------------------------------------------------------------------------
# Async multi-model helpers
# -------------------------------------------------------------------------------------
def run_models_async(models, x):
    """
    Runs a list of models asynchronously using CUDA streams.
    Each model can have a different architecture.
    Returns a list of their outputs.
    """
    assert torch.cuda.is_available(), "CUDA required for async inference"
    device = next(models[0].parameters()).device

    streams = [torch.cuda.Stream(device=device) for _ in models]
    results = [None] * len(models)

    for i, (model, stream) in enumerate(zip(models, streams)):
        with torch.cuda.stream(stream):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                results[i] = model(x.clone())  # clone avoids memory sharing

    torch.cuda.synchronize()
    return results
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class MergedEnsemble(nn.Module):
    """Wraps several models to produce parallel outputs."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList([torch.compile(m).to('cuda') for m in models])
        #self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        if isinstance(outputs[0], torch.Tensor):
            try:
                return torch.stack(outputs)  # (num_models, batch, ...)
            except RuntimeError:
                return torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        return outputs
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def parallel_classify_tiles(classifiers, rgba_image, tile_size, step, majorityVote=False, max_workers=None):
    """
    Runs classify_tiles() for each model in parallel threads.
    Returns a list of predictions (torch tensors, GPU-resident).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [None] * len(classifiers)

    with ThreadPoolExecutor(max_workers=max_workers or len(classifiers)) as executor:
        futures = {
            executor.submit(
                classify_tiles,
                clf.model,
                rgba_image,
                tile_size=tile_size,
                step=step,
                majorityVote=majorityVote,
            ): i
            for i, clf in enumerate(classifiers)
        }

        for future in as_completed(futures):
            i = futures[future]
            preds = future.result()
            results[i] = torch.tensor(preds, device=classifiers[i].device)

    return results
# -------------------------------------------------------------------------------------
# Ensemble Classifier Implementation
# -------------------------------------------------------------------------------------
class EnsembleClassifierPnm:
    def __init__(self, initial_model_cfg, model_cfg_list, tile_size=48, step=16):
        """
        initial_model_cfg: (model_path, cfg_path) for the first classifier
        model_cfg_list: list of (model_path, cfg_path) for ensemble models
        """
        assert len(model_cfg_list) > 0, "You must provide at least one ensemble model."

        self.maxProbabilityThreshold = 0.0
        self.tile_size = tile_size
        self.step = step
        self.hz = 0.0

        # --- Load the first classifier (pre-filter) ---
        init_model_path, init_cfg_path = initial_model_cfg
        self.first_clf = ClassifierPnm(
                                       model_path=init_model_path,
                                       cfg_path=init_cfg_path,
                                       tile_size=tile_size,
                                       step=step,
                                      )

        self.name = "EnsembleClassifier"
        # --- Load the ensemble classifiers ---
        self.classifiers = [
                             ClassifierPnm(model_path=mp, cfg_path=cp, tile_size=tile_size, step=step)
                             for mp, cp in model_cfg_list
                           ]

        # --- Common definitions from the first model ---
        self.classes      = self.classifiers[0].classes
        self.class_colors = self.classifiers[0].class_colors
        self.device       = self.classifiers[0].device

        # Precompute clean class ID once
        def find_clean_id(cls_list):
            for i, c in enumerate(cls_list):
                if c.lower() in ("class_clean", "clean"):
                    return i
            return None

        self.firstCleanClassID = find_clean_id(self.first_clf.classes)
        self.cleanClassID = find_clean_id(self.classes)

        if self.firstCleanClassID is None or self.cleanClassID is None:
            raise ValueError("Could not find 'class_clean' in model class lists")

        # Precompute color tensors for reuse
        self.class_id_to_color = [torch.tensor(c, dtype=torch.uint8) for c in self.class_colors]

        # --- Build merged model for optional joint GPU execution ---
        ensemble_models = [clf.model for clf in self.classifiers]
        self.merged_model = MergedEnsemble(ensemble_models).to(self.device)
        if torch.cuda.is_available():
            torch.compile(self.merged_model)

        print(f"Initialized EnsembleClassifierPnm with 1 initial + {len(self.classifiers)} ensemble models")
        print("Clean class ID:", self.cleanClassID)





    # -------------------------------------------------------------------------
    @torch.no_grad()
    def ensemble_vote_and_answer_for_all_tiles(self, num_models, global_number_of_tiles, all_predictions, all_confidences, non_clean_indices, cleanClassID, strict=True):
        """
        Perform ensemble voting on non-clean tiles and assign confidences.

        Parameters
        ----------
        all_predictions : torch.Tensor [M, N] 
            Predicted class IDs from M models for N tiles
        all_confidences : torch.Tensor [M, N] 
            Max probabilities from M models for N tiles
        non_clean_indices : list[int] 
            Indices of non-clean tiles in the global grid
        cleanClassID : int 
            ID of the clean class
        strict : bool
            If True, enforce strict majority vote to revert to clean class

        Returns
        -------
        final_predictions : torch.Tensor, shape [total_tiles]
        final_confidences : torch.Tensor, shape [total_tiles]
        """
        #
        num_models_est, num_tiles = all_predictions.shape

        #total_tiles = max(non_clean_indices.max().item() + 1, num_tiles) # ?
        #print("all_predictions.shape ",all_predictions)
        #print("Num_tiles ( from all_predictions ) ",num_tiles,"  / total_tiles ",total_tiles )
        #print("Correct number of global tiles ",global_number_of_tiles)

        total_tiles = global_number_of_tiles

        assert num_models_est == num_models, str("num_models mismatch, expected %u, encountered %u " % (num_models,num_models_est) )
    

        # Initialize full-length predictions and confidences
        final_predictions = torch.full((total_tiles,),  fill_value=cleanClassID, dtype=torch.int32, device=all_predictions.device)
        final_confidences = torch.zeros((total_tiles,), dtype=torch.float32, device=all_predictions.device)

        # Decide whether predictions are in local (selected) indexing or global
        use_local_index = (num_tiles == len(non_clean_indices))
        #use_local_index = True

        if use_local_index:
            for local_j, global_idx in enumerate(non_clean_indices):
                votes = all_predictions[:, local_j].tolist()
                confs = all_confidences[:, local_j]
                clean_votes = votes.count(cleanClassID)

                if strict and clean_votes > num_models / 2:
                    voted_class = cleanClassID
                    voted_conf  = confs[votes.index(cleanClassID)].item()
                else:
                    voted_class = max(set(votes), key=votes.count)
                    voted_conf  = confs[[i for i, v in enumerate(votes) if v == voted_class][0]].item()

                final_predictions[global_idx] = voted_class
                final_confidences[global_idx] = voted_conf

        else:
            for global_idx in non_clean_indices:
                votes = all_predictions[:, global_idx].tolist()
                confs = all_confidences[:, global_idx]
                clean_votes = votes.count(cleanClassID)

                if strict and clean_votes > num_models / 2:
                    voted_class = cleanClassID
                    voted_conf  = confs[votes.index(cleanClassID)].item()
                else:
                    voted_class = max(set(votes), key=votes.count)
                    voted_conf  = confs[[i for i, v in enumerate(votes) if v == voted_class][0]].item()

                final_predictions[global_idx] = voted_class
                final_confidences[global_idx] = voted_conf

        return final_predictions, final_confidences


    # -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, image, majorityVote=False, legend=True, strict=True, parallel=False, multimodel=True , debugExecuteSecondStage=False, log=True):
        """
        Step 1: Run initial classifier
        Step 2: Identify non-clean tiles
        Step 3: Run ensemble classifiers on those tiles
        Step 4: Vote on non-clean predictions
        Step 5: Produce final heatmap
        """
        start = time.time()

        # --- Step 1: Run first classifier (prefilter) ---
        init_clf = self.first_clf

        # --- Prepare image tensor ---
        rgba_image = readPolarPNMToRGBALive(image)
        rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
        rgba_image = (rgba_image.astype("float32") / 255.0)
        rgba_image = torch.as_tensor(rgba_image, device=self.device, dtype=torch.float32)

        # --- Step 2: Get predictions from first binary (clean/non-clean) model ---
        base_preds, base_confidences = classify_tiles(
                                                      init_clf.model,
                                                      rgba_image,
                                                      tile_size=self.tile_size,
                                                      step=self.step,
                                                      majorityVote=majorityVote,
                                                      thresholdMaxProbability=self.maxProbabilityThreshold,
                                                      forceLowMaxProbToThisClass=self.firstCleanClassID,
                                                     )
        #base_preds is filled with 0 and 1 predictions, self.firstCleanClassID points to wether 0 or 1 is the clean class 
        #dump_predictions_to_file(base_preds, "base_predictions.txt", header="Base ensemble-voted predictions") #<= debug

        #Base predictions and their confidences are now extracted
        base_preds       = torch.tensor(base_preds, device=self.device, dtype=torch.int32)
        base_confidences = torch.tensor(base_confidences, device=self.device, dtype=torch.float32)


        # --- Identify non-clean tiles ---
        non_clean_indices = (base_preds != self.firstCleanClassID).nonzero(as_tuple=True)[0]
        if len(non_clean_indices) == 0:
            print("All tiles are clean — no ensemble voting needed!")
            final_predictions = base_preds
            final_confidences = base_confidences
        else:
            print(f"{len(non_clean_indices)} non-clean tiles for ensemble voting")


            # --- Step 3: Ensemble inference ---
            #First of all we need to cast the selected tiles to Torch, so that they are ready to be consumed
            if (debugExecuteSecondStage):
                #Execute second stage regardless of first..
                npTiles = tile_and_cast_data_torch(rgba_image, tile_size=self.tile_size, step=self.step)
            else:
                npTiles = tile_and_cast_selected_tiles_torch(rgba_image,non_clean_indices, tile_size=self.tile_size, step=self.step)
            npTiles = npTiles.to(torch.float32).permute(0, 3, 1, 2).contiguous().to(self.device)


            #multimodel=False
            all_predictions = None
            preds_list = []
            conf_list  = []


            #===========================================================================================
            #       The following is the same thing with 3 different optimization attempts..
            #     Serial execution of the nets is the fallback especially on low VRAM machines
            #===========================================================================================
            if multimodel:
                print("Running ensemble via async CUDA streams")
                outputs = run_models_async([clf.model for clf in self.classifiers], npTiles)
                for o in outputs:
                     probs = torch.nn.functional.softmax(o, dim=1)
                     max_probs, predictions = torch.max(probs, dim=1)
                     preds_list.append(predictions)
                     conf_list.append(max_probs)
            elif parallel:
            #===========================================================================================
                print("Running ensemble via CPU thread pool ( This runs all tiles, not just selected btw ) ")
                ensemble_results = parallel_classify_tiles(self.classifiers, rgba_image, self.tile_size, self.step, majorityVote)
                for preds, confs in ensemble_results:
                     preds_list.append(torch.tensor(preds, device=self.device))
                     conf_list.append(torch.tensor(confs, device=self.device))
            else:
            #===========================================================================================
                print("Running ensemble serially")
                for clf in self.classifiers:
                    if (debugExecuteSecondStage):
                       preds, confs = classify_tiles(
                                                     clf.model,
                                                     rgba_image,
                                                     tile_size=self.tile_size,
                                                     step=self.step,
                                                     majorityVote=majorityVote,
                                                     thresholdMaxProbability=self.maxProbabilityThreshold,
                                                     forceLowMaxProbToThisClass=self.cleanClassID,
                                                    )
                    else:
                       preds, confs = classify_selected_tiles(
                                                              clf.name,
                                                              clf.model,
                                                              rgba_image,
                                                              npTiles,
                                                              tile_size=self.tile_size,
                                                              step=self.step,
                                                              thresholdMaxProbability=self.maxProbabilityThreshold,
                                                              forceLowMaxProbToThisClass=self.cleanClassID
                                                             )
                    #---------------------------------------------------------
                    preds_list.append(torch.tensor(preds, device=self.device))
                    conf_list.append(torch.tensor(confs, device=self.device))
            #===========================================================================================
            all_predictions = torch.stack(preds_list)  
            all_confidences = torch.stack(conf_list)      # [M, N]


        # --- Step 4: Voting among ensemble ---
        # IMPORTANT: base_preds is in the FIRST model's label space (binary),
        # while the ensemble uses self.classes with self.cleanClassID.
        # So we must express final_predictions entirely in the ensemble label space.

        # Start by assuming everything is CLEAN in the ensemble space
        final_predictions = torch.full_like(base_preds, fill_value=self.cleanClassID)
        final_confidences = base_confidences.clone()


        #ensemble_vote_and_answer_for_all_tiles should answer for ALL tiles both selected (non-clean) and non-selected (clean) ones!
        if len(non_clean_indices) > 0:
                    final_predictions, final_confidences = self.ensemble_vote_and_answer_for_all_tiles(len(self.classifiers), final_predictions.numel(), all_predictions, all_confidences, non_clean_indices, self.cleanClassID, strict=strict)

        # Convert final predictions now that indexing is correct
        final_predictions = final_predictions.cpu().numpy()
        final_confidences = final_confidences.cpu().numpy()

        if (majorityVote):
           # Compute tile grid dimensions
           height, width, _ = rgba_image.shape
           tilesW = (width  - self.tile_size) // self.step 
           tilesH = (height - self.tile_size) // self.step 

           # Apply majority vote smoothing
           final_predictions, final_confidences = majority_vote_final(final_predictions, final_confidences, tilesW, tilesH, window_size=3)

        # final_predictions should now be mostly 'self.cleanClassID' (e.g. 7),
        # with other ensemble class IDs only on non-clean tiles.
        #dump_predictions_to_file(final_predictions, "final_predictions.txt",header="Final ensemble-voted predictions")
        #global_votes = global_votes.cpu().numpy()  # (optional debugging)

        # --- Step 5: Generate heatmap using final voted results ---
        heatmap, occupancy, responses = generate_heatmap(
                                                         final_predictions,
                                                         final_confidences,
                                                         self.classes,
                                                         self.class_id_to_color,
                                                         self.cleanClassID,
                                                         rgba_image * 255.0,
                                                         tile_size=self.tile_size,
                                                         step=self.step,
                                                        )

        if legend:
            heatmap = self.classifiers[0].add_legend(heatmap)

        self.hz = 1 / (time.time() - start + 1e-4)

 
        if (log):
          runid="ensemble"
          if parallel:
             runid="%s-parallel" % runid
          if multimodel:
             runid="%s-multimodel" % runid
          log_performance("perf.csv", runid, self.step, self.tile_size, majorityVote, self.maxProbabilityThreshold, len(non_clean_indices), self.hz)
        return heatmap, occupancy, responses

