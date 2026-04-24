import cv2
import time
import numpy as np
import torch
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

    # Keep as uint8 — normalisation (/255) happens inside the model on the GPU.
    tiles = tiles.to(torch.uint8)

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
                            forceLowMaxProbToThisClass=None,
                            return_torch=False):
    """
    Classify only the tiles in npTiles (already selected subset).

    return_torch : if True, return GPU tensors directly (avoids GPU→CPU copy
                  when the caller will immediately wrap back to tensor).
    """

    start = time.time()

    channels = 4
    if npTiles.shape[1:] != (channels, tile_size, tile_size):
        raise ValueError(f"Expected {channels}x{tile_size}x{tile_size}, got {npTiles.shape[1:]}")

    low_activations = 0

    # --- Inference ---
    if chunks == 0:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            preds = model(npTiles)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            mask = max_probs < thresholdMaxProbability
            low_activations += mask.sum().item()
            predictions[mask] = forceLowMaxProbToThisClass
    else:
        preds_list = []
        for chunk in npTiles.chunk(chunks):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                preds_list.append(model(chunk))
        preds = torch.cat(preds_list)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            mask = max_probs < thresholdMaxProbability
            low_activations += mask.sum().item()
            predictions[mask] = forceLowMaxProbToThisClass

    print(f"Low-confidence tiles reassigned: {low_activations}")
    print(f"classify_selected_tiles ({name}) done in {time.time() - start:.2f}s, on {len(predictions)} selected tiles")

    if return_torch:
        return predictions, max_probs
    return predictions.cpu().numpy(), max_probs.cpu().numpy()
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
@torch.no_grad()
def majority_vote_final(predictions, confidences, tilesW, tilesH, window_size=3):
    """
    Vectorised 2-D majority voting using unfold + torch.mode (no Python loops).
    Confidences are smoothed by mean-pooling within each window.

    Returns 1-D numpy arrays (smoothed_predictions, smoothed_confidences).
    """
    import torch.nn.functional as F

    preds = (torch.as_tensor(predictions, dtype=torch.long)
             if not isinstance(predictions, torch.Tensor) else predictions.long().cpu())
    confs = (torch.as_tensor(confidences, dtype=torch.float32)
             if not isinstance(confidences, torch.Tensor) else confidences.float().cpu())

    expected = tilesW * tilesH
    n = preds.numel()
    if n < expected:
        fill_cls  = int(torch.mode(preds).values.item()) if n > 0 else 0
        fill_conf = float(confs.mean().item())           if n > 0 else 0.0
        preds = F.pad(preds, (0, expected - n), value=fill_cls)
        confs = F.pad(confs, (0, expected - n), value=fill_conf)
        print(f"[majority_vote_final] WARNING: padded from {n} to {expected}")
    elif n > expected:
        preds = preds[:expected]
        confs = confs[:expected]
        print(f"[majority_vote_final] WARNING: truncated from {n} to {expected}")

    pad = window_size // 2

    # --- Predictions: unfold → mode ---
    grid_p = preds.view(1, 1, tilesH, tilesW).float()
    padded_p = F.pad(grid_p, (pad, pad, pad, pad), mode='replicate')
    windows_p = padded_p.unfold(2, window_size, 1).unfold(3, window_size, 1)
    # shape: [1, 1, tilesH, tilesW, window_size, window_size]
    flat_p = windows_p.contiguous().view(tilesH, tilesW, -1).long()
    smooth_preds = torch.mode(flat_p, dim=2).values   # [tilesH, tilesW]

    # --- Confidences: unfold → mean ---
    grid_c = confs.view(1, 1, tilesH, tilesW)
    padded_c = F.pad(grid_c, (pad, pad, pad, pad), mode='replicate')
    windows_c = padded_c.unfold(2, window_size, 1).unfold(3, window_size, 1)
    flat_c = windows_c.contiguous().view(tilesH, tilesW, -1)
    smooth_confs = flat_c.mean(dim=2)                 # [tilesH, tilesW]

    return smooth_preds.flatten().numpy(), smooth_confs.flatten().numpy()

# -------------------------------------------------------------------------------------
# Async multi-model helpers
# -------------------------------------------------------------------------------------
def run_models_async(models, x):
    """
    Run all models in *models* concurrently on separate CUDA streams.

    Input *x* is converted to channels_last memory format (zero-copy stride update)
    before fanning out to streams. Uses FP16 autocast. Requires CUDA.
    """
    assert torch.cuda.is_available(), "CUDA required for async inference"
    device = next(models[0].parameters()).device

    streams = [torch.cuda.Stream(device=device) for _ in models]
    results = [None] * len(models)

    # Convert input once to channels-last before fanning out to streams.
    # Each model was loaded with .to(memory_format=torch.channels_last) so the
    # layout must match here; the conversion is zero-copy (stride update only).
    x = x.to(memory_format=torch.channels_last)

    for i, (model, stream) in enumerate(zip(models, streams)):
        with torch.cuda.stream(stream):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                results[i] = model(x)   # read-only input, no clone needed

    torch.cuda.synchronize()
    return results
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def parallel_classify_tiles(classifiers, rgba_image, tile_size, step, majorityVote=False, max_workers=None):
    """
    Run classify_tiles() for each model in parallel CPU threads.

    Returns a list of GPU-resident prediction tensors, one per classifier.
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
    def __init__(self, initial_model_cfg, model_cfg_list, tile_size=48, step=16,
                 min_hz=0.0, benchmark_tiles=512):
        """
        initial_model_cfg: (model_path, cfg_path) for the first classifier
        model_cfg_list:    list of (model_path, cfg_path) for ensemble models
        min_hz:            drop any ensemble model whose single-forward-pass benchmark
                           is below this threshold (0.0 = keep all, default)
        benchmark_tiles:   batch size used for the Hz benchmark (default 512)
        """
        assert len(model_cfg_list) > 0, "You must provide at least one ensemble model."

        self.maxProbabilityThreshold = 0.0
        self.tile_size = tile_size
        self.step = step
        self.hz = 0.0
        self.model_perf = {}   # name → Hz, updated each forward() call

        # --- Load the first classifier (pre-filter) ---
        init_model_path, init_cfg_path = initial_model_cfg
        self.first_clf = ClassifierPnm(
                                       model_path=init_model_path,
                                       cfg_path=init_cfg_path,
                                       tile_size=tile_size,
                                       step=step,
                                      )

        self.name = "EnsembleClassifier"
        self._benchmark_tiles = benchmark_tiles

        # --- Load ALL ensemble classifiers (kept in full for re-filtering) ---
        self._all_classifiers = [
                                   ClassifierPnm(model_path=mp, cfg_path=cp, tile_size=tile_size, step=step)
                                   for mp, cp in model_cfg_list
                                 ]
        self.classifiers = list(self._all_classifiers)   # active subset

        # --- Common definitions from the first model ---
        self.classes      = self._all_classifiers[0].classes
        self.class_colors = self._all_classifiers[0].class_colors
        self.device       = self._all_classifiers[0].device

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

        # --- Apply initial min_hz filter (benchmarks lazily) ---
        self.apply_min_hz(min_hz)

        print(f"Initialized EnsembleClassifierPnm with 1 initial + {len(self.classifiers)} ensemble models")
        print("Clean class ID:", self.cleanClassID)

    @staticmethod
    def _benchmark_clf(model, n_tiles, tile_size, device):
        """
        Benchmark a single classifier's throughput.

        Returns Hz (forward passes per second) for a batch of *n_tiles* dummy
        uint8 tiles. Synchronizes CUDA before/after timing if on GPU.
        """
        # Use uint8 dummy input to match the live-pipeline data format.
        dummy = torch.randint(0, 256, (n_tiles, 4, tile_size, tile_size),
                              dtype=torch.uint8, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        return 1.0 / (time.time() - t0 + 1e-9)

    def apply_min_hz(self, min_hz):
        """
        Re-filter the active classifier list using a minimum throughput threshold.

        Benchmarks any model not yet measured (cached in self.model_perf), then
        keeps only those meeting or exceeding *min_hz* Hz. Safe to call at any time
        without reloading weights. A value of 0.0 or negative keeps all models.
        """
        self.min_hz = min_hz

        # Benchmark any model not yet measured
        for clf in self._all_classifiers:
            if clf.name not in self.model_perf or self.model_perf[clf.name] == 0.0:
                hz = self._benchmark_clf(clf.model, self._benchmark_tiles,
                                         self.tile_size, self.device)
                self.model_perf[clf.name] = hz
                print(f"[Ensemble] Benchmarked {clf.name}: {hz:.2f} Hz")

        # Filter
        if min_hz > 0.0:
            kept = [clf for clf in self._all_classifiers
                    if self.model_perf.get(clf.name, 0.0) >= min_hz]
            print(f"[Ensemble] apply_min_hz({min_hz:.1f}): "
                  f"keeping {len(kept)}/{len(self._all_classifiers)} models")
        else:
            kept = list(self._all_classifiers)
            print(f"[Ensemble] apply_min_hz(0): keeping all {len(kept)} models")

        self.classifiers = kept

    def print_perf(self):
        """Print a formatted per-model throughput table with ASCII bar charts."""
        if not self.model_perf:
            print("[Ensemble] No performance data yet.")
            return
        n_tiles   = getattr(self, "_last_tile_count", 0)
        elapsed   = getattr(self, "_last_elapsed",    1e-4)
        tiles_sec = n_tiles / elapsed
        print("\n" + "=" * 68)
        print(f" Ensemble per-model performance  "
              f"(ensemble Hz: {self.hz:.2f}  |  {n_tiles} tiles  |  {tiles_sec:.0f} tiles/sec)")
        print("=" * 68)
        for name, hz in sorted(self.model_perf.items(), key=lambda kv: -kv[1]):
            bar = "#" * min(40, max(1, int(hz * 2)))
            print(f"  {name:<45}  {hz:6.2f} Hz  {bar}")
        print("=" * 68 + "\n")





    # -------------------------------------------------------------------------
    @torch.no_grad()
    def ensemble_vote_and_answer_for_all_tiles(self, num_models, global_number_of_tiles, all_predictions, all_confidences, non_clean_indices, cleanClassID, strict=True):
        """
        Vectorised ensemble voting across models for non-clean tile positions.

        Performs majority vote across the model dimension, optionally enforces a
        clean-class override in strict mode, and computes per-tile confidence as
        the mean confidence of models agreeing with the winner. Scatters results
        into full-grid output tensors at non-clean tile positions.
        """
        dev = all_predictions.device

        # Initialize full grid as clean
        final_predictions = torch.full((global_number_of_tiles,), fill_value=cleanClassID,
                                       dtype=torch.int32,   device=dev)
        final_confidences = torch.zeros((global_number_of_tiles,), dtype=torch.float32, device=dev)

        if all_predictions.numel() == 0:
            return final_predictions, final_confidences

        # --- Majority vote across model dimension [M, N_selected] → [N_selected] ---
        all_predictions_long = all_predictions.long()
        voted_class = torch.mode(all_predictions_long, dim=0).values   # [N_selected]

        # Strict mode: if more than half the models voted clean, revert to clean
        if strict:
            clean_votes = (all_predictions_long == cleanClassID).sum(dim=0)  # [N_selected]
            majority_is_clean = clean_votes > (num_models / 2)
            voted_class = torch.where(majority_is_clean,
                                      torch.tensor(cleanClassID, device=dev, dtype=torch.long),
                                      voted_class)

        # --- Confidence: mean of models that agree with the winner ---
        agree_mask = (all_predictions_long == voted_class.unsqueeze(0)).float()  # [M, N_selected]
        voted_conf = (agree_mask * all_confidences).sum(dim=0) / agree_mask.sum(dim=0).clamp(min=1.0)

        # --- Scatter into full-grid output ---
        final_predictions[non_clean_indices] = voted_class.to(torch.int32)
        final_confidences[non_clean_indices] = voted_conf

        return final_predictions, final_confidences


    # -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, image, majorityVote=False, legend=True, strict=True, parallel=False, multimodel=True , debugExecuteSecondStage=False, log=True):
        """
        Two-stage ensemble inference: prefilter (binary clean/non-clean) then ensemble voting.

        Pipeline:
          1. Run the first binary classifier on the full image to identify non-clean tiles.
          2. If all tiles are clean, skip ensemble voting and return the prefilter result.
          3. Otherwise, run all ensemble classifiers on only the non-clean tile subset.
             Supports three execution modes: async CUDA streams (multimodel=True, default),
             CPU thread pool (parallel=True), or serial (multimodel=False, parallel=False).
          4. Majority-vote across ensemble predictions and scatter into full-grid output.
          5. Optionally apply spatial majority-vote smoothing and generate a heatmap.
        """
        start = time.time()

        # --- Step 1: Run first classifier (prefilter) ---
        init_clf = self.first_clf

        # --- Prepare image tensor (upload uint8, normalize on GPU) ---
        rgba_image = readPolarPNMToRGBALive(image)
        rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
        # Keep as uint8 tensor — tile_and_cast_data_torch and classify_tiles pass
        # it through unchanged; normalisation (/255) happens inside the model on GPU.
        rgba_image = torch.as_tensor(rgba_image, device=self.device, dtype=torch.uint8)

        # --- Step 2: Get predictions from first binary (clean/non-clean) model ---
        # return_torch=True  → stays on GPU, no PCIe round-trip
        # return_tiles=True  → get the full tile tensor for reuse in stage 2
        # (majority-vote in the binary stage returns numpy, so return_torch is False there)
        if majorityVote:
            # majority_vote_2d_pytorch forces a CPU round-trip internally, so we get
            # numpy back; also we need tiles before voting truncates the count.
            base_preds_np, base_confs_np, all_tiles = classify_tiles(
                                                          init_clf.model,
                                                          rgba_image,
                                                          tile_size=self.tile_size,
                                                          step=self.step,
                                                          majorityVote=True,
                                                          thresholdMaxProbability=self.maxProbabilityThreshold,
                                                          forceLowMaxProbToThisClass=self.firstCleanClassID,
                                                          return_tiles=True,
                                                         )
            base_preds       = torch.tensor(base_preds_np, device=self.device, dtype=torch.int32)
            base_confidences = torch.tensor(base_confs_np, device=self.device, dtype=torch.float32)
        else:
            base_preds, base_confidences, all_tiles = classify_tiles(
                                                          init_clf.model,
                                                          rgba_image,
                                                          tile_size=self.tile_size,
                                                          step=self.step,
                                                          majorityVote=False,
                                                          thresholdMaxProbability=self.maxProbabilityThreshold,
                                                          forceLowMaxProbToThisClass=self.firstCleanClassID,
                                                          return_torch=True,
                                                          return_tiles=True,
                                                         )
            base_preds       = base_preds.to(dtype=torch.int32)
            base_confidences = base_confidences.to(dtype=torch.float32)

        # --- Identify non-clean tiles ---
        non_clean_indices = (base_preds != self.firstCleanClassID).nonzero(as_tuple=True)[0]
        if len(non_clean_indices) == 0:
            print("All tiles are clean — no ensemble voting needed!")
            final_predictions = base_preds
            final_confidences = base_confidences
        else:
            print(f"{len(non_clean_indices)} non-clean tiles for ensemble voting")

            # --- Step 3: Ensemble inference ---
            # Reuse all_tiles from stage 1 — no second unfold over the full image
            if debugExecuteSecondStage:
                npTiles = all_tiles   # all tiles already in (N, C, H, W) format
            else:
                npTiles = all_tiles[non_clean_indices]   # select subset via index


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
                # Models may have different class counts — process each separately
                for o in outputs:
                    probs = torch.nn.functional.softmax(o.float(), dim=1)
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
                    _t0 = time.time()
                    if (debugExecuteSecondStage):
                       preds, confs = classify_tiles(
                                                     clf.model,
                                                     rgba_image,
                                                     tile_size=self.tile_size,
                                                     step=self.step,
                                                     majorityVote=majorityVote,
                                                     thresholdMaxProbability=self.maxProbabilityThreshold,
                                                     forceLowMaxProbToThisClass=self.cleanClassID,
                                                     return_torch=not majorityVote,
                                                    )
                       if majorityVote:  # numpy path — wrap back
                           preds = torch.tensor(preds, device=self.device)
                           confs = torch.tensor(confs, device=self.device)
                    else:
                       preds, confs = classify_selected_tiles(
                                                              clf.name,
                                                              clf.model,
                                                              rgba_image,
                                                              npTiles,
                                                              tile_size=self.tile_size,
                                                              step=self.step,
                                                              thresholdMaxProbability=self.maxProbabilityThreshold,
                                                              forceLowMaxProbToThisClass=self.cleanClassID,
                                                              return_torch=True,
                                                             )
                    self.model_perf[clf.name] = 1.0 / (time.time() - _t0 + 1e-9)
                    #---------------------------------------------------------
                    preds_list.append(preds)   # already GPU tensors
                    conf_list.append(confs)
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
           # Compute tile grid dimensions — must match majority_vote_2d_pytorch
           # which uses (dim - tile_size) // step (no +1), truncating raw unfold output.
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
                                                         rgba_image,  # uint8, no scaling needed
                                                         tile_size=self.tile_size,
                                                         step=self.step,
                                                        )

        if legend:
            heatmap = self.classifiers[0].add_legend(heatmap)

        elapsed = time.time() - start + 1e-4
        self.hz = 1.0 / elapsed
        self._last_tile_count = len(final_predictions)
        self._last_elapsed    = elapsed

        self.print_perf()

        if (log):
          runid="ensemble"
          if parallel:
             runid="%s-parallel" % runid
          if multimodel:
             runid="%s-multimodel" % runid
          log_performance("perf.csv", runid, self.step, self.tile_size, majorityVote, self.maxProbabilityThreshold, len(non_clean_indices), self.hz)
        return heatmap, occupancy, responses

