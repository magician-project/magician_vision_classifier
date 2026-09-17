import cv2
import time
import numpy as np
import torch
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
from mvc.inference.live_torch import (
    ClassifierPnm,
    readPolarPNMToRGBALive,
    tile_and_cast_data_torch,
    classify_tiles,
    render_predictions,
    runSingle,
    log_performance,
    gate_tiles,
    GATE_DEFECT_MASS,
    GATE_MAX_PROB,
    GATE_OFF
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

    # Convert to tensor if needed. Preserve dtype for the same reason as
    # tile_and_cast_data_torch(): uint8 must reach the model so that
    # Classifier.build_input_features() can apply the /255 on the GPU.
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

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
                            thresholdMaxProbability=0.655,
                            forceLowMaxProbToThisClass=None,
                            gateMode=GATE_DEFECT_MASS,
                            assignBestDefectClass=True,
                            return_torch=False):
    """
    Classify only the tiles in npTiles (already selected subset).

    thresholdMaxProbability : cut on the gate's score — see gate_tiles(). Under
                  the default gateMode this thresholds 1 - P(clean), NOT the max
                  probability, so it is not interchangeable with the old 0.50.
    gateMode, assignBestDefectClass : see gate_tiles().
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
        # max_probs stays the reported per-tile confidence; the gate decides the class.
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            predictions, forced = gate_tiles(probs, forceLowMaxProbToThisClass,
                                             thresholdMaxProbability,
                                             gateMode=gateMode,
                                             assignBestDefectClass=assignBestDefectClass)
            low_activations += forced
    else:
        preds_list = []
        for chunk in npTiles.chunk(chunks):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                preds_list.append(model(chunk))
        preds = torch.cat(preds_list)
        probs = torch.nn.functional.softmax(preds.float(), dim=1)
        max_probs, predictions = torch.max(probs, dim=1)
        if forceLowMaxProbToThisClass is not None and thresholdMaxProbability > 0.0:
            predictions, forced = gate_tiles(probs, forceLowMaxProbToThisClass,
                                             thresholdMaxProbability,
                                             gateMode=gateMode,
                                             assignBestDefectClass=assignBestDefectClass)
            low_activations += forced

    #print(f"Low-confidence tiles reassigned: {low_activations}")
    #print(f"classify_selected_tiles ({name}) done in {time.time() - start:.2f}s, on {len(predictions)} selected tiles")

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
# Fixed batch-size buckets for the ensemble's stage-2 models
# -------------------------------------------------------------------------------------
# ClassifierPnm.compile()s every model (classifier_pnm.py), and live_torch.py turns on
# torch.backends.cudnn.benchmark -- both assume a STABLE input shape and pay a one-off
# multi-second re-trace/re-autotune cost the first time they see a new one. That's fine
# for the single-classifier and stage-1 paths, which always batch the full, constant
# tile grid (e.g. 5950 tiles). It is NOT fine for the stage-2 ensemble members: they only
# run on the non_clean_indices subset, whose size is scene-dependent and changes on
# nearly every frame, so nearly every frame paid that one-off cost -- measured at
# 11-16s/call with a synthetic random-size repro (mirrored in production: 3-6 frames
# processed in 45-60s instead of a steady ~12-23Hz). Rounding the batch up to the
# nearest of a handful of fixed sizes below means only ~7 distinct shapes are ever
# seen (well under torch._dynamo's default recompile_limit of 8), so the compile/
# autotune cost is paid at most once per bucket instead of once per frame.
_TILE_BATCH_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)


def _bucketed_batch_size(n, max_n):
    """Round n up to the next fixed bucket in _TILE_BATCH_BUCKETS, or max_n if n is
    larger than every bucket. max_n should be the total tile count available to pad
    from (the full per-frame tile grid), which is also the shape the stage-1 model
    already runs at every frame, so its compile/autotune cache is warm too."""
    for b in _TILE_BATCH_BUCKETS:
        if n <= b:
            return min(b, max_n)
    return max_n


# -------------------------------------------------------------------------------------
# Async multi-model helpers
# -------------------------------------------------------------------------------------
def run_models_async(models, x, streams):
    """
    Run all models in *models* concurrently on the given *streams* (one per model).

    *streams* must be persistent torch.cuda.Stream objects created once by the caller
    (e.g. EnsembleClassifierPnm keeps one per model for its whole lifetime) and reused
    across calls. Allocating a fresh torch.cuda.Stream() on every frame -- the previous
    version of this function did -- let per-stream caching-allocator bookkeeping pile up
    call after call: the per-call Hz timer (wall clock of just this function) stayed
    fast, but the process as a whole showed multi-second, GPU-idle (0% utilization)
    stalls between frames -- 6 frames processed in 60s instead of the ~700-1400 a
    steady ~12-23Hz would give. Reusing the same streams every call removes that growth.

    Input *x* is converted to channels_last memory format (zero-copy stride update)
    before fanning out to streams. Uses FP16 autocast. Requires CUDA.
    """
    assert torch.cuda.is_available(), "CUDA required for async inference"
    assert len(streams) == len(models), "run_models_async needs one persistent stream per model"
    results = [None] * len(models)

    # Convert input once to channels-last before fanning out to streams.
    # Each model was loaded with .to(memory_format=torch.channels_last) so the
    # layout must match here; the conversion is zero-copy (stride update only).
    x = x.to(memory_format=torch.channels_last)

    for i, (model, stream) in enumerate(zip(models, streams)):
        with torch.cuda.stream(stream):
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                results[i] = model(x)   # read-only input, no clone needed

    # Wait only on the streams this call actually used, not a whole-device
    # torch.cuda.synchronize() (which also blocks on unrelated queued work).
    for stream in streams:
        stream.synchronize()
    return results
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def parallel_classify_tiles(classifiers, rgba_image, tile_size, step, majorityVote=False, max_workers=None,
                            majority_window=3):
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
                majority_window=majority_window,
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
                 min_hz=0.0, benchmark_tiles=512, precache=False):
        """
        initial_model_cfg: (model_path, cfg_path) for the first classifier
        model_cfg_list:    list of (model_path, cfg_path) for ensemble models
        min_hz:            drop any ensemble model whose single-forward-pass benchmark
                           is below this threshold (0.0 = keep all, default)
        benchmark_tiles:   batch size used for the Hz benchmark (default 512)
        precache:          run a forward-pass timing test on each model at load time
        """
        assert len(model_cfg_list) > 0, "You must provide at least one ensemble model."

        # Tile decision gate -- see gate_tiles(). Threshold 0.0 leaves the gate
        # OFF (plain argmax), the historical behaviour here. Plain attributes so
        # a caller can set them after construction -- wxAnnotator drives them
        # from its Classifier tab via _applyGateSettings():
        #     ens.gateMode = GATE_DEFECT_MASS; ens.maxProbabilityThreshold = 0.57
        # Thresholds are per-model and per-mode -- take them from the trainer's
        # sweep, never reuse a max_prob threshold under defect_mass.
        self.maxProbabilityThreshold = 0.0
        self.gateMode                = GATE_DEFECT_MASS
        self.assignBestDefectClass   = True
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
                                       precache=precache,
                                      )

        self.name = "EnsembleClassifier"
        self._benchmark_tiles = benchmark_tiles

        # --- Load ALL ensemble classifiers (kept in full for re-filtering) ---
        self._all_classifiers = [
                                   ClassifierPnm(model_path=mp, cfg_path=cp, tile_size=tile_size, step=step, precache=precache)
                                   for mp, cp in model_cfg_list
                                 ]
        self.classifiers = list(self._all_classifiers)   # active subset

        # --- Common definitions from the first model ---
        self.classes      = self._all_classifiers[0].classes
        self.class_colors = self._all_classifiers[0].class_colors
        self.device       = self._all_classifiers[0].device

        # One persistent CUDA stream per ensemble model, created once here and reused by
        # run_models_async() on every forward() call -- see that function's docstring for
        # why creating fresh streams per-frame is not just wasteful but actively harmful.
        # Keyed by name (not list position) so a later apply_min_hz() re-filter can look
        # streams up for whatever subset is active without recreating anything.
        self._streams_by_name = {}
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self._streams_by_name = {
                clf.name: torch.cuda.Stream(device=self.device) for clf in self._all_classifiers
            }

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

        if min_hz > 0.0:
            # Benchmark any model not yet measured
            for clf in self._all_classifiers:
                if clf.name not in self.model_perf or self.model_perf[clf.name] == 0.0:
                    hz = self._benchmark_clf(clf.model, self._benchmark_tiles,
                                             self.tile_size, self.device)
                    self.model_perf[clf.name] = hz
                    print(f"[Ensemble] Benchmarked {clf.name}: {hz:.2f} Hz")

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
    def forward(self, image, majorityVote=False, legend=True, strict=True, parallel=False, multimodel=True , debugExecuteSecondStage=False, log=True,
                erosion_kernel=0, erosion_threshold=0, majority_window=3):
        """
        Two-stage ensemble inference: prefilter (binary clean/non-clean) then ensemble voting.

        Pipeline:
          1. Run the first binary classifier on the full image to identify non-clean tiles.
          2. If all tiles are clean, skip ensemble voting and return the prefilter result.
          3. Otherwise, run all ensemble classifiers on only the non-clean tile subset.
             Supports three execution modes: async CUDA streams (multimodel=True, default),
             CPU thread pool (parallel=True), or serial (multimodel=False, parallel=False).
          4. Majority-vote across ensemble predictions and scatter into full-grid output.
          5. Optionally apply spatial majority-vote smoothing (majority_window x majority_window)
             and generate a heatmap, keeping only tiles that pass the neighbourhood vote
             (erosion_kernel/erosion_threshold, same rule as ClassifierPnm.forward).
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
                                                          gateMode=self.gateMode,
                                                          assignBestDefectClass=self.assignBestDefectClass,
                                                          return_tiles=True,
                                                          majority_window=majority_window,
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
                                                          gateMode=self.gateMode,
                                                          assignBestDefectClass=self.assignBestDefectClass,
                                                          return_torch=True,
                                                          return_tiles=True,
                                                         )
            base_preds       = base_preds.to(dtype=torch.int32)
            base_confidences = base_confidences.to(dtype=torch.float32)

        # --- Identify non-clean tiles ---
        non_clean_indices = (base_preds != self.firstCleanClassID).nonzero(as_tuple=True)[0]
        if len(non_clean_indices) == 0:
            #print("All tiles are clean — no ensemble voting needed!")
            final_predictions = base_preds
            final_confidences = base_confidences
        else:
            #print(f"{len(non_clean_indices)} non-clean tiles for ensemble voting")

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
                #print("Running ensemble via async CUDA streams")
                # Pad the selected-tile batch up to a fixed bucket size -- see
                # _bucketed_batch_size()'s docstring for why a raw, scene-dependent
                # batch size stalls torch.compile/cudnn.benchmark for 11-16s/frame.
                n_selected = npTiles.shape[0]
                bucket_n = _bucketed_batch_size(n_selected, all_tiles.shape[0])
                if bucket_n > n_selected:
                    pad_tiles = npTiles[:1].repeat(bucket_n - n_selected, 1, 1, 1)
                    npTiles_batch = torch.cat([npTiles, pad_tiles], dim=0)
                else:
                    npTiles_batch = npTiles

                streams = [self._streams_by_name[clf.name] for clf in self.classifiers]
                outputs = run_models_async([clf.model for clf in self.classifiers], npTiles_batch, streams)
                # Drop the padding rows before voting -- they were never real tiles.
                outputs = [o[:n_selected] for o in outputs]
                # Models may have different class counts — process each separately
                for o in outputs:
                    probs = torch.nn.functional.softmax(o.float(), dim=1)
                    max_probs, predictions = torch.max(probs, dim=1)
                    preds_list.append(predictions)
                    conf_list.append(max_probs)
            elif parallel:
            #===========================================================================================
                #print("Running ensemble via CPU thread pool ( This runs all tiles, not just selected btw ) ")
                ensemble_results = parallel_classify_tiles(self.classifiers, rgba_image, self.tile_size, self.step, majorityVote,
                                                           majority_window=majority_window)
                for preds, confs in ensemble_results:
                     preds_list.append(torch.tensor(preds, device=self.device))
                     conf_list.append(torch.tensor(confs, device=self.device))
            else:
            #===========================================================================================
                #print("Running ensemble serially")
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
                                                     gateMode=self.gateMode,
                                                     assignBestDefectClass=self.assignBestDefectClass,
                                                     return_torch=not majorityVote,
                                                     majority_window=majority_window,
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
                                                              gateMode=self.gateMode,
                                                              assignBestDefectClass=self.assignBestDefectClass,
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
           # Compute tile grid dimensions — must match the unfold-based grid every
           # other tile count in this pipeline uses (classify_tiles, generate_heatmap):
           # (dim - tile_size) // step + 1. Omitting the +1 here used to undercount the
           # grid by one row/column, so majority_vote_final() silently truncated
           # final_predictions/final_confidences (~154 tiles on a 2048x2448 frame),
           # dropping real tiles from the ensemble vote every frame.
           height, width, _ = rgba_image.shape
           tilesW = (width  - self.tile_size) // self.step + 1
           tilesH = (height - self.tile_size) // self.step + 1

           # Apply majority vote smoothing
           final_predictions, final_confidences = majority_vote_final(final_predictions, final_confidences, tilesW, tilesH, window_size=majority_window)

        # final_predictions should now be mostly 'self.cleanClassID' (e.g. 7),
        # with other ensemble class IDs only on non-clean tiles.
        #dump_predictions_to_file(final_predictions, "final_predictions.txt",header="Final ensemble-voted predictions")
        #global_votes = global_votes.cpu().numpy()  # (optional debugging)

        # --- Step 5: Generate heatmap using final voted results ---
        heatmap, occupancy, responses = render_predictions(
                                                         final_predictions,
                                                         final_confidences,
                                                         self.classes,
                                                         self.class_id_to_color,
                                                         self.cleanClassID,
                                                         rgba_image,  # uint8, no scaling needed
                                                         self.tile_size,
                                                         self.step,
                                                         erosion_kernel=erosion_kernel,
                                                         erosion_threshold=erosion_threshold,
                                                        )

        if legend:
            heatmap = self.classifiers[0].add_legend(heatmap)

        elapsed = time.time() - start + 1e-4
        self.hz = 1.0 / elapsed
        self._last_tile_count = len(final_predictions)
        self._last_elapsed    = elapsed

        # print_perf() is an on-demand diagnostic, not per-frame output. Calling it here
        # printed a block every frame -- and since model_perf is only ever filled by the
        # SERIAL branch, in the default async path that block was the literal line
        # "[Ensemble] No performance data yet." at 20+ Hz. It also scrolled the live status
        # line away. The same numbers now reach the status line through infer_stats.

        if (log):
          runid="ensemble"
          if parallel:
             runid="%s-parallel" % runid
          if multimodel:
             runid="%s-multimodel" % runid
          log_performance("perf.csv", runid, self.step, self.tile_size, majorityVote, self.maxProbabilityThreshold, len(non_clean_indices), self.hz)
        return heatmap, occupancy, responses


# =======================================================================================
# Cascade classifier: an ORDERED chain of stages, each with its own tiling step and gate
# threshold, replacing EnsembleClassifierPnm's fixed "one screen + N-way majority vote"
# shape with a genuine screen-then-recheck-then-recheck... pipeline.
#
# analysis/eval/eval_cascade_step_sweep.py measured this exact mechanism OFFLINE, from
# cached per-frame scores, before any live code existed to run it -- the containment-
# matrix geometry below is a direct port of that script's containment_matrix/
# eligible_mask (same derivation: the tile grid is separable, square tiles, uniform
# steps per axis, so "which target cells overlap a promoted source cell" is two matrix
# multiplies, not a pixel raster), adapted from per-frame numpy arrays swept across many
# cached frames offline to GPU torch tensors computed once per live frame.
# =======================================================================================
@torch.no_grad()
def containment_matrix_torch(step_from, step_to, size, tile_size, device):
    """(n_from, n_to) bool, ONE axis: does tile `f` of step_from contain the CENTER of
    tile `t` of step_to? Geometry only -- independent of frame content and threshold, so
    the caller should compute this once per (step_from, step_to, size, tile_size) and
    reuse it for every frame that pair of stages ever processes (see
    CascadeClassifierPnm._containment).

    Mirrors tile_and_cast_data_torch's unfold convention exactly: tile origins at
    0, step, 2*step, ... while the tile still fits (`torch.arange(0, size-tile+1, step)`,
    the same formula grid_origins() in analysis/eval/eval_step_curve.py uses) -- any
    independent formula here would silently drift from what the tiler actually produces.
    """
    xs_from = torch.arange(0, size - tile_size + 1, step_from, device=device)
    xs_to   = torch.arange(0, size - tile_size + 1, step_to,   device=device)
    centers_to = xs_to + tile_size // 2
    return ((xs_from[:, None] <= centers_to[None, :]) &
            (centers_to[None, :] < xs_from[:, None] + tile_size))


@torch.no_grad()
def eligible_mask_torch(promoted2d, y_contain, x_contain):
    """(ny_to, nx_to) bool: which of the TARGET grid's cells overlap a promoted cell of
    the SOURCE grid.

        eligible[y2,x2] = OR over (y1,x1) promoted[y1,x1] AND y_contain[y1,y2] AND x_contain[x1,x2]
                         = (y_contain.T @ promoted @ x_contain) > 0

    See eval_cascade_step_sweep.py's docstring for the full derivation.

    dtype note: the numpy original (eval_cascade_step_sweep.py) does this matmul in
    int8/int32, which numpy's CPU BLAS handles fine. cuBLAS does NOT implement integer
    GEMM ("addmm_cuda not implemented for 'Int'", caught live on GPU) -- so this uses
    float32 instead. Every product/sum here is exactly 0 or a small positive integer
    representable exactly in float32, so `> 0.5` is an exact, not approximate, test."""
    promoted = promoted2d.to(torch.float32)
    result = (y_contain.t().to(torch.float32) @ promoted) @ x_contain.to(torch.float32)
    return result > 0.5


class CascadeStage:
    """One stage of a CascadeClassifierPnm: a model plus its own tiling step and gate."""

    def __init__(self, model_path, cfg_path, tile_size=48, step=16, threshold=0.0,
                 gate_mode=GATE_DEFECT_MASS, assign_best_defect_class=True, precache=False):
        self.clf = ClassifierPnm(model_path=model_path, cfg_path=cfg_path,
                                 tile_size=tile_size, step=step, precache=precache)
        # ClassifierPnm overwrites tile_size from the model's own cfg["hparams"]["tile_size"]
        # -- read it back rather than trust the constructor argument, so a stage's
        # geometry always matches what its model actually expects.
        self.tile_size = self.clf.tile_size
        self.step = step
        self.threshold = threshold
        self.gate_mode = gate_mode
        self.assign_best_defect_class = assign_best_defect_class

    @property
    def model(self):
        return self.clf.model

    @property
    def classes(self):
        return self.clf.classes

    @property
    def name(self):
        return self.clf.name


class CascadeClassifierPnm:
    """Ordered screen-then-recheck cascade over N stages, each free to use its own
    tiling step and gate threshold -- the live counterpart to
    analysis/eval/eval_cascade_step_sweep.py, which measured this mechanism offline
    before it existed to run here.

    THE CHAIN, not a vote (unlike EnsembleClassifierPnm): stage 0 classifies every tile
    of the full frame at its OWN step. A tile is "promoted" if stage 0's gate flags it
    non-clean. Stage 1 then re-tiles ONLY the region stage 0 promoted, at stage 1's OWN
    (possibly different) step, and re-classifies just that subset -- everything stage 0
    locked clean never reaches stage 1 at all. This repeats for every later stage. The
    FINAL stage's decision on a promoted tile is the frame's answer for that tile; a tile
    no stage ever promotes stays clean by construction.

    Because two consecutive stages can tile at different steps, mapping "stage i-1
    promoted these grid cells" onto "these are the flat indices stage i's own grid
    should classify" needs a geometric correspondence between two different regular
    grids over the same frame -- containment_matrix_torch/eligible_mask_torch above.

    CAVEAT carried over from the eval tool: the containment geometry assumes every stage
    uses the SAME tile_size (48px, the one deployment geometry this repo trains against
    -- see analysis/sweeps/bench_inference.py). A stage whose model was trained at a
    different tile_size would need the containment math generalised; nothing in this
    repo currently is.

    STATUS: this is the ported geometry + a working forward() implementation, NOT yet
    wired into live_torch.py/live_torch_ros.py's construction or recommended_
    configuration.json's schema (both still only know EnsembleClassifierPnm's fixed
    screen+vote shape) -- that wiring, and real-hardware Hz validation, are separate,
    deliberately not-yet-done next steps.
    """

    def __init__(self, stage_cfgs, precache=False):
        """stage_cfgs: ORDERED list of dicts, each at minimum
            {'model_path': ..., 'cfg_path': ..., 'step': ..., 'threshold': ...}
        and optionally 'tile_size', 'gate_mode', 'assign_best_defect_class'.
        stage_cfgs[0] is the first screen; stage_cfgs[-1] makes the final decision.
        """
        assert len(stage_cfgs) >= 2, ("CascadeClassifierPnm needs at least 2 stages -- "
                                      "use ClassifierPnm directly for a single model.")
        self.stages = [CascadeStage(precache=precache, **sc) for sc in stage_cfgs]
        self.name = "CascadeClassifier"
        self.device = self.stages[0].clf.device

        # The final stage's label space is what the frame is ultimately reported in --
        # mirrors eval_cascade_step_sweep.py's convention (its "frames2" argument, whose
        # classes/clean_id drive every macro/FA number).
        self.classes = self.stages[-1].classes
        self.class_colors = self.stages[-1].clf.class_colors
        self.class_id_to_color = [torch.tensor(c, dtype=torch.uint8) for c in self.class_colors]

        def find_clean_id(cls_list):
            for i, c in enumerate(cls_list):
                if c.lower() in ("class_clean", "clean"):
                    return i
            return None

        self.clean_ids = [find_clean_id(s.classes) for s in self.stages]
        if any(c is None for c in self.clean_ids):
            raise ValueError("Could not find 'class_clean' in every stage's class list")

        self._containment_cache = {}   # (step_from, step_to, H, W, tile_size) -> (y_c, x_c)
        self.hz = 0.0
        self.model_perf = {}

        print(f"Initialized CascadeClassifierPnm with {len(self.stages)} stages: "
              f"{[s.name for s in self.stages]} "
              f"(steps={[s.step for s in self.stages]}, "
              f"thresholds={[s.threshold for s in self.stages]})")

    def _containment(self, step_from, step_to, h, w, tile_size):
        key = (step_from, step_to, h, w, tile_size)
        cached = self._containment_cache.get(key)
        if cached is None:
            y_c = containment_matrix_torch(step_from, step_to, h, tile_size, self.device)
            x_c = containment_matrix_torch(step_from, step_to, w, tile_size, self.device)
            cached = (y_c, x_c)
            self._containment_cache[key] = cached
        return cached

    @staticmethod
    def _grid_dims(size, tile_size, step):
        return (size - tile_size) // step + 1   # matches unfold's window count exactly

    @torch.no_grad()
    def forward(self, image, legend=True, log=True, erosion_kernel=0, erosion_threshold=0):
        """One frame through the full chain. Returns (heatmap, occupancy, responses),
        the same triple ClassifierPnm.forward()/EnsembleClassifierPnm.forward() return.
        erosion_kernel/erosion_threshold apply the neighbourhood vote to the FINAL stage's
        grid, same rule as ClassifierPnm.forward.
        """
        start = time.time()

        rgba_image = readPolarPNMToRGBALive(image)
        rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
        rgba_image = torch.as_tensor(rgba_image, device=self.device, dtype=torch.uint8)
        h, w = rgba_image.shape[:2]

        # --- Stage 0: classify every tile of the full frame at its own step ---
        stage0 = self.stages[0]
        preds, confs, _tiles0 = classify_tiles(
            stage0.model, rgba_image, tile_size=stage0.tile_size, step=stage0.step,
            majorityVote=False, thresholdMaxProbability=stage0.threshold,
            forceLowMaxProbToThisClass=self.clean_ids[0], gateMode=stage0.gate_mode,
            assignBestDefectClass=stage0.assign_best_defect_class,
            return_torch=True, return_tiles=True)

        ny0 = self._grid_dims(h, stage0.tile_size, stage0.step)
        nx0 = self._grid_dims(w, stage0.tile_size, stage0.step)
        promoted2d = (preds.to(torch.int32) != self.clean_ids[0]).view(ny0, nx0)

        final_predictions = final_confidences = None
        final_ny, final_nx = ny0, nx0
        prev_step, prev_tile_size = stage0.step, stage0.tile_size

        # --- Stages 1..N-1: re-tile only what the PREVIOUS stage promoted, at THIS
        #     stage's own step, and re-classify just that subset ---
        for i in range(1, len(self.stages)):
            stage = self.stages[i]
            ny = self._grid_dims(h, stage.tile_size, stage.step)
            nx = self._grid_dims(w, stage.tile_size, stage.step)

            y_c, x_c = self._containment(prev_step, stage.step, h, w, prev_tile_size)
            eligible2d = eligible_mask_torch(promoted2d, y_c, x_c)
            selected_indices = eligible2d.flatten().nonzero(as_tuple=True)[0]

            # Default every cell to THIS stage's clean class -- a tile the previous
            # stage didn't promote never reaches this stage at all, and stays clean.
            final_predictions = torch.full((ny * nx,), self.clean_ids[i],
                                           dtype=torch.int32, device=self.device)
            final_confidences = torch.zeros((ny * nx,), dtype=torch.float32, device=self.device)

            if len(selected_indices) > 0:
                _t0 = time.time()
                sel_tiles = tile_and_cast_selected_tiles_torch(
                    rgba_image, selected_indices, tile_size=stage.tile_size, step=stage.step)
                # (N, tile, tile, C) -> (N, C, tile, tile): tile_and_cast_selected_tiles_torch
                # does NOT do this permute itself (unlike tile_and_cast_data_torch inside
                # classify_tiles, whose return_tiles=True output already has it applied).
                sel_tiles = sel_tiles.permute(0, 3, 1, 2).contiguous()

                sel_preds, sel_confs = classify_selected_tiles(
                    stage.name, stage.model, rgba_image, sel_tiles,
                    tile_size=stage.tile_size, step=stage.step,
                    thresholdMaxProbability=stage.threshold,
                    forceLowMaxProbToThisClass=self.clean_ids[i],
                    gateMode=stage.gate_mode,
                    assignBestDefectClass=stage.assign_best_defect_class,
                    return_torch=True)

                final_predictions[selected_indices] = sel_preds.to(torch.int32)
                final_confidences[selected_indices] = sel_confs.to(torch.float32)
                self.model_perf[stage.name] = 1.0 / (time.time() - _t0 + 1e-9)

            promoted2d = (final_predictions != self.clean_ids[i]).view(ny, nx)
            prev_step, prev_tile_size = stage.step, stage.tile_size
            final_ny, final_nx = ny, nx

        final_predictions_np = final_predictions.cpu().numpy()
        final_confidences_np = final_confidences.cpu().numpy()

        heatmap, occupancy, responses = render_predictions(
            final_predictions_np, final_confidences_np, self.classes, self.class_id_to_color,
            self.clean_ids[-1], rgba_image,
            self.stages[-1].tile_size, self.stages[-1].step,
            erosion_kernel=erosion_kernel, erosion_threshold=erosion_threshold)

        if legend:
            heatmap = self.stages[0].clf.add_legend(heatmap)

        elapsed = time.time() - start + 1e-4
        self.hz = 1.0 / elapsed
        self._last_tile_count = final_ny * final_nx
        self._last_elapsed = elapsed

        if log:
            log_performance("perf.csv", "cascade", self.stages[-1].step,
                            self.stages[-1].tile_size, False, self.stages[-1].threshold,
                            int(promoted2d.sum().item()), self.hz)

        return heatmap, occupancy, responses

