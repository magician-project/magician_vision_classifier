from liveClassifierTorch import ClassifierPnm, runSingle, readPolarPNMToRGBALive, tile_and_cast_data_torch, classify_tiles

class EnsembleClassifierPnm:
    def __init__(self, model_cfg_list, tile_size=64, step=16):
        """
        model_cfg_list: list of tuples (model_path, cfg_path)
        """
        assert len(model_cfg_list) > 0, "You must provide at least one (model_path, cfg_path) pair"
        self.classifiers = []
        self.tile_size = tile_size
        self.step = step

        # Initialize all ClassifierPnm instances
        for model_path, cfg_path in model_cfg_list:
            clf = ClassifierPnm(model_path=model_path, cfg_path=cfg_path, tile_size=tile_size, step=step)
            self.classifiers.append(clf)

        # Use class/color definitions from the first classifier
        self.classes      = self.classifiers[0].classes
        self.class_colors = self.classifiers[0].class_colors
        self.device       = self.classifiers[0].device

        # Find clean class ID (shared across models)
        self.cleanClassID = None
        for i, c in enumerate(self.classes):
            if c in ("class_clean", "Clean"):
                self.cleanClassID = i
                break
        if self.cleanClassID is None:
            raise ValueError("No 'class_clean' or 'Clean' class found in model classes")

        print(f"Initialized EnsembleClassifierPnm with {len(self.classifiers)} models")
        print("Clean class ID:", self.cleanClassID)

    @torch.no_grad()
    def forward(self, image, majorityVote=False, legend=True):
        """
        Runs ensemble classification on the given image.
        First classifier predicts normally.
        For tiles not predicted as 'class_clean', runs all classifiers and does majority voting.
        """
        # Run first classifier
        first_clf = self.classifiers[0]
        heatmap, occupancy, responses = runSingle(
            image, first_clf.model, first_clf.device, self.classes, self.class_colors,
            first_clf.tile_size, first_clf.step, majorityVote=majorityVote
        )

        # Convert the image for tiling again (needed for reclassification)
        rgba_image = readPolarPNMToRGBALive(image)
        rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
        rgba_image = (rgba_image.astype('float32') / 255.0)
        rgba_image = torch.tensor(rgba_image).float().to(self.device, dtype=torch.float32)

        # Tile image for reuse
        npTiles = tile_and_cast_data_torch(rgba_image, tile_size=self.tile_size, step=self.step).to(torch.float32).permute(0,3,1,2).contiguous().to(self.device)

        # Get base predictions from first model
        base_preds = classify_tiles(first_clf.model, rgba_image, tile_size=self.tile_size, step=self.step, majorityVote=majorityVote)
        base_preds = torch.tensor(base_preds).to(self.device)

        # Find indices of tiles that are not 'clean'
        non_clean_indices = (base_preds != self.cleanClassID).nonzero(as_tuple=True)[0]

        if len(non_clean_indices) == 0:
            print("All tiles classified as clean — skipping ensemble voting.")
        else:
            print(f"Running ensemble voting for {len(non_clean_indices)} non-clean tiles")

            # Collect predictions from all classifiers for those tiles
            all_predictions = []
            for clf in self.classifiers:
                preds = classify_tiles(clf.model, rgba_image, tile_size=self.tile_size, step=self.step, majorityVote=False)
                preds = torch.tensor(preds).to(self.device)
                all_predictions.append(preds)

            # Stack predictions: shape = (num_models, num_tiles)
            all_predictions = torch.stack(all_predictions)

            # Perform majority voting only on non-clean tiles
            final_predictions = base_preds.clone()
            for idx in non_clean_indices:
                votes = all_predictions[:, idx].tolist()
                voted_class = max(set(votes), key=votes.count)
                final_predictions[idx] = voted_class

            # Convert predictions back to numpy for visualization
            final_predictions = final_predictions.cpu().numpy()

            # Recompute heatmap using final ensemble decisions
            heatmap, occupancy, responses = generate_heatmap(
                final_predictions, self.classes, self.class_colors, rgba_image * 255.0,
                tile_size=self.tile_size, step=self.step
            )

        if legend:
            heatmap = first_clf.add_legend(heatmap)

        return heatmap, occupancy, responses

