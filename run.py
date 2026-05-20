
import sys
import os
import argparse
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def save_anomaly_visualization(idx, ts_data, img_data, ts_recon, img_recon, anomalies_dir, detection):
   
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(26, 4))
    anomaly_type = detection.get("anomaly_type", "?")
    confidence = detection.get("confidence", 0)
    fig.suptitle(f"Detection {idx:03d} | Type: {anomaly_type} | Confidence: {confidence:.4f}")

    # Panel 0: Original Image (display exactly as in input - FLIR style thermal)
    ax_orig = axes[0]
    if img_data is not None:
        img_orig = np.array(img_data)
        # Remove channel dimension if it's 1 (grayscale)
        if img_orig.ndim == 3 and img_orig.shape[0] == 1:
            img_orig = img_orig[0]  # (64, 64)
        # Normalize to [0, 1] for display (original is in 0-255 range)
        img_orig = np.clip(img_orig / 255.0, 0.0, 1.0)
        # Use turbo colormap to match FLIR thermal image appearance (blue=cold -> red=hot)
        ax_orig.imshow(img_orig, cmap="turbo", aspect="equal")
        ax_orig.set_title("Original Image (Input)")
    else:
        ax_orig.text(0.5, 0.5, "No Original Image", ha="center", va="center")
    ax_orig.axis("off")

    # Panel 1: Reconstructed Image (display as model reconstructed)
    ax_recon = axes[1]
    if img_recon is not None:
        img_rec = np.array(img_recon)
        # Remove channel dimension if it's 1 (grayscale)
        if img_rec.ndim == 3 and img_rec.shape[0] == 1:
            img_rec = img_rec[0]  # (64, 64)
        # Clip to [0, 1] to match input range
        img_rec = np.clip(img_rec, 0.0, 1.0)
        # Use same thermal colormap for direct comparison
        ax_recon.imshow(img_rec, cmap="turbo", aspect="equal")
        ax_recon.set_title("Reconstructed Image (Model Output)")
    else:
        ax_recon.text(0.5, 0.5, "No Reconstructed Image", ha="center", va="center")
    ax_recon.axis("off")

    # Panels 2-4: Time Series Channels with Original vs Reconstructed Overlay
    channel_names = ["Temperature (°C)", "Current (A)", "Voltage (V)"]
    colors = ["crimson", "steelblue", "forestgreen"]

    if ts_data is not None and ts_recon is not None:
        ts_orig = np.array(ts_data)      # (seq_len, 3)
        ts_rec = np.array(ts_recon)      # (seq_len, 3)
        steps = np.arange(ts_orig.shape[0])

        for c, (name, color) in enumerate(zip(channel_names, colors)):
            ax = axes[c + 2]

            # Plot original (solid line)
            ax.plot(steps, ts_orig[:, c], color=color, linewidth=2.5, label="Original", zorder=3)

            # Plot reconstructed (dashed line)
            ax.plot(steps, ts_rec[:, c], color=color, linewidth=2.5, linestyle="--",
                   alpha=0.7, label="Reconstructed", zorder=2)

            # Fill error region between them
            ax.fill_between(steps, ts_orig[:, c], ts_rec[:, c], alpha=0.2, color=color,
                           label="Error", zorder=1)

            ax.set_title(name)
            ax.set_xlabel("Time Step")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
    else:
        for c in range(3):
            ax = axes[c + 2]
            ax.text(0.5, 0.5, "No TS Data", ha="center", va="center", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    vis_path = os.path.join(anomalies_dir, f"detection_{idx:03d}_visualization.png")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()


def display_detection_results(results, output_dir):
    """Display detection results in a detailed table format."""
    print("\n" + "="*120)
    print("DETECTION RESULTS TABLE")
    print("="*120)

    filter_results = results.get("filter_results", [])

    if not filter_results:
        print("No detections found.")
        return

    # Create table data
    table_data = []
    anomalies_dir = os.path.join(output_dir, "anomalies")
    os.makedirs(anomalies_dir, exist_ok=True)

    for idx, detection in enumerate(filter_results):
        try:
            # Extract time series and image data
            ts_data = detection.get("ts_data")
            img_data = detection.get("img_data")
            ts_recon = detection.get("ts_recon")
            img_recon = detection.get("img_recon")
            img_jpg_bytes = detection.get("img_jpg_bytes")

            # Save original FLIR JPG image to output (legacy multimodal path)
            if img_jpg_bytes is not None:
                jpg_filename = f"detection_{idx:03d}_original.jpg"
                jpg_filepath = os.path.join(anomalies_dir, jpg_filename)
                with open(jpg_filepath, "wb") as f:
                    f.write(img_jpg_bytes)
            # TS-only mode: img_data may be None; skip image-side I/O
            if img_data is None:
                # Save raw time series only.
                ts_filename = f"detection_{idx:03d}_timeseries.npy"
                ts_filepath = os.path.join(anomalies_dir, ts_filename)
                if ts_data is not None:
                    np.save(ts_filepath, np.array(ts_data))
                save_anomaly_visualization(idx, ts_data, None, ts_recon, None,
                                            anomalies_dir, detection)
                ts_array = np.array(ts_data) if ts_data is not None else None
                row = {
                    "ID": idx,
                    "Timestamp": detection.get("timestamp", "N/A"),
                    "Anomaly Type": detection.get("anomaly_type", "Unknown"),
                    "Confidence": f"{detection.get('confidence', 0):.4f}",
                    "Validated": "Yes" if detection.get("validated") else "No",
                    "TS Min": f"{float(np.min(ts_array)):.4f}" if ts_array is not None else "N/A",
                    "TS Max": f"{float(np.max(ts_array)):.4f}" if ts_array is not None else "N/A",
                    "TS Mean": f"{float(np.mean(ts_array)):.4f}" if ts_array is not None else "N/A",
                    "TS Std": f"{float(np.std(ts_array)):.4f}" if ts_array is not None else "N/A",
                    "Image Shape": "N/A (TS-only)",
                    "IMG Min": "N/A", "IMG Max": "N/A", "IMG Mean": "N/A",
                    "TS File": ts_filename,
                    "IMG File": "N/A",
                }
                table_data.append(row)
                continue

            # Calculate time series statistics
            ts_stats = {}
            if ts_data is not None:
                ts_array = np.array(ts_data) if not isinstance(ts_data, np.ndarray) else ts_data
                ts_stats = {
                    "min": float(np.min(ts_array)),
                    "max": float(np.max(ts_array)),
                    "mean": float(np.mean(ts_array)),
                    "std": float(np.std(ts_array))
                }

            # Calculate image statistics
            img_stats = {}
            img_shape = "N/A"
            if img_data is not None:
                img_array = np.array(img_data) if not isinstance(img_data, np.ndarray) else img_data
                img_shape = str(img_array.shape)
                img_stats = {
                    "shape": img_shape,
                    "min": float(np.min(img_array)),
                    "max": float(np.max(img_array)),
                    "mean": float(np.mean(img_array))
                }

            # Save raw image data to file
            img_filename = f"detection_{idx:03d}.npy"
            img_filepath = os.path.join(anomalies_dir, img_filename)
            if img_data is not None:
                np.save(img_filepath, np.array(img_data))

            # Save raw time series data to file
            ts_filename = f"detection_{idx:03d}_timeseries.npy"
            ts_filepath = os.path.join(anomalies_dir, ts_filename)
            if ts_data is not None:
                np.save(ts_filepath, np.array(ts_data))

            # Save combined visualization (original vs reconstructed comparison)
            save_anomaly_visualization(idx, ts_data, img_data, ts_recon, img_recon, anomalies_dir, detection)

            # Build row
            row = {
                "ID": idx,
                "Timestamp": detection.get("timestamp", "N/A"),
                "Anomaly Type": detection.get("anomaly_type", "Unknown"),
                "Confidence": f"{detection.get('confidence', 0):.4f}",
                "Validated": "Yes" if detection.get("validated") else "No",
                "TS Min": f"{ts_stats.get('min', 0):.4f}",
                "TS Max": f"{ts_stats.get('max', 0):.4f}",
                "TS Mean": f"{ts_stats.get('mean', 0):.4f}",
                "TS Std": f"{ts_stats.get('std', 0):.4f}",
                "Image Shape": img_shape,
                "IMG Min": f"{img_stats.get('min', 0):.4f}",
                "IMG Max": f"{img_stats.get('max', 0):.4f}",
                "IMG Mean": f"{img_stats.get('mean', 0):.4f}",
                "TS File": ts_filename,
                "IMG File": img_filename
            }
            table_data.append(row)
        except Exception as e:
            print(f"Warning: Error processing detection {idx}: {e}")
            continue

    # Display as formatted table
    if table_data:
        # Print header
        headers = list(table_data[0].keys())
        col_widths = {h: max(len(h), max(len(str(row[h])) for row in table_data)) for h in headers}

        # Print header row
        header_row = " | ".join(f"{h:{col_widths[h]}}" for h in headers)
        print("\n" + header_row)
        print("-" * len(header_row))

        # Print data rows
        for row in table_data:
            data_row = " | ".join(f"{str(row[h]):{col_widths[h]}}" for h in headers)
            print(data_row)

        print("\n" + "="*120)
        print(f"Total detections: {len(table_data)}")
        print(f"Validated: {sum(1 for row in table_data if 'Yes' in row['Validated'])}")
        print(f"\nData files saved to: {os.path.abspath(anomalies_dir)}/")
        print("  - detection_XXX_timeseries.npy: Time series data for each detection")
        print("  - detection_XXX.npy: Image data for each detection")
        print("  - detection_XXX_visualization.png: Combined visualization (image + time series)")
        print("="*120)


def run_single_pipeline(config):
    """Run a single pipeline instance with given config."""
    from pipeline import BatteryAnomalyPipeline
    pipeline = BatteryAnomalyPipeline(config)
    return pipeline.run()


def main():
    parser = argparse.ArgumentParser(
        description="Battery Anomaly Detection Pipeline (Tests with 3 LLMs by default)"
    )
    parser.add_argument("--real-llm", action="store_true",
                        help="Use real HuggingFace models instead of mock")
    parser.add_argument("--single-llm", action="store_true",
                        help="Test with only one LLM (use --llm-model to specify)")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (used with --single-llm or as first model)")
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive human-in-the-loop review")
    parser.add_argument("--demo-only", action="store_true",
                        help="Only run SPARQL and filtering demos")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--max-per-exp", type=int, default=60,
                        help="Cap on samples per experiment (from Real_data/)")
    args = parser.parse_args()

    if args.demo_only:
        run_demos()
        return

    # Three LLM configurations for comparison
    llm_configs = [
        {
            "name": "Qwen2.5-7B (Default)",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
        },
        {
            "name": "Mistral-7B",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        },
        {
            "name": "Phi-3-Mini",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
        },
    ]

    # If --single-llm flag is set, use only the specified LLM
    if args.single_llm:
        llm_configs = [{
            "name": args.llm_model.split("/")[-1],
            "model_id": args.llm_model,
        }]

    # Base configuration
    base_config = {
        "seq_len": 50,
        "img_size": 64,
        "ts_latent_dim": 32,
        "img_latent_dim": 32,
        "epochs": args.epochs,
        "lr": 1e-3,
        "batch_size": 32,
        "confidence_threshold": 0.55,
        "temporal_k": 3,
        "temporal_n": 5,
        "threshold_percentile": 95,
        "use_mock_llm": not args.real_llm,
        "auto_approve": not args.interactive,
        "data_source": "mit",
    }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "#"*100)
    print(f"#  BATTERY ANOMALY DETECTION — MULTI-LLM COMPARISON")
    print(f"#  Testing {len(llm_configs)} LLM configuration(s)")
    print("#"*100)

    results_dict = {}
    all_results = []

    for i, llm_config in enumerate(llm_configs, 1):
        llm_name = llm_config["name"]
        model_id = llm_config["model_id"]

        print(f"\n{'='*100}")
        print(f"RUN {i}/{len(llm_configs)}: {llm_name}")
        print(f"Model: {model_id}")
        print(f"{'='*100}")

        # Create config for this LLM
        config = base_config.copy()
        config["llm_model"] = model_id
        config["output_dir"] = os.path.join(args.output_dir, f"run_{i}_{llm_name.replace('/', '_').replace(' ', '_')}")

        try:
            # Run pipeline
            results = run_single_pipeline(config)

            # Display results
            display_detection_results(results, config["output_dir"])

            # Print summary
            print("\n\nRun Summary:")
            print(f"  Detection threshold: {results['threshold']:.6f}")
            print(f"  Total predictions:   {sum(results['predictions'])}")
            validated = sum(1 for r in results["filter_results"] if r["validated"])
            print(f"  Validated anomalies: {validated}")
            print(f"  Output saved to: {os.path.abspath(config['output_dir'])}/")

            results_dict[llm_name] = {
                "model_id": model_id,
                "threshold": float(results['threshold']),
                "total_predictions": int(sum(results['predictions'])),
                "validated_anomalies": int(validated),
            }
            all_results.append((llm_name, model_id, results))

        except Exception as e:
            print(f"\nRun {i} FAILED with error:")
            print(f"  {str(e)}")
            results_dict[llm_name] = {"error": str(e)}

    # Generate comparison summary if multiple LLMs tested
    if len(llm_configs) > 1:
        print("\n" + "="*100)
        print("MULTI-LLM COMPARISON SUMMARY")
        print("="*100)

        print("\nDetection & Validation Metrics:")
        print("-" * 100)
        print(f"{'LLM Model':<40} {'Threshold':<15} {'Predictions':<15} {'Validated':<15}")
        print("-" * 100)

        for llm_name, metrics in results_dict.items():
            if "error" not in metrics:
                print(f"{llm_name:<40} {metrics['threshold']:<15.6f} {metrics['total_predictions']:<15} {metrics['validated_anomalies']:<15}")

        print("\nKey Observations:")
        print("  - Core detection (Steps 1-6) is independent of LLM choice")
        print("  - LLM selection affects ONLY Step 7 (Ontology Enrichment)")
        print("  - Detection metrics should be identical across all LLM configurations")
        print("  - Differences in validation or thresholds indicate data-dependent variations")

    print("\n" + "#"*100)
    print(f"#  PIPELINE COMPLETE")
    print(f"#  Output saved to: {os.path.abspath(args.output_dir)}/")
    print("#"*100)


def run_demos():
    """Run standalone demos for semantic reasoning and filtering."""
    print("="*70)
    print("RUNNING STANDALONE DEMOS")
    print("="*70)

    # Demo 1: Semantic Reasoner
    print("\n--- Demo 1: Semantic Reasoning ---")
    from reasoning.semantic_reasoner import SemanticReasoner
    reasoner = SemanticReasoner()
    reasoner.demo_queries()

    # Demo 2: False Positive Filter
    print("\n--- Demo 2: False Positive Filtering ---")
    from filtering.false_positive_filter import demonstrate_filter
    demonstrate_filter(reasoner=reasoner)

    # Demo 3: Real Data Loading
    print("\n--- Demo 3: Real Data Loading ---")
    from data.real_data import load_real_dataset
    dataset = load_real_dataset(seq_len=50, img_size=64, max_samples_per_exp=20)
    print(f"  Loaded: {dataset['ts_data'].shape[0]} samples")
    print(f"  Types: {set(dataset['anomaly_types'])}")


if __name__ == "__main__":
    main()
