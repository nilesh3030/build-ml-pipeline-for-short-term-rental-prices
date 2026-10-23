#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact.
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    # Read the data
    logger.info("Loading data")
    df = pd.read_csv(artifact_local_path)

    # Clean the data: filter prices and convert dates
    logger.info("Cleaning data")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Filter geographic boundaries
    logger.info("Filtering geographic boundaries")
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()
    
    # Save to CSV (always set index=False for ML workflows)
    logger.info("Saving cleaned dataset")
    df.to_csv("clean_sample.csv", index=False)

    # Upload to W&B
    logger.info("Uploading artifact to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact in W&B",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price for filtering",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price for filtering",
        required=True,
    )

    args = parser.parse_args()
    go(args)
