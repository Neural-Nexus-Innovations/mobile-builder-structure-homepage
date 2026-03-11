import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

image_df = pd.read_csv("./csvsheets/image_data.csv")
video_df = pd.read_csv("./csvsheets/video_data.csv")
product_df = pd.read_csv("./csvsheets/shopify_products_dataset.csv")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Build embedding text
image_df["embedding_text"] = (
    image_df["image_description"].fillna("") + ". " +
    "Store type: " + image_df["store_types"].fillna("") + ". " +
    "Mood: " + image_df["mood"].fillna("") + ". " +
    "Theme: " + image_df["primary_theme"].fillna("") + ". " +
    "Colors: " + image_df["color_palette"].fillna("") + ". " +
    "Orientation: " + image_df["orientation"].fillna("") + ". " +
    "Tags: " + image_df["tags"].fillna("")
)

# Build embedding text for videos
video_df["embedding_text"] = (
    video_df["video_description"].fillna("") + ". " +
    "Store type: " + video_df["store_types"].fillna("") + ". " +
    "Mood: " + video_df["mood"].fillna("") + ". " +
    "Theme: " + video_df["primary_theme"].fillna("") + ". " +
    "Colors: " + video_df["color_palette"].fillna("") + ". " +
    "Orientation: " + video_df["orientation"].fillna("") + ". " +
    "Tags: " + video_df["tags"].fillna("")
)
# Build embedding text for products
product_df["embedding_text"] = (
    product_df["product_title"].fillna("") + ". " +
    product_df["product_description"].fillna("") + ". " +
    "Store type: " + product_df["store_types"].fillna("") + ". " +
    "Tags: " + product_df["tags"].fillna("")
)
def generate_incremental_embeddings(df, embeddings_path,uniqueKey = "unique_id"):
    """
    Generates embeddings only for new entries that are not already present
    in the given embeddings CSV.
    """

    # Ensure embedding_text column exists
    if "embedding_text" not in df.columns:
        # Build embedding text from all columns as a safe fallback
        df["embedding_text"] = df.astype(str).fillna("").agg(". ".join, axis=1)

    # embeddings_path is passed as parameter

    # If embeddings file exists, load it
    if os.path.exists(embeddings_path):
        existing_df = pd.read_csv(embeddings_path)
        existing_ids = set(existing_df[uniqueKey].tolist())
    else:
        existing_df = pd.DataFrame(columns=[uniqueKey, "embedding_text", "embedding"])
        existing_ids = set()

    # Find new rows needing embeddings
    new_rows = df[~df[uniqueKey].isin(existing_ids)].copy()

    if new_rows.empty:
        print("No new images to embed.")
        return existing_df

    print(f"Generating embeddings for {len(new_rows)} new images")

    texts = new_rows["embedding_text"].fillna("").tolist()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]

    new_rows["embedding"] = embeddings

    new_embedding_df = new_rows[[
        uniqueKey,
        "embedding_text",
        "embedding"
    ]]

    # Append new embeddings to existing ones
    final_df = pd.concat([existing_df, new_embedding_df], ignore_index=True)

    final_df.to_csv(embeddings_path, index=False)

    print(f"Saved {len(new_embedding_df)} new embeddings")

    return final_df


# Run incremental embedding generation
generate_incremental_embeddings(image_df, "csvsheets/image_embeddings.csv")
generate_incremental_embeddings(video_df, "csvsheets/video_embeddings.csv")