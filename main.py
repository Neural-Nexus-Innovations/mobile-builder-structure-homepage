# This is a sample Python script.
from http.client import responses

from fastapi import  FastAPI
from pydantic import BaseModel
import pandas as pd
import ollama
import json
import random
from create_embedding import generate_incremental_embeddings
app = FastAPI()
#CSVs
components_df = pd.read_csv("csvsheets/components.csv")
# Load latest image data
# Load latest video data
image_df  = pd.read_csv("csvsheets/image_data.csv")
video_df = pd.read_csv("csvsheets/video_data.csv")
# Load collections
collections_df = pd.read_csv("csvsheets/collection.csv")
# Load products and build embedding text
product_df = pd.read_csv("csvsheets/shopify_products_dataset.csv")
# Build embedding text for products (only desired semantic fields)
product_df["embedding_text"] = (
    product_df["product_title"].fillna("") + ". " +
    product_df["product_description"].fillna("") + ". " +
    "Store type: " + product_df["store_types"].fillna("") + ". " +
    "Tags: " + product_df["tags"].fillna("")
)
# Run incremental embedding generation
embedding_df = generate_incremental_embeddings(image_df, "csvsheets/image_embeddings.csv")
video_embedding_df = generate_incremental_embeddings(video_df, "csvsheets/video_embeddings.csv")
product_embedding_df = generate_incremental_embeddings(product_df, "csvsheets/product_embeddings.csv", "product_id")

# --- Embedding + FAISS setup ---
import numpy as np
import ast
import faiss

# Load image embeddings
embedding_df = pd.read_csv("csvsheets/image_embeddings.csv")
embedding_df["embedding"] = embedding_df["embedding"].apply(ast.literal_eval)

# Load video embeddings
video_embedding_df = pd.read_csv("csvsheets/video_embeddings.csv")
video_embedding_df["embedding"] = video_embedding_df["embedding"].apply(ast.literal_eval)

product_embedding_df = pd.read_csv("csvsheets/product_embeddings.csv")
product_embedding_df["embedding"] = product_embedding_df["embedding"].apply(ast.literal_eval)
# merge embeddings with image metadata
if "embedding" in image_df.columns:
    image_df = image_df.drop(columns=["embedding"])

image_df = image_df.merge(
    embedding_df[["unique_id", "embedding"]],
    on="unique_id",
    how="left"
)

# merge embeddings with video metadata
if "embedding" in video_df.columns:
    video_df = video_df.drop(columns=["embedding"])

video_df = video_df.merge(
    video_embedding_df[["unique_id", "embedding"]],
    on="unique_id",
    how="left"
)

if "embedding" in product_df.columns:
    product_df = product_df.drop(columns=["embedding"])

product_df = product_df.merge(
    product_embedding_df[["product_id", "embedding"]],
    on="product_id",
    how="left"
)


def get_random_collection():
    row = collections_df.sample(1).iloc[0]
    return row["collection_id"], row["collection_name"]
def faiss_search(filtered_df, query_embedding, k=3):
    """
    Performs FAISS nearest neighbor search on a filtered dataframe of images.
    filtered_df must contain an 'embedding' column.
    """
    if filtered_df.empty:
        return filtered_df

    embeddings = np.vstack(filtered_df["embedding"].values).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    distances, indices = index.search(query_vector, min(k, len(filtered_df)))

    return filtered_df.iloc[indices[0]]

import random
def assign_videos_to_layout(structure,description , store_type):
    mask = video_df["allowed_components"].str.contains("video_slider", na=False)

    video_df.loc[mask, "allowed_components"] = video_df.loc[mask, "allowed_components"].apply(
        lambda x: x if "brand_product_slider" in x else x + ",brand_product_slider"
    )
    print("Assigning videos to layout")
    layout = structure.get("layout")
    description = description.lower()
    print(structure.get("theme"),"structure layout in vide")
    component_video_counts = {
        "videoWithOneCTA": 1,
        "videoWithTwoCTA": 1,
        "video_banner": 1,
        "brand_product_slider":1,
        "product_slider_with_video":3,
        "video_slider": 3,
    }
    print(structure.get)
    for component in layout:
        video_component = component.get("component")

        # Skip components that are not video components
        if video_component not in component_video_counts:
            continue

        filtered_videos = video_df[
            (video_df["store_types"].str.contains(store_type, case=False, na=False)) &
            (video_df["allowed_components"].str.contains(video_component, case=False, na=False))
        ]

        print(len(filtered_videos), "filter vid length")

        if not filtered_videos.empty:
            from openai import OpenAI
            client = OpenAI()

            products = component.get("products", [])

            # If this is product_slider_with_video, search videos using product embedding
            if video_component == "product_slider_with_video" and products:

                product_ids = [p["product_id"] for p in products]

                product_rows = product_df[product_df["product_id"].isin(product_ids)]

                if not product_rows.empty:
                    product_embedding = product_rows.iloc[0]["embedding"]
                    selected_rows = faiss_search(filtered_videos, product_embedding)
                else:
                    query_text = f"{description} store type {store_type} banner video"

                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=query_text
                    )

                    query_embeddings = response.data[0].embedding
                    selected_rows = faiss_search(filtered_videos, query_embeddings)

            else:
                query_text = f"{description} store type {store_type} banner video"

                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query_text
                )

                query_embeddings = response.data[0].embedding
                selected_rows = faiss_search(filtered_videos, query_embeddings)

            videos = []

            for _, row in selected_rows.iterrows():
                videos.append(row["video_url"])
                print(video_component != "video_slider", video_component + "  videocomps")
            if video_component == "video_slider" or video_component == "product_slider_with_video":
                component["videos"] = [{
                    "banner_url": vid,
                    "banner_link_to": "collection",
                    "banner_link_value": "__COLLECTION__"
                }
                for vid in videos]
            else:
                component["videos"] = [{
                    "banner_url": videos[0],
                    "banner_link_to": "collection",
                    "banner_link_value": "__COLLECTION__"
                }]

            continue
    return structure

def assign_images_to_layout(structure, description, store_type):
    # write your own code here . resume here


    print("Assigning images to layout")
    layout = structure.get("layout", [])
    description_lower = description.lower()

    for component in layout:
        # Special handling for banner-style components
        banner_component = component.get("component")

        banner_image_counts = {
            "single_banner": 1,
            "brand_product_slider":1,
            "banner_slider": 3,
            "longBannerSlider": 3,
            "threeLayoutBanner": 3,
            "sevenLayoutBanner": 7,
            "twoGridLayoutBanner": 2,
            "bannerSliderClassic": 3,
            "bannerSliderPeekable": 3,
            "continuousBannerSlider": 3,
        }

        if banner_component in banner_image_counts:

            component_type = banner_component  # logical mapping
            #nice
            print("hi")
            filtered = image_df[
                (image_df["store_types"].str.contains(store_type, case=False, na=False)) &
                (image_df["allowed_components"].str.contains(component_type, case=False, na=False))
            ]
            print(len(filtered), "filtered images ")
            if not filtered.empty:
                image_count = banner_image_counts[banner_component]
                # Create query embedding from the user description instead of using a random image
                from openai import OpenAI
                client = OpenAI()

                query_text = f"{description} store type {store_type} banner image"

                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=query_text,
                )

                query_embedding = response.data[0].embedding

                selected_rows = faiss_search(filtered, query_embedding, image_count)

                images = []

                for _, row in selected_rows.iterrows():
                    images.append(row["image_url"])
                #     or use this both means same
                # images = [row["image_url"] for _, row in selected_rows.iterrows()]

                # SINGLE BANNER → one image only
                if banner_component == "single_banner" or banner_component == "brand_product_slider":

                    component["images"] = {
                            "banner_url": images[0],
                            "banner_link_to": "collection",
                            "banner_link_value": "__COLLECTION__"
                        }

                else:
                    # Unified multi-image structure
                    component["images"] = [
                        {
                            "banner_url": img,
                            "banner_link_to": "collection",
                            "banner_link_value": "__COLLECTION__"
                        }
                        for img in images
                    ]

            continue

    return structure

def assign_products_to_layout(structure, description, store_type):
    print("Assigning products to layout")
    layout = structure.get("layout", [])
    description_lower = description.lower()

    # sortings = ["relevance","best selling" , "newest","low to high","high to low", "A to Z", "Z to A"]
    for component in layout:
        product_component = component.get("component")
        product_image_counts = {
            "product_slider_with_video":3,
        }

        if product_component in product_image_counts:
            component_type = product_component
            filtered = product_df[product_df["store_types"].str.contains(store_type, case=False, na=False)]
            print(len(filtered), "filtered products")

            if not filtered.empty:
                product_count = product_image_counts[product_component]

                # Try to match products with the selected video
                video_url = component.get("videos", [{}])[0].get("banner_url")

                if video_url:
                    video_row = video_df[video_df["video_url"] == video_url]

                    if not video_row.empty:
                        video_embedding = video_row.iloc[0]["embedding"]
                        selected_rows = faiss_search(filtered, video_embedding, product_count)
                    else:
                        selected_rows = filtered.sample(n=min(product_count, len(filtered)))

                else:
                    # fallback to description search
                    from openai import OpenAI
                    client = OpenAI()

                    query_text = f"{description} store type {store_type} product"

                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=query_text,
                    )

                    query_embedding = response.data[0].embedding
                    selected_rows = faiss_search(filtered, query_embedding, product_count)
                products = []

                for _, row in selected_rows.iterrows():
                    products.append(row["product_id"])
                #     or use this both means same
                # images = [row["image_url"] for _, row in selected_rows.iterrows()]


                # Unified multi-image structure
                component["products"] = [
                    {
                    "product_id": item
                    }
                    for item in products
                ]

                continue

    print(description_lower)
    return structure


class PromptRequest(BaseModel):
    description: str
    store_type: str
    max_sections:int | None = None

@app.post("/getstructure")
async def root(request: PromptRequest):
    description = request.description
    max_sections = request.max_sections or 29
    componentCatalog = components_df.to_dict(orient="records")
    # Build a compressed catalog — only what the model needs
    compressed_catalog = []
    for _, row in components_df.iterrows():
        config_value = row["config"]
        if pd.isna(config_value) or config_value == "":
            config_value = ""

        compressed_catalog.append({
            "name": row["component_name"],
            "type": row["component_type"],
            "config": config_value
        })

    catalog_str = json.dumps(compressed_catalog, separators=(',', ':'))  # no spaces at all
    system_prompt = f"""
    You are an AI Layout Planner for a mobile commerce app builder (Shopify-based).
    AND YOU ARE A JSON API. YOU OUTPUT ONLY RAW JSON. NOTHING ELSE.

    DO NOT write any explanation, greeting, description, markdown, or prose.
    DO NOT use ```json or ``` code blocks.
    DO NOT start with "Here is", "Sure", "Based on", or any words.
    YOUR ENTIRE RESPONSE MUST START WITH {{ AND END WITH }}

VIOLATION: Any response that is not pure JSON will be rejected.

    Your task: given a store description and type, output a complete home screen layout as a JSON object.

    ════════════════════════════════════════
    SECTION 1 — OUTPUT FORMAT (STRICT)
    ════════════════════════════════════════

    Return ONLY a valid JSON object. No markdown. Just no explanation. No extra text.

    {{
      "theme": "<light | dark | colorful>",
      "tone": "<modern | playful | luxury | minimal | bold>",
      "layout": [
        {{
          "component": "<exact component_name from catalog>",
          "settings": {{
            // populated fields — see Section 3
          }}
        }}
      ]
    }}

    ════════════════════════════════════════
    SECTION 2 — COMPONENT SELECTION RULES
    ════════════════════════════════════════

    RULE 1 — USE ONLY VALID COMPONENTS:
      Only use component_name values from COMPONENT_CATALOG below.
      Never invent new component names.

    RULE 2 — HONOR EXPLICIT REQUESTS:
      If the user explicitly names a component (e.g. "banner", "product slider",
      "brand product slider", "social links"), you MUST include it.
      Map user language to the closest component_name:
        "banner"               → single_banner OR banner_slider
        "video banner"         → video_banner
        "category"             → category_circle_slider OR rectangleCategorySlider
        "product slider"       → product_slider
        "brand product slider" → brand_product_slider 
        "video slider"         → video_slider
        "social links"/"socials" → social_links
        "search"               → search_bar
        "countdown"/"timer"    → timer

    RULE 3 — CONTROLLED REPEAT SECTIONS:
      Some components may repeat to create realistic ecommerce layouts.

      Allowed repetitions:
        product_slider → up to 4 times
        brand_product_slider → up to 2 times
        categoryWithProducts → up to 2 times

      These repetitions are especially common for grocery, electronics,
      and large catalog stores.

      Example:
        banner_slider
        category_circle_slider
        product_slider
        product_slider
        product_slider

      Do NOT repeat banners, announcement_bar, search_bar,product_slider_with_video or social_links.

    RULE 4 — SMART DEFAULTS BY STORE TYPE:
      If the user gives a vague description, build a sensible layout using these
      guidelines per store_type:

      fashion / apparel:
        → announcement_bar, banner_slider, bannerSliderClassic, single_banner,categoryWithProducts,
          category_circle_slider, rectangleCategorySlider,
          product_slider, brand_product_slider, productSliderWithVideo,
          threeLayoutBanner, twoGridLayoutBanner,
          paragraphWithCTA, text_paragraph,
          video_banner, video_slider,
          social_links

      electronics / tech:
        → search_bar, announcement_bar,
          banner_slider, bannerSliderClassic, single_banner,
          categoryWithProducts, rectangleCategorySlider,
          product_slider, brand_product_slider, productSliderWithVideo,
          timer, continuousBannerSlider, longBannerSlider,
          video_banner, video_slider,
          text_paragraph, paragraphWithCTA,
          social_links

      food / grocery:
        → announcement_bar, search_bar,
          single_banner, banner_slider,
          rectangleCategoryGrid, rectangleCategorySlider,
          categoryWithProducts,
          product_slider, brand_product_slider,
          paragraphWithCTA, text_paragraph,
          threeLayoutBanner, twoGridLayoutBanner,
          timer, continuousBannerSlider,
          social_links
        Layout preference:
          Grocery stores should be product-heavy. Prefer multiple product_slider
          sections (2–4) and at least one category navigation component.

      beauty / cosmetics:
        → announcement_bar,
          video_banner, video_slider, videoWithOneCTA,
          banner_slider, bannerSliderClassic, single_banner,
          category_circle_slider, rectangleCategorySlider,
          product_slider, brand_product_slider, productSliderWithVideo,
          paragraphWithCTA, text_paragraph,
          threeLayoutBanner,
          social_links

      general / other:
        → announcement_bar, search_bar,
          banner_slider, bannerSliderClassic, single_banner,
          category_circle_slider, rectangleCategorySlider, categoryWithProducts,
          product_slider, brand_product_slider,
          paragraphWithCTA, text_paragraph,
          threeLayoutBanner, twoGridLayoutBanner,
          video_banner, video_slider,
          timer, continuousBannerSlider,
          social_links, spacer

    RULE 5 — LAYOUT ORDER (top → bottom):
      Always arrange components in this priority order when applicable:
      1. announcement_bar (always first if used)
      2. search_bar
      3. Hero visual: video_banner / banner_slider / single_banner / bannerSliderClassic
      4. Category navigation: category_circle_slider / rectangleCategorySlider / categoryWithProducts
      5. Product sections: product_slider / brand_product_slider / categoryWithProducts
      6. Editorial/content: paragraphWithCTA / text_paragraph / threeLayoutBanner / twoGridLayoutBanner
      7. Promotional: timer / continuousBannerSlider / longBannerSlider
      8. Video content: video_slider / videoWithOneCTA / videoWithTwoCTA / productSliderWithVideo
      9. social_links (always last if used)
      10. spacer (use between major sections for breathing room)

    RULE 6 — SECTION COUNT:
      Minimum: 1 component.
      Maximum: {max_sections} components.

    ════════════════════════════════════════
    SECTION 3 — SETTINGS POPULATION RULES
    ════════════════════════════════════════

    For every component in the layout, populate its settings object using the
    component's config schema from COMPONENT_CATALOG  only if component's respective config is there.
    Do not use other components setting in each other.
    DO NOT MISS ANY SETTINGS IF PRESENT 
    DO NOT ADD EXTRA SETTINGS
    
    For title and description and text:
    TEXT FIELDS (titles, taglines, descriptions, CTA labels)
      → Generate real, high-quality marketing copy suited to the store_type and description.
      → Example for fashion store: title → "Discover the New Season", ctaText → "Shop Now"
      → NEVER use the config field description as the value (e.g. never output "Main headline text").
    
    For sortBy choose ONE value from the following options:
    "relevance", "best selling", "newest", "low to high", "high to low", "A to Z", "Z to A".
    Select the option that best fits the component and store type.
    
    For productSelection 
    For product_slider_with_video in settings the productSelection will be manual
    BOOLEAN / TOGGLE FIELDS (isTitleOn, isCtaOn, isTitleOn, animation, etc.):
      → Set to true unless context suggests it should be hidden.

    URL / MEDIA FIELDS — use ONLY these exact placeholders:
      → Image URL fields      → "__IMAGE__"
      → Video URL fields      → "__VIDEO__"
      → Product array fields  → "__PRODUCTS__"
      → Collection ID fields  → "__COLLECTION__"

    LINK FIELDS (linkTo, bannerLinkTo, ctaLinkTo):
      → Use one of: "collection" | "product" | "page" | "external"

    LINK VALUE FIELDS (linkValue, bannerLinkValue, ctaLinkValue):
      → Use "__COLLECTION__" for collection links, "__PRODUCTS__" for product links.

    SIZE / ENUM FIELDS:
      → spacer size: DEFAULT 16
      → video_banner size: "large" for hero use, "medium" otherwise
    Strict Rule :
    IF A COMPONENT HAS AN EMPTY CONFIG (config = ""):
      → Output an empty settings object: "settings": {{}}

    ════════════════════════════════════════
    COMPONENT_CATALOG:
    ════════════════════════════════════════
    {catalog_str}
    """

    max_retries = 2
    attempt = 0
    parsed_json = None
    content = None
    number_of_components = 0

    valid_components = set(components_df["component_name"].tolist())

    while attempt <= max_retries:
        response = ollama.chat(
            model="llama3:latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": description}
            ],
            keep_alive="5m",
            options={
                "temperature": 0.75,  # lower = faster + more deterministic = fewer retries
                "num_predict": 2048,  # cap max output tokens
                "top_p": 0.9,
            }
        )

        content = response["message"]["content"]

        import re
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            attempt += 1
            continue

        json_string = match.group(0)

        try:
            parsed_json = json.loads(json_string)

            layout = parsed_json.get("layout", [])

            # ✅ Remove invalid components automatically

            filtered_layout = []

            for section in layout:
                component_name = section.get("component") or section.get("component_name")

                if component_name in ["brand_product_slider", "categoryWithProducts"]:
                    collection_id, collection_name = get_random_collection()

                    section["collection_id"] = collection_id
                    section["collection_name"] = collection_name
                if component_name in valid_components:

                    config = components_df.loc[
                        components_df["component_name"] == component_name
                        ]["config"].values[0]

                    # If component has no config
                    if pd.isna(config) or config == "":
                        section["settings"] = {}
                    else:
                        allowed_settings = set(json.loads(config).keys())

                        settings = section.get("settings", {})

                        # Remove extra settings from LLM
                        cleaned_settings = {
                            k: v for k, v in settings.items() if k in allowed_settings
                        }

                        section["settings"] = cleaned_settings

                    filtered_layout.append(section)

            parsed_json["layout"] = filtered_layout
            # ✅ Map images dynamically
            jsonType = type(parsed_json)
            print("json type ", jsonType)
            parsed_json = assign_images_to_layout(
                parsed_json,
                description,
                request.store_type
            )
            parsed_json = assign_products_to_layout(parsed_json, description, request.store_type)
            parsed_json = assign_videos_to_layout(parsed_json,description,request.store_type)


            # print(parsed_json, "parsed_json is ")
            number_of_components = len(filtered_layout)
            break  # Success

        except Exception:
            attempt += 1
            continue

    if parsed_json is None:
        return {
            "error": "Failed to generate valid structure after retries",
            "raw_output": content
        }

    return {"structure": parsed_json.get("layout", []), "components_fetched": number_of_components}

    # return {"message": request.prompt}

# Press ⌃F5 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
