import pandas as pd
from fastapi import  FastAPI

app = FastAPI()

import pandas as pd

df = pd.read_csv("csvsheets/video_data.csv")

def map_video_collection(row):

    desc = row["video_description"].lower()
    tags = str(row["tags"]).lower()
    store = row["store_types"]

    if store == "fashion":

        if "kids" in tags:
            return ("gid://shopify/Collection/10006","Kidswear")

        if "coat" in desc or "winter" in desc:
            return ("gid://shopify/Collection/10002","Winter Collection")

        if "denim" in desc:
            return ("gid://shopify/Collection/10005","Denim Collection")

        if any(x in desc for x in ["shoe","sneaker","boots"]):
            return ("gid://shopify/Collection/10004","Sneakers and Shoes")

        if any(x in desc for x in ["belt","watch","handbag","accessories","sunglasses","scarf"]):
            return ("gid://shopify/Collection/10003","Fashion Accessories")

        return ("gid://shopify/Collection/10001","Formal Wear")


    if store == "grocery":

        if any(x in desc for x in ["fruit","vegetable","produce","fresh","market"]):
            return ("gid://shopify/Collection/10007","Fresh Produce")

        return ("gid://shopify/Collection/10008","Grocery Essentials")


    if store == "electronics":

        if any(x in desc for x in ["smart","watch","hub","camera","fitness","smartphone"]):
            return ("gid://shopify/Collection/10009","Smart Electronics")

        return ("gid://shopify/Collection/10010","Computer Accessories")


df[["collection_id","collection_name"]] = df.apply(
    lambda r: pd.Series(map_video_collection(r)), axis=1
)

df.to_csv("videos_with_collections.csv", index=False)