from src.data_api.api_sentinel import SentinelHubS2ImageryAPI

api = SentinelHubS2ImageryAPI(
    mode="available_only",             # or "daily_all"
    min_interval_sec=0.25              # gentle pacing
)

# Hòn Mun default AOI is baked in; you can override with bbox_4326=[minx,miny,maxx,maxy]
df = api.fetch_daily(location=(12.16764, 109.30581))

print(df.head())
# Columns: date, collection, mime, bytes, path, crs, bbox, width, height, status
