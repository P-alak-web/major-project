import pandas as pd
from urllib.parse import urlparse

http = pd.read_csv("processed/cleaned_http.csv")

http["domain"] = http["url"].apply(
    lambda x: urlparse(str(x)).netloc.lower()
)

print("\n--- URL INSPECTION ---")

print("\nSample URLs and domains:")
print(http[["url", "domain"]].head(20).to_string(index=False))

print("\nURLs with paths:")
print(
    http[http["url"].apply(
        lambda x: bool(urlparse(str(x)).path.strip("/"))
    )][["url", "domain"]].head(10).to_string(index=False)
)

print("\nTotal unique URLs:", http["url"].nunique())
print("Total unique domains:", http["domain"].nunique())

print("\nMost frequently visited domains:")
print(http["domain"].value_counts().head(10))