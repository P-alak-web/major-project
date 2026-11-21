import pandas as pd
from pathlib import Path
from urllib.parse import urlsplit
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"

HTTP_FILE = PROCESSED / "cleaned_http.csv"
OUTPUT_FILE = PROCESSED / "http_daily_features.csv"

# Illustrative domain categories. These may need adapting
# to the domains actually present in CERT r4.2.
CLOUD_DOMAINS = {
    "dropbox.com",
    "drive.google.com",
    "onedrive.live.com",
    "box.com",
    "mega.nz",
}

JOB_DOMAINS = {
    "linkedin.com",
    "indeed.com",
    "monster.com",
    "naukri.com",
}

seen_domains = defaultdict(set)
daily_stats = defaultdict(lambda: [0, 0, 0, 0])
daily_unique = defaultdict(set)


def extract_domain(url):
    if pd.isna(url):
        return ""

    try:
        hostname = urlsplit(str(url)).hostname
        return (hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def matches_domain(domain, targets):
    return any(
        domain == target or domain.endswith("." + target)
        for target in targets
    )


def main():
    print("Processing HTTP behavioral features...", flush=True)

    processed_rows = 0

    for chunk in pd.read_csv(
        HTTP_FILE,
        usecols=["date", "user", "url"],
        chunksize=100000
    ):
        chunk["day"] = chunk["date"].str[:10]
        chunk["domain"] = chunk["url"].map(extract_domain)

        # One employee/domain/day combination.
        unique_visits = chunk[
            ["user", "day", "domain"]
        ].drop_duplicates()

        for row in unique_visits.itertuples(index=False):
            user, day, domain = row

            if not domain:
                continue

            key = (user, day)

            # Count a domain only once per employee-day.
            if domain in daily_unique[key]:
                continue

            daily_unique[key].add(domain)

            # First observed use of this domain by this employee.
            if domain not in seen_domains[user]:
                daily_stats[key][1] += 1
                seen_domains[user].add(domain)

        # Count individual visits to predefined categories.
        for row in chunk[
            ["user", "day", "domain"]
        ].itertuples(index=False):
            user, day, domain = row

            key = (user, day)

            if matches_domain(domain, CLOUD_DOMAINS):
                daily_stats[key][2] += 1

            if matches_domain(domain, JOB_DOMAINS):
                daily_stats[key][3] += 1

        processed_rows += len(chunk)

        if processed_rows % 1000000 < 100000:
            print(
                f"Processed {processed_rows:,} HTTP records",
                flush=True
            )

    records = []

    for (user, day), stats in daily_stats.items():
        records.append({
            "user": user,
            "day": day,
            "unique_domains": len(daily_unique[(user, day)]),
            "new_domain_count": stats[1],
            "cloud_storage_visits": stats[2],
            "job_site_visits": stats[3],
        })

    result = pd.DataFrame(records)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nHTTP feature engineering completed!")
    print("Records processed:", processed_rows)
    print("Employee-days:", len(result))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()