import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"

INTERNAL_DOMAINS = {"dtaa.com"}


def email_features():
    print("Processing email features...")

    path = PROCESSED / "cleaned_email.csv"
    results = []

    columns = [
        "date", "user", "to", "cc", "bcc", "attachments"
    ]

    for chunk in pd.read_csv(
        path,
        usecols=columns,
        chunksize=50000
    ):
        chunk["day"] = pd.to_datetime(
            chunk["date"]
        ).dt.normalize()

        for col in ["to", "cc", "bcc"]:
            chunk[col] = chunk[col].fillna("")

        recipients = (
            chunk["to"] + ";" +
            chunk["cc"] + ";" +
            chunk["bcc"]
        )

        def has_external(addresses):
            for address in addresses.split(";"):
                address = address.strip().lower()

                if "@" not in address:
                    continue

                domain = address.rsplit("@", 1)[-1]

                if domain not in INTERNAL_DOMAINS:
                    return 1

            return 0

        chunk["external_email"] = recipients.map(
            has_external
        )

        chunk["external_attachments"] = (
            pd.to_numeric(
                chunk["attachments"],
                errors="coerce"
            ).fillna(0)
            * chunk["external_email"]
        )

        grouped = chunk.groupby(
            ["user", "day"]
        ).agg(
            external_email_count=(
                "external_email", "sum"
            ),
            external_attachment_count=(
                "external_attachments", "sum"
            )
        ).reset_index()

        results.append(grouped)

    result = pd.concat(results)

    result = result.groupby(
        ["user", "day"],
        as_index=False
    ).sum()

    return result


def pc_features():
    print("Processing PC features...")

    path = PROCESSED / "cleaned_logon.csv"

    df = pd.read_csv(
        path,
        usecols=["date", "user", "pc", "activity"]
    )

    df["day"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    # Count actual logons, not logoffs.
    df = df[
        df["activity"].str.lower() == "logon"
    ].copy()

    # One record per employee, day, and PC.
    df = df.drop_duplicates(
        ["user", "day", "pc"]
    )

    # Sort chronologically to identify first use.
    df = df.sort_values(
        ["user", "day", "pc"]
    )

    # Mark a PC as new only on its first observed day.
    df["new_pc"] = (
        ~df.duplicated(["user", "pc"])
    ).astype(int)

    result = df.groupby(
        ["user", "day"]
    ).agg(
        new_pc_count=("new_pc", "sum"),
        unique_pc_count=("pc", "nunique")
    ).reset_index()

    return result


def main():
    print("Loading existing behavioral features...")

    base = pd.read_csv(
        PROCESSED / "user_daily_features.csv"
    )

    base["day"] = pd.to_datetime(base["day"])

    emails = email_features()
    pcs = pc_features()

    print("Merging advanced features...")

    result = base.merge(
        emails,
        on=["user", "day"],
        how="left"
    )

    result = result.merge(
        pcs,
        on=["user", "day"],
        how="left"
    )

    new_columns = [
        "external_email_count",
        "external_attachment_count",
        "new_pc_count",
        "unique_pc_count"
    ]

    result[new_columns] = (
        result[new_columns].fillna(0)
    )

    output = (
        PROCESSED /
        "user_daily_features_advanced.csv"
    )

    result.to_csv(output, index=False)

    print("\nAdvanced feature engineering completed!")
    print("Employee-days:", len(result))
    print("Total columns:", len(result.columns))
    print("Saved to:", output)


if __name__ == "__main__":
    main()