
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def generate_ground_truth(answers_dir: Path, output_dir: Path):
    insiders_file = answers_dir / "insiders.csv"

    if not insiders_file.is_file():
        raise FileNotFoundError(f"Missing file: {insiders_file}")

    events = []
    user_days = defaultdict(set)
    insider_count = 0

    with insiders_file.open("r", encoding="utf-8-sig", newline="") as f:
        insider_rows = csv.DictReader(f)

        for insider in insider_rows:
            if insider["dataset"].strip() != "4.2":
                continue

            insider_count += 1
            user = insider["user"].strip()
            scenario = insider["scenario"].strip()

            detail_file = (
                answers_dir
                / f"r4.2-{scenario}"
                / insider["details"].strip()
            )

            if not detail_file.is_file():
                raise FileNotFoundError(
                    f"Missing detail file: {detail_file}"
                )

            with detail_file.open(
                "r", encoding="utf-8-sig", newline=""
            ) as detail:
                reader = csv.reader(detail)

                for line_number, row in enumerate(reader, start=1):
                    if len(row) < 4:
                        raise ValueError(
                            f"Invalid row at {detail_file}:{line_number}"
                        )

                    activity_type = row[0].strip()
                    event_id = row[1].strip()
                    timestamp = row[2].strip()
                    event_user = row[3].strip()

                    if event_user != user:
                        continue

                    event_date = datetime.strptime(
                        timestamp, "%m/%d/%Y %H:%M:%S"
                    ).date().isoformat()

                    events.append({
                        "event_id": event_id,
                        "user": user,
                        "date": event_date,
                        "timestamp": timestamp,
                        "activity_type": activity_type,
                        "scenario": scenario,
                        "label": 1,
                    })

                    user_days[(user, event_date)].add(scenario)

    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "ground_truth_events.csv"
    with events_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "event_id", "user", "date", "timestamp",
            "activity_type", "scenario", "label"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)

    days_path = output_dir / "ground_truth_user_day.csv"
    with days_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["user", "date", "scenario", "label"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for (user, date), scenarios in sorted(user_days.items()):
            writer.writerow({
                "user": user,
                "date": date,
                "scenario": ";".join(sorted(scenarios)),
                "label": 1,
            })

    print(f"Insider entries processed: {insider_count}")
    print(f"Labeled events: {len(events)}")
    print(f"Positive user-days: {len(user_days)}")
    print(f"Event labels saved: {events_path}")
    print(f"User-day labels saved: {days_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CERT r4.2 ground-truth labels"
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        required=True,
        help="Folder containing insiders.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ground_truth"),
    )

    args = parser.parse_args()
    generate_ground_truth(args.answers_dir, args.output_dir)
