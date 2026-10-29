"""Keep disposable commit metadata tied to its configured UTC cutoff."""

from datetime import datetime, timedelta, timezone


def commit_dates(configuration):
    cutoff = datetime.fromisoformat(
        configuration["schedule"]["submission_cutoff"].replace("Z", "+00:00")
    )
    evaluation_day = cutoff.astimezone(timezone.utc).date()
    return evaluation_day, evaluation_day - timedelta(days=1)
