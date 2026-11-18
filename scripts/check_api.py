"""Send live requests through the API for every advertised trip."""
import argparse
import sys

import httpx


def verify(url):
    with httpx.Client(base_url=url, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        listing = client.get("/trips")
        listing.raise_for_status()
        ids = listing.json()["trips"]
        print(f"GET /trips → {listing.status_code} · {len(ids)} trips · {listing.json()['mode']}", flush=True)
        assert len(ids) == len(set(ids)) == 10
        bad = client.get("/match/123456789")
        assert bad.status_code == 404
        for trip_id in ids:
            response = client.get(f"/match/{trip_id}")
            response.raise_for_status()
            payload = response.json()
            arc_ids = payload["matched_arc_ids"]
            assert payload["trip_id"] == trip_id
            assert len(payload["gps_latlon"]) == len(arc_ids) == payload["observations_summary"]["count"]
            assert arc_ids and min(arc_ids) >= 0
            assert payload["route_latlon"] and all(len(run) >= 2 for run in payload["route_latlon"])
            assert set(payload) == {
                "trip_id", "mode", "model", "observations_summary", "inference_seconds",
                "evaluation", "matched_arc_ids", "gps_latlon", "route_latlon", "nearby_roads_latlon",
            }
            evaluation = payload["evaluation"]
            assert evaluation and 0.0 <= evaluation["sequence_f1"] <= 1.0
            accuracy = evaluation["top1_accuracy"]
            metric = "N/A" if accuracy is None else f"{accuracy:.1%}"
            print(f"GET /match/{trip_id} → {response.status_code} · {len(arc_ids):,} fixes"
                  f" · Top-1 {metric} · {payload['inference_seconds']:.3f}s", flush=True)
    print("All 10 live trip requests passed.", file=sys.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    verify(parser.parse_args().url)
