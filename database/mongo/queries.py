"""MongoDB query catalogue (find + aggregation pipelines).

Each entry is plain data so it can be pasted into MongoDB Compass / mongosh as-is
(see database/results/mongo_query_results.md for the JSON form of every pipeline).
"""
from datetime import datetime

HH = "hh_001"

QUERIES = [
    {
        "name": "M1_latest_reading",
        "desc": "Most recent hourly document (uses the unique {household_id, timestamp} index, sort descending, limit 1).",
        "collection": "hourly_readings", "op": "find",
        "filter": {"household_id": HH},
        "sort": [("timestamp", -1)], "limit": 1,
    },
    {
        "name": "M1b_latest_valid_reading",
        "desc": "Most recent hour that actually contains measurements (the very last hour of the file is only partly observed, so its values are null).",
        "collection": "hourly_readings", "op": "find",
        "filter": {"household_id": HH, "quality.is_valid_hour": True},
        "sort": [("timestamp", -1)], "limit": 1,
    },
    {
        "name": "M2_date_range_summary",
        "desc": "Summary of all valid hours between 2008-01-01 and 2008-01-31 (range scan on the compound index, then $group).",
        "collection": "hourly_readings", "op": "aggregate",
        "pipeline": [
            {"$match": {"household_id": HH, "timestamp": {"$gte": datetime(2008, 1, 1), "$lt": datetime(2008, 2, 1)},
                        "quality.is_valid_hour": True}},
            {"$group": {"_id": None, "valid_hours": {"$sum": 1},
                        "energy_kwh": {"$sum": "$measurements.global_active_power_kw"},
                        "avg_kw": {"$avg": "$measurements.global_active_power_kw"},
                        "peak_kw": {"$max": "$measurements.global_active_power_kw"},
                        "avg_voltage": {"$avg": "$measurements.voltage_v"}}},
            {"$project": {"_id": 0, "valid_hours": "$valid_hours",
                          "energy_kwh": {"$round": ["$energy_kwh", 1]}, "avg_kw": {"$round": ["$avg_kw", 3]},
                          "peak_kw": {"$round": ["$peak_kw", 3]}, "avg_voltage": {"$round": ["$avg_voltage", 1]}}},
        ],
    },
    {
        "name": "M3_monthly_energy",
        "desc": "Monthly energy (kWh), average and peak load - the same question as SQL Q2, answered with $group on the embedded calendar sub-document.",
        "collection": "hourly_readings", "op": "aggregate",
        "pipeline": [
            {"$match": {"quality.is_valid_hour": True}},
            {"$group": {"_id": {"yr": "$calendar.year", "mon": "$calendar.month"},
                        "valid_hours": {"$sum": 1},
                        "energy_kwh": {"$sum": "$measurements.global_active_power_kw"},
                        "avg_kw": {"$avg": "$measurements.global_active_power_kw"},
                        "peak_hour_kw": {"$max": "$measurements.global_active_power_kw"}}},
            {"$sort": {"_id.yr": 1, "_id.mon": 1}},
            {"$limit": 15},
            {"$project": {"_id": 0, "yr": "$_id.yr", "mon": "$_id.mon", "valid_hours": "$valid_hours",
                          "energy_kwh": {"$round": ["$energy_kwh", 1]}, "avg_kw": {"$round": ["$avg_kw", 3]},
                          "peak_hour_kw": {"$round": ["$peak_hour_kw", 3]}}},
        ],
    },
    {
        "name": "M4_avg_load_by_hour_weekday_vs_weekend",
        "desc": "Average daily load profile for weekdays vs weekends (group by hour and is_weekend).",
        "collection": "hourly_readings", "op": "aggregate",
        "pipeline": [
            {"$match": {"quality.is_valid_hour": True}},
            {"$group": {"_id": {"hour": "$calendar.hour", "weekend": "$calendar.is_weekend"},
                        "avg_kw": {"$avg": "$measurements.global_active_power_kw"}}},
            {"$group": {"_id": "$_id.hour",
                        "pairs": {"$push": {"k": {"$cond": ["$_id.weekend", "avg_kw_weekend", "avg_kw_weekday"]},
                                            "v": {"$round": ["$avg_kw", 3]}}}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "hour_of_day": "$_id", "profile": {"$arrayToObject": "$pairs"}}},
            {"$project": {"hour_of_day": 1, "avg_kw_weekday": "$profile.avg_kw_weekday", "avg_kw_weekend": "$profile.avg_kw_weekend"}},
        ],
    },
    {
        "name": "M5_energy_share_by_circuit",
        "desc": "Share of total energy per circuit, summing the fields of the embedded sub_metering_wh document.",
        "collection": "hourly_readings", "op": "aggregate",
        "pipeline": [
            {"$match": {"quality.is_valid_hour": True}},
            {"$group": {"_id": None,
                        "Kitchen": {"$sum": "$sub_metering_wh.kitchen"},
                        "Laundry room": {"$sum": "$sub_metering_wh.laundry_room"},
                        "Water heater & A/C": {"$sum": "$sub_metering_wh.water_heater_ac"},
                        "Other (not sub-metered)": {"$sum": "$sub_metering_wh.other"}}},
            {"$project": {"_id": 0, "circuits": {"$objectToArray": "$$ROOT"}}},
            {"$unwind": "$circuits"},
            {"$match": {"circuits.k": {"$ne": "_id"}}},          # drop the _id key produced by $group
            {"$group": {"_id": None, "rows": {"$push": {"circuit": "$circuits.k", "wh": "$circuits.v"}},
                        "total": {"$sum": "$circuits.v"}}},
            {"$unwind": "$rows"},
            {"$project": {"_id": 0, "circuit": "$rows.circuit", "total_kwh": {"$round": [{"$divide": ["$rows.wh", 1000]}, 1]},
                          "pct_of_total": {"$round": [{"$multiply": [{"$divide": ["$rows.wh", "$total"]}, 100]}, 2]}}},
            {"$sort": {"total_kwh": -1}},
        ],
    },
    {
        "name": "M6_top10_peak_hours",
        "desc": "The ten highest-consumption hours (matches SQL Q5).",
        "collection": "hourly_readings", "op": "aggregate",
        "pipeline": [
            {"$match": {"quality.is_valid_hour": True}},
            {"$sort": {"measurements.global_active_power_kw": -1}},
            {"$limit": 10},
            {"$project": {"_id": 0, "timestamp": "$timestamp", "kw": {"$round": ["$measurements.global_active_power_kw", 3]},
                          "voltage_v": {"$round": ["$measurements.voltage_v", 1]},
                          "water_heater_ac_wh": {"$round": ["$sub_metering_wh.water_heater_ac", 0]}}},
        ],
    },
    {
        "name": "M7_daily_summaries_top_days",
        "desc": "Five most energy-hungry days, answered instantly from the pre-aggregated daily_summaries collection (no scan of 34k hourly docs).",
        "collection": "daily_summaries", "op": "find",
        "filter": {"household_id": HH, "valid_hours": 24},
        "projection": {"_id": 0, "date": 1, "energy_kwh": 1, "mean_kw": 1, "max_kw": 1, "peak_hour": 1},  # (stored field order)
        "sort": [("energy_kwh", -1)], "limit": 5,
    },
    {
        "name": "M8_missing_hours_per_year",
        "desc": "Data-quality query: how many invalid (missing) hours per year.",
        "collection": "hourly_readings", "op": "aggregate",
        "pipeline": [
            {"$group": {"_id": "$calendar.year", "hours": {"$sum": 1},
                        "invalid_hours": {"$sum": {"$cond": ["$quality.is_valid_hour", 0, 1]}}}},
            {"$sort": {"_id": 1}},
            {"$project": {"_id": 0, "year": "$_id", "hours": "$hours", "invalid_hours": "$invalid_hours",
                          "pct_invalid": {"$round": [{"$multiply": [{"$divide": ["$invalid_hours", "$hours"]}, 100]}, 2]}}},
        ],
    },
]
