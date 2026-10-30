-- Reference rows that every environment needs (idempotent: run after 01_schema.sql).
USE power_db;

INSERT INTO households (household_id, name, description, data_source) VALUES
 (1, 'Household 1',
  'Single household whose power was metered every minute (Dec 2006 - Nov 2010)',
  'UCI Individual household electric power consumption, hourly aggregate from Task 1 (hourly_observed.csv)');

-- Descriptions follow the public UCI dataset documentation (verify against your source before citing).
INSERT INTO circuits (circuit_id, household_id, code, name, description, unit, is_derived) VALUES
 (1, 1, 'sub_metering_1', 'Kitchen',            'Dishwasher, oven, microwave',                     'Wh', 0),
 (2, 1, 'sub_metering_2', 'Laundry room',       'Washing machine, tumble-drier, refrigerator, light', 'Wh', 0),
 (3, 1, 'sub_metering_3', 'Water heater & A/C', 'Electric water-heater and air-conditioner',       'Wh', 0),
 (4, 1, 'other',          'Other (not sub-metered)', 'Derived: hourly active energy (kW x 1000) minus the three sub-meters', 'Wh', 1);
