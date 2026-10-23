"""
Project configuration: paths, model parameters, business parameters
and ALL assumptions used for synthetic data.

Every parameter marked [SYNTHETIC] does not come from the public
'Hotel Booking Demand' dataset - it is an assumption that should be replaced
with the hotel's own data (PMS, finance, revenue management) in a real deployment.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"
DAILY = OUTPUTS / "daily"
SITE = OUTPUTS / "site"            # static website for online publishing (index.html + archive)
LOGS = ROOT / "logs"

# Public data source (Antonio, Almeida, Nunes, 2019, Data in Brief,
# CC BY 4.0 licence; cleaned version from the TidyTuesday repository).
DATA_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-02-11/hotels.csv"
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Time-based split (by arrival date) - NEVER a random split
# ---------------------------------------------------------------------------
TRAIN_END = "2016-06-30"   # training: arrivals 2015-07-01 .. 2016-06-30
VALID_END = "2016-12-31"   # validation + calibration: 2016-07-01 .. 2016-12-31
# test (backtest): arrivals 2017-01-01 .. 2017-08-31

# Snapshots: how many days before arrival we "take a picture" of the booking state.
SNAPSHOT_HORIZONS = [0, 1, 3, 7, 14, 30, 60, 90, 120, 180]

# Features that must NOT be used (information leakage from the future).
# - country: in this dataset the country is filled in at check-in, so cancelled
#   bookings show 'PRT' unnaturally often (a known artefact).
# - booking_changes, assigned_room_type, required_car_parking_spaces,
#   total_of_special_requests: final values, known only after the fact
#   or updated during the life of the booking.
LEAKY_COLUMNS = [
    "country", "booking_changes", "assigned_room_type",
    "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date", "is_canceled",
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=300,
    cat_smooth=50,
    min_data_per_group=200,
    cat_l2=10,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=RANDOM_STATE,
)
LGBM_NUM_ROUNDS = 2000
LGBM_EARLY_STOPPING = 100
TOP_AGENTS = 40          # agents outside the top-N -> 'other'

# ---------------------------------------------------------------------------
# Hotel capacity
# ---------------------------------------------------------------------------
# The dataset does not contain the number of rooms. None = estimate it as the
# maximum realised occupancy (the data clearly hits a fixed ceiling).
# In a real deployment: enter the room count from the PMS, e.g. {"City Hotel": 230}.
CAPACITY_OVERRIDE = None

# Window of nights used for analysis (the first days of July 2015 lack the full
# history of stays that began before the start of the dataset).
NIGHTS_START = "2015-07-15"
NIGHTS_END = "2017-08-31"

# ---------------------------------------------------------------------------
# Overbooking - business parameters  [SYNTHETIC]
# ---------------------------------------------------------------------------
DECISION_HORIZON_DAYS = 3        # decision on extra sales at D-3
MAX_EXTRA_BOOKINGS = 40          # maximum number of extra bookings per night

# Cost of relocating a guest ("walk"): room in another hotel, transfer,
# compensation, loss of loyalty. Expressed relative to that night's ADR.
WALK_COST_ADR_MULTIPLIER = 2.0   # [SYNTHETIC]
WALK_COST_FIXED_EUR = 150.0      # [SYNTHETIC]

# Average ancillary spend per occupied room-night (F&B, bar, spa etc.)
ANCILLARY_PER_ROOM_NIGHT = {     # [SYNTHETIC]
    "Resort Hotel": 35.0,
    "City Hotel": 20.0,
}

# Unconstrained demand (turned-away requests) on sold-out nights.
# The dataset only contains accepted bookings, so turned-away demand is
# generated synthetically: Poisson(lambda * capacity) on nights where
# realised occupancy >= SOLD_OUT_THRESHOLD * capacity.
SOLD_OUT_THRESHOLD = 0.95        # [SYNTHETIC]
EXTRA_DEMAND_LAMBDA = 0.10       # [SYNTHETIC] 10% of capacity

# Benchmark policy: static overbooking (as used by many hotels)
STATIC_OVERBOOKING_PCT = 0.05

# Look-back window for the "pickup" forecast (bookings still to come)
PICKUP_LOOKBACK_WEEKS = 8

# ---------------------------------------------------------------------------
# Preventive actions (outreach) - A/B experiment  [SYNTHETIC]
# ---------------------------------------------------------------------------
OUTREACH_HORIZON_DAYS = 14       # guest contact at D-14
OUTREACH_TREATMENT_SHARE = 0.5   # half treatment, half control
OUTREACH_MAX_EFFECT = 0.30       # [SYNTHETIC] max. relative reduction in cancellations
OUTREACH_COST_PER_CONTACT = 1.0  # [SYNTHETIC] EUR (automated e-mail + handling)
OUTREACH_DISCOUNT_PCT = 0.05     # [SYNTHETIC] discount for prepayment
OUTREACH_DISCOUNT_TAKE_UP = 0.15 # [SYNTHETIC] share of guests accepting the discount
# Probability that a released room would have been resold anyway
RESALE_PROBABILITY = 0.50        # [SYNTHETIC]

# ---------------------------------------------------------------------------
# Cancellation / deposit policies - "what-if" simulation  [SYNTHETIC]
# ---------------------------------------------------------------------------
POLICY_CANCEL_REDUCTION = 0.40   # a prepayment requirement reduces cancellations by 40%
POLICY_BOOKING_LOSS = 0.08       # ... but deters 8% of bookers
POLICY_EARLY_RESALE = 0.80       # [SYNTHETIC] a room released early usually gets resold
POLICY_REPLACEMENT = 0.50        # [SYNTHETIC] a lost booking is partly replaced by another guest
POLICY_MIN_BOOKINGS = 500        # min. number of bookings in a segment

# ---------------------------------------------------------------------------
# Business case  [SYNTHETIC]
# ---------------------------------------------------------------------------
PROJECT_COST_EUR = 30000.0       # 1 person x ~3 months
RUN_COST_EUR_PER_YEAR = 6000.0   # maintenance, monitoring, retraining

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
PSI_WARN = 0.10
PSI_ALERT = 0.25

# ---------------------------------------------------------------------------
# "Today" for the daily-run demonstration
# (must lie within the test period of the dataset)
# ---------------------------------------------------------------------------
DEMO_AS_OF = "2017-06-15"
FORWARD_DAYS = 60

# "Replay" demo mode for scheduled runs: the public data ends on 31 Aug 2017,
# so each real calendar day is mapped to the next day of the test period.
# The window ends 60 days before the end of the data so the 60-night plan is complete.
REPLAY_START = "2017-01-15"
REPLAY_END = "2017-07-02"
REPLAY_ANCHOR = "2026-01-01"       # real date corresponding to REPLAY_START
ARCHIVE_DAYS = 30                  # how many recent reports to list in the site archive

# ---------------------------------------------------------------------------
# Palette (validated for colour-vision deficiency)
# ---------------------------------------------------------------------------
COLORS = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text2": "#52514e",
    "grid": "#e4e3df",
    "s1": "#2a78d6",   # blue
    "s2": "#eb6834",   # orange
    "s3": "#1baf7a",   # aqua
    "good": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
    "neutral": "#b9b8b3",
    "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
}
HOTEL_COLORS = {"City Hotel": COLORS["s1"], "Resort Hotel": COLORS["s2"]}
